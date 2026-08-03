# 第 12 章：Host-Device 流水线、准入控制与系统优化证据

## 12.1 Kernel 之外的 GPU 空闲区间

Decode step 可能只运行很短的 kernel。若 CPU 还在构造 token tensor、等待队列锁、执行同步 copy，GPU 就会在两轮之间出现空洞。

另一类问题更隐蔽：服务轻载时一切正常，高并发时仍继续接受所有请求，队列越来越长，TTFT 和超时率一起上升。GPU 看起来一直满载，用户却几乎都无法满足 SLO。

本章讨论模型算子之外的两类系统问题：

```text
数据路径：CPU 准备与 H2D copy 能否和 GPU 计算重叠？
容量保护：系统是否只接纳自己真正能及时完成的请求？
```

---

## 12.2 Decode 迭代中的 Host 工作

即使模型权重和 KV cache 已在 GPU，每轮仍可能由 CPU 准备：

```text
最新 token IDs
position IDs
context lengths
block tables
slot mappings
sampling metadata
batch offsets
```

这些小 tensor 要传到 GPU。单次 bytes 不大，但 decode 轮数多、kernel 短，固定开销占比会上升。

```mermaid
flowchart LR
    S["Scheduler on CPU"] --> M["构造 metadata tensors"]
    M --> H["Host-to-Device copy"]
    H --> G["GPU decode kernels"]
    G --> R["结果 / next tokens"]
    R --> S
```

优化不能只看某次 copy 几微秒，而要看它是否造成 GPU 等待，以及在每 token 循环中累计多少。

## 12.3 Pageable Memory 与 Pinned Memory

普通 CPU 内存由操作系统虚拟内存管理，物理页可能被换出或移动。GPU DMA 传输需要稳定的物理页地址。

从 pageable memory 向 GPU copy 时，runtime 可能先把数据复制到临时 pinned staging buffer，再发起 DMA。[PyTorch 的 pinned memory 与 `non_blocking` 教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)分别测量了 pageable/pinned 路径，并强调 pinning 本身也有 host 端开销：

```text
pageable host
-> pinned staging copy
-> DMA to device
```

如果源本身是 pinned/page-locked memory：

```text
pinned host
-> DMA to device
```

这能减少 staging，并为真正异步 H2D 提供条件。

### 12.3.1 Pinned Memory 的系统代价

页锁定内存不能被系统随意换出。过量使用会减少操作系统可灵活管理的内存，影响整机性能。

因此不应对所有临时对象无界 `pin_memory()`。更合理的是：

- 复用有限 pinned buffers；
- 对高频 H2D 输入使用；
- 监控 host memory；
- 在 workload 下验证收益。

## 12.4 异步 Host-to-Device 传输的成立条件

```python
gpu = cpu.to("cuda", non_blocking=True)
```

想要 copy 与 compute 重叠，至少需要：

```text
源内存适合异步 DMA（通常 pinned）
copy 使用可并发的 stream
硬件支持 copy/compute overlap
数据依赖允许重叠
没有紧随其后的 host/device 同步
目标 tensor 生命周期正确
```

若 copy 后 default stream 立刻使用该 tensor，就必须等待数据就绪。代码调用是 non-blocking 的，不代表 GPU 时间线发生有效重叠。

## 12.5 CUDA Stream 的顺序语义

同一 stream 中操作按提交顺序执行：

```text
copy A -> kernel A -> copy B
```

不同 streams 的操作可以并发，但前提是没有数据依赖、硬件资源允许且同步关系正确。[CUDA Programming Guide 的 asynchronous concurrent execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)规定了 stream 顺序、重叠能力和显式同步的基本语义。

### 12.5.1 Transfer Stream

将下一 batch H2D 放到 transfer stream，同时 compute stream 处理当前 batch：

```mermaid
sequenceDiagram
    participant CPU
    participant T as Transfer Stream
    participant C as Compute Stream
    CPU->>T: copy batch N
    T-->>C: event N ready
    C->>C: compute batch N
    CPU->>T: copy batch N+1
    Note over T,C: copy N+1 与 compute N 尝试重叠
    T-->>C: event N+1 ready
    C->>C: compute batch N+1
```

### 12.5.2 CUDA Event 建立跨 Stream 依赖

Compute stream 不能猜 copy 何时完成。Transfer stream 记录 event，compute stream 等待该 event，再使用 tensor。

这不是全设备同步，只建立必要的局部依赖。

## 12.6 双缓冲的数据所有权

只有一个输入 buffer 时，CPU/GPU 可能要等当前计算结束才能安全覆盖。双缓冲：

```text
buffer 0: compute batch N
buffer 1: prepare/copy batch N+1
```

下一轮交换角色。

但 buffer 复用必须等上一次使用完成。过早覆盖会产生数据竞争；过度同步又失去 overlap。事件和明确 ownership 是关键。

## 12.7 Copy/Compute Overlap 的时间线证据

不能用“代码创建了 stream”作为证据。Nsight Systems 时间线应看到：

```text
H2D batch N+1 的时间区间
与 compute batch N 的 kernel 区间重叠
```

还应比较：

- GPU idle gap；
- H2D 总时间；
- decode step 间隔；
- 端到端 TPOT/tokens/s。

Timeline 有重叠但端到端没提升也可能合理：copy 原本占比很小，或瓶颈转移到其他部分。

## 12.8 隐式同步的来源

常见同步源：

```text
torch.cuda.synchronize()
读取 GPU tensor 标量到 CPU，例如 .item()
某些打印/异常检查
跨 stream 缺少正确事件导致保守等待
同步 D2H copy
allocator 或库内部边界
```

不能因为 `.item()` 代码短就忽略它。若在每个 decode step 读取 GPU 标量，CPU 可能每轮等待 GPU。

Sampling token ID 最终可能需要回 CPU 发给客户端，但可以考虑异步路径、批量处理或避免无关中间标量同步。

## 12.9 Admission Control 的资源保护目标

Admission control 在请求进入昂贵执行路径前判断是否接纳。

没有保护时：

```text
到达率 > 服务率
-> queue 持续增长
-> TTFT 增长
-> 客户端超时但后端继续算
-> 无效工作增加
-> goodput 下降
```

这叫 overload collapse 的一种表现：系统很忙，却很少完成仍有价值的请求。

```mermaid
flowchart LR
    R["Incoming request"] --> V{"输入是否合法?"}
    V -- "否" --> E1["400/422"]
    V -- "是" --> Q{"队列/Token/KV/SLO 容量足够?"}
    Q -- "否" --> E2["明确拒绝或限流"]
    Q -- "是" --> A["Admit -> waiting queue"]
```

拒绝一部分请求，可能让更多已接纳请求在 SLO 内完成。

## 12.10 Queue Length 与请求成本的差异

100 个请求可能都是 20-token prompt，也可能都是 20k prompt。成本差异巨大。

更合理的 admission 信号包括：

```text
tokenized prompt length
requested max output tokens
当前 queue token work
KV free blocks
running contexts
用户/租户配额
估计完成时间与 deadline
```

### 12.10.1 基于 Token 的 Cost Estimate

简单估算：

```text
prefill_cost ~ prompt_tokens
future_kv_reservation ~ prompt_tokens + max_output_tokens
decode_work ~ max_output_tokens
```

它不精确，但比字符数和请求数更接近真实资源。

## 12.11 字符限制与 Token 限制

入口可以先用 `max_prompt_chars` 快速拒绝极端请求，避免对巨大文本做昂贵 tokenization。但字符与 token 非一一对应。

更完整流程：

```text
cheap char/body limit
-> tokenize
-> exact token limit
-> capacity estimate
```

中文、代码和罕见字符的 token/char 比不同，不能只依赖字符限制做容量规划。

## 12.12 Queue Limit 与 Backpressure

当 queue 满时，服务应快速返回明确错误，而不是无限等待。上游可以重试、退避或转移流量。

Backpressure 意味着下游容量信息向上游传播：

```text
服务拒绝/限流
-> gateway 降低发送或排队
-> client exponential backoff
```

若所有层都无限排队，只会把延迟藏到不同位置。

### 12.12.1 过载响应语义

- 请求格式/长度非法可返回 4xx；
- 临时容量不足常用 429 或 503，具体取决于 API 契约；
- 可提供 retry hint，但不能保证重试一定成功。

## 12.13 Admission Control 与 Scheduler 的职责边界

Admission 决定“是否进入系统”；scheduler 决定“已进入的请求下一轮谁执行”。

Admission 使用容量估计，scheduler 使用实时 token/KV budget。两者需要共享统计，但不能互相替代。

如果 admission 过松，scheduler queue 爆炸；过严则 GPU 空闲、吞吐损失。阈值应由 workload 与 SLO 校准。

## 12.14 单变量优化与 A/B 证据

如果同时启用：

```text
pinned memory
transfer stream
chunked prefill
cache-aware scheduling
```

性能变化后无法归因。建议阶梯实验：

```text
baseline
baseline + pinned
baseline + pinned + transfer stream
再单独评估 scheduler 开关
最后组合
```

每步保持模型、prompt 分布、并发、输出长度和端口环境一致。

## 12.15 短负载中的固定开销

若 kernel 计算 100 ms、copy 只需 0.2 ms，即使完全隐藏 copy，理论收益也很小。

若 decode kernel 每步 0.3 ms、CPU 准备和 copy 0.2 ms，重叠可能更明显。

优化是否有用取决于比例。没有显著提升不是实验失败，只要能用 timeline 解释原因。

## 12.16 正确性与并发安全

### 12.16.1 Pinned Buffer 生命周期

Copy 完成前不能释放或覆盖 host buffer。

### 12.16.2 Stream 依赖

Compute 必须等待对应 batch copy event，不能错误等待别的 batch，也不能漏等。

### 12.16.3 Admission 原子性

多个请求同时检查 queue size 时，检查与入队需要一致；否则都看到“还有一个空位”并同时进入。

### 12.16.4 取消路径

已在 transfer 的请求取消后，buffer/event 仍需安全回收；已 admission 但未执行应从 queue 移除。

## 12.17 系统优化结论的适用边界

### 12.17.1 `non_blocking=True` 不保证执行重叠

还需要 pinned source、合适 stream、依赖和硬件支持。

### 12.17.2 Pinned Memory 不是无限资源

它占用不可换出的 host 资源，应限制和复用。

### 12.17.3 多 Stream 不自动产生并行

数据依赖、资源竞争和同步可能让时间线仍串行。

### 12.17.4 有界拒绝是过载保护的一部分

受控拒绝能保护已接纳请求和 goodput，比所有请求超时更可靠。

### 12.17.5 Queue Size 不能表达 Token 成本

LLM 请求成本与 token 长度和输出预算强相关。

### 12.17.6 小收益需要结合固定开销与测量噪声解释

也可能该 workload 中目标开销占比很小，需要 timeline 和更敏感 workload 判断。

## 12.18 本章小结与思考题

1. Pageable H2D 为什么可能经过 pinned staging？
2. 真正异步 copy 需要哪些条件？
3. Stream 与 event 分别解决什么？
4. 双缓冲怎样让 copy N+1 与 compute N 重叠？
5. 哪些操作可能造成隐式同步？
6. Admission control 为什么能提高过载 goodput？
7. 为什么 queue length 不足以估计 LLM 负载？
8. 如何做逐项 ablation 并证明 overlap？

## 12.19 参考资料

- NVIDIA：[CUDA C++ Programming Guide - Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
- NVIDIA：[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- NVIDIA：[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- PyTorch：[A Guide on Good Usage of `non_blocking` and `pin_memory()`](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
- 课程工程：[`qwen3_continuous_backend.py`](../../../course_vllm/model/qwen3_continuous_backend.py)、[`batching.py`](../../../course_vllm/server/batching.py)、[`api.py`](../../../course_vllm/server/api.py) 与 [`system_optimization.py`](../../../course_vllm/benchmarks/system_optimization.py)

系统优化结论只对给定 workload、硬件和测量边界成立，必须通过单变量 ablation 与时间线证据验证。
