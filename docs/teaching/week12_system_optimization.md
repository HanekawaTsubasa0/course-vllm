# Week 12: 系统优化与 Admission Control

## 1. 本周核心问题

前面几周主要学习模型计算、KV cache 和调度。本周换一个角度：即使模型算子本身已经正确，在线服务仍然可能慢，甚至可能被过多请求拖垮。原因是请求进入 GPU 之前还有 CPU 侧准备、数据传输、同步等待、队列堆积和容量保护。

本周要回答四个问题：

- CPU 到 GPU 的数据传输为什么会影响 serving 延迟？
- pinned memory 和 CUDA stream 为什么能帮助 copy 与 compute 重叠？
- 为什么“写了异步 copy”不等于真的异步？
- admission control 为什么是服务系统必须具备的保护机制？

## 2. 背景知识：Host-to-device copy

模型权重通常已经在 GPU 上，但请求 token、attention mask、position ids 等输入数据可能从 CPU 构造后拷到 GPU。这个过程叫 host-to-device copy, H2D copy。

如果每次 copy 都同步阻塞，CPU 和 GPU 之间会出现等待。理想情况下，数据传输可以和部分计算重叠。

在 LLM serving 中，H2D copy 往往不是最大 FLOPs 开销，但它容易制造 timeline 上的空洞。尤其是在小 batch、短 prompt、decode step 很小的场景里，GPU kernel 本身很短，CPU 准备输入和数据传输的相对占比会变大。此时单纯优化模型 kernel 可能看不到明显收益，因为系统瓶颈已经移动到数据准备和调度。

学习时要注意：系统优化不是“所有开关都打开就一定更快”。它依赖 workload。长 prompt、大 batch 时，模型计算可能淹没 copy 成本；短请求、高并发时，copy、调度和同步可能显得更重要。

H2D copy 的完整路径可以粗略理解为：

```text
CPU 构造输入
-> 输入 tensor 位于 host memory
-> 通过 PCIe / NVLink 等通道传到 GPU memory
-> GPU kernel 读取 device tensor 开始计算
```

如果 copy 和 compute 串行，时间线像这样：

```text
copy batch 1 -> compute batch 1 -> copy batch 2 -> compute batch 2
```

理想的重叠是：

```text
compute batch 1
    overlaps with copy batch 2
compute batch 2
    overlaps with copy batch 3
```

重叠不是改变数学结果，而是减少硬件等待。CPU、copy engine 和 GPU compute unit 如果能同时忙起来，总耗时就可能下降。

## 3. 原理详解：Pinned memory

普通 CPU 内存可能被操作系统分页移动。GPU DMA 更喜欢页锁定内存，也就是 pinned memory。pinned memory 不能被操作系统随意换出，因此可以支持更高效的异步传输。

使用 pinned memory 的代价是它会占用宝贵的页锁定资源，过度使用会影响系统。它不是所有 workload 都一定更快，但在高频 H2D copy 场景中很重要。

PyTorch 中常见用法是先把 CPU tensor 放到 pinned memory，再用 non-blocking copy 传到 GPU：

```text
cpu_tensor.pin_memory()
gpu_tensor = cpu_tensor.to(device="cuda", non_blocking=True)
```

但 `non_blocking=True` 不等于一定异步。它需要 pinned memory、合适的 stream、没有立即消费导致的同步等条件共同成立。否则代码看起来是异步，timeline 上仍然可能串行。

为什么普通内存不够理想？因为操作系统管理普通 pageable memory 时，可能把页面换出、移动或延迟映射。GPU copy engine 要做 DMA 传输时，需要稳定的物理内存页。如果源内存不是 pinned memory，运行时可能先把数据拷到临时 pinned buffer，再从 pinned buffer 传给 GPU。这样多了一步，且更难异步。

pinned memory 的优点：

- 更适合 DMA。
- 更容易支持真正的 non-blocking H2D copy。
- 对频繁小批量输入传输有帮助。

pinned memory 的代价：

- 占用不能随便换出的物理内存。
- 分配和释放本身可能更贵。
- 使用过多会影响整机内存管理。

因此 pinned memory 适合复用，而不是每个请求临时大量分配。系统里常见做法是维护缓冲区或让数据加载管线复用 pinned buffer。

## 4. 原理详解：CUDA stream 和 transfer stream

CUDA stream 是 GPU 操作的有序队列。同一个 stream 内操作按顺序执行，不同 stream 之间可能并发或重叠。

如果把 H2D copy 放到专门的 transfer stream，同时 compute 在另一个 stream 上运行，理论上可以实现传输和计算重叠。但是否真正重叠取决于硬件、依赖关系、tensor 生命周期和同步点。

这就是为什么系统优化必须配合 Nsight Systems 看 timeline。

一个常见错误是：创建了 transfer stream，但马上在 default stream 上使用刚拷贝的 tensor，并隐式等待 copy 完成。这样程序语义正确，但没有得到预期重叠。真正的重叠需要合理安排依赖，让 GPU 在 copy 另一个 batch 输入时，仍能计算当前 batch。

判断异步是否真的生效，不能只看代码里有没有 `non_blocking=True` 或 stream 对象。要看 timeline：

```text
copy batch N+1
overlaps with
compute batch N
```

如果 copy 和 compute 在时间线上仍然前后串行，就说明依赖、同步点或内存条件没有满足。

理解 stream 时，先记住两个原则。

第一，同一个 stream 内的操作按提交顺序执行：

```text
stream 0: copy A -> compute A -> copy B -> compute B
```

这里天然串行。即使 copy 本身支持异步，只要后面的 compute 必须等待它，stream 内也不会乱序。

第二，不同 stream 之间可以并发，但必须显式处理依赖：

```text
transfer stream: copy batch N+1
compute stream:  compute batch N
```

如果 compute batch N+1 要读取 copy batch N+1 的结果，就必须等待对应 copy 完成。常见做法是用 event 表达依赖：

```text
transfer stream copy 完成
-> record event
compute stream wait event
-> compute 使用这份输入
```

初学者常见误区是把 stream 当成“自动并行开关”。实际上 stream 只是给运行时提供并发执行的可能性，真正能不能重叠还取决于：

- 硬件是否有独立 copy engine。
- copy 和 compute 是否使用不同资源。
- 源内存是否 pinned。
- 后续操作是否立刻同步等待。
- 是否调用了会强制同步的 API。
- tensor 生命周期是否安全。

所以系统优化必须看 timeline，而不是只看代码表面。

## 5. 原理详解：Admission control

admission control 是请求进入系统前的保护机制。LLM 服务里常见限制包括：

- 最大 prompt 长度。
- 最大队列长度。
- 最大 batch size。
- 最大 batched tokens。
- 最大并发请求。

没有 admission control 时，过长 prompt 或过多请求可能导致队列无限增长、TTFT 爆炸、显存耗尽。拒绝请求有时比让所有请求超时更好。

admission control 的本质是把系统容量边界显式化。在线服务不应该假设所有请求都能被无限接收。尤其是 LLM 请求成本差异很大：一个 20 token prompt 和一个 20k token prompt 对 prefill、KV cache 和排队的影响完全不同。

常见策略包括：

- prompt token 数限制。
- 请求队列长度限制。
- 单用户并发限制。
- 预计 KV cache 占用限制。
- 按优先级接收或拒绝。

入门实现通常会先做 prompt 长度和队列深度限制。生产系统会进一步用 tokenizer 后的 token 数、预计输出长度和当前 KV cache 剩余容量做判断。

admission control 不是“服务偷懒拒绝用户”，而是避免系统进入不可恢复的坏状态。没有 admission control 时，常见后果包括：

- 队列无限增长，所有请求 TTFT 都变差。
- 大量超长 prompt 占满 KV cache，新请求无法进入。
- GPU OOM 导致正在服务的请求一起失败。
- 请求已经排队很久，最后仍然超时，浪费用户等待时间。

一个简单的接收判断可以这样理解：

```text
如果 prompt_tokens > max_prompt_tokens:
    拒绝，因为单个请求太长

如果 waiting_queue_size >= max_queue_size:
    拒绝，因为排队已经过长

如果 estimated_kv_tokens > remaining_kv_capacity:
    拒绝或等待，因为显存容量不足
```

更高级的策略会加入优先级、公平性、租户隔离、SLA、请求超时和取消机制。无论策略多复杂，本质都是一个问题：系统当前容量是否足够接纳这个请求，并在合理时间内完成它？

## 6. 背景知识：投机解码与量化的方向性理解

投机解码 speculative decoding 是一种降低 decode 成本的思路。它通常使用一个较小、较快的 draft model 先提出若干候选 token，再由目标大模型一次性验证这些 token。若候选被接受，就能减少大模型逐 token 调用的次数；若候选被拒绝，则需要回退到大模型结果。

投机解码的关键不是“让小模型替代大模型”，而是“用小模型猜测，大模型保证分布正确或近似正确”。它适合大模型 decode 成本高、draft model 足够快且接受率较高的场景。工程上要处理 draft/verify 两套模型、KV cache 同步、接受率统计和回退逻辑。

量化 quantization 是减少权重、activation 或 KV cache 数值精度的方法，例如从 FP16/BF16 降到 INT8、INT4 或 FP8。它的收益通常包括更低显存占用、更高带宽效率和可能更高吞吐；风险是精度下降、kernel 更复杂、不同硬件支持差异明显。

这一节把投机解码和量化作为方向性知识理解。它们和 pinned memory、stream、admission control 一样，都是从系统层面降低成本或保护服务的手段。

## 7. 本节小结

系统优化必须用对照测量判断收益。只打开 pinned memory 或 transfer stream，并不自动说明性能变好。要比较优化前后：

- TTFT 是否降低。
- TPOT 是否降低。
- output tokens/s 是否提高。
- GPU timeline 是否减少空洞。
- H2D copy 是否和 compute 有重叠。
- admission control 是否减少超长排队或 OOM。

还要注意 workload。长 prompt、大 batch 时，模型计算可能是主导；短请求、高并发时，CPU 调度、H2D copy 和同步开销更容易显现。

系统优化的正确姿势是一次只改变一个主要因素。否则看到性能变化时，很难判断是 pinned memory、生效的 stream、batch size 变化，还是请求分布变化导致的。学完本节后，应能解释为什么异步拷贝不一定真的异步，为什么 admission control 是保护系统而不是“故意拒绝用户”。
