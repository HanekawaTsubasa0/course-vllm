# Week 13: 多卡容量规划与并行策略

## 0. 本节学习目标

本节讨论两个问题：第一，如何估算一个 LLM 推理服务在单张 GPU 上能承载多长上下文、多少并发、多少吞吐；第二，当单卡放不下或跑不动时，多卡并行到底解决什么问题，又会引入什么通信代价。

容量规划不是背公式，而是把系统资源拆开：显存放了什么，计算花在哪里，通信什么时候发生，延迟指标受谁影响。只有先把这些量估清楚，才知道应该优化 KV cache、换更大显存、降低上下文长度、增加并发，还是引入张量并行和流水并行。

## 1. 推理服务里的显存构成

LLM 推理时，显存主要被几类对象占用：

```text
总显存
≈ 模型权重
+ KV cache
+ 临时 activation
+ CUDA workspace / 通信 buffer
+ allocator 碎片
+ 安全余量
```

模型权重是最直观的一部分。一个参数如果用 FP16/BF16 存储，通常占 2 bytes；如果有 7B 参数，权重本身大约是：

```text
7e9 * 2 bytes = 14 GB
```

这只是权重，不包括 KV cache 和运行时开销。推理不像训练那样保存大量反向传播 activation，也没有 optimizer state，但它有一个训练时不那么突出的显存大户：KV cache。

KV cache 随着请求上下文长度和并发数增长。权重大小主要由模型决定，而 KV cache 大小由“模型结构 + 当前服务负载”共同决定。一个服务刚启动时显存看起来够用，不代表高并发长上下文时仍然够用。

## 2. 为什么 KV cache 是容量规划核心

自回归生成时，每一层 attention 都需要历史 token 的 key 和 value。为了避免每一步重新计算历史 K/V，服务会把它们存下来。每新增一个 token，就要为每一层写入一份 K 和一份 V。

单个 token 的 KV cache 大小近似为：

```text
kv_bytes_per_token =
num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
```

这里每一项都有含义：

- `num_layers`: 每层都要保存历史 K/V。
- `2`: K 和 V 两份。
- `num_kv_heads`: KV head 数量。GQA/MQA 会让 KV heads 少于 query heads，从而减少 KV cache。
- `head_dim`: 每个 head 的维度。
- `dtype_bytes`: FP16/BF16 通常是 2 bytes，FP8/INT8 KV cache 会更小，但有精度和 kernel 支持问题。

假设一个模型有 28 层、8 个 KV heads、head_dim 为 128，KV cache 使用 FP16：

```text
kv_bytes_per_token
= 28 * 2 * 8 * 128 * 2
= 114688 bytes
≈ 112 KB / token
```

如果一个请求上下文长度是 4096 token，则这个请求的 KV cache 大约是：

```text
4096 * 112 KB ≈ 448 MB
```

如果同时有 16 个这样的请求，只 KV cache 就接近：

```text
16 * 448 MB ≈ 7 GB
```

这说明长上下文和高并发为什么很快吃掉显存。并发数翻倍，KV cache 近似翻倍；上下文长度翻倍，KV cache 也近似翻倍。

## 3. 从显存预算到最大 token slots

容量估算的第一步是算出能给 KV cache 留多少显存：

```text
kv_budget =
gpu_memory * usable_ratio
- weight_memory
- runtime_overhead
- safety_margin
```

`usable_ratio` 不能随便取 1。真实系统需要给 CUDA allocator、workspace、通信 buffer、临时 tensor 和碎片留空间。常见估算会用 0.85 到 0.95 之间的比例，再根据实际 profiler 和 OOM 情况修正。

有了 `kv_budget`，就能估算最多能容纳多少历史 token：

```text
token_slots = floor(kv_budget / kv_bytes_per_token)
```

如果使用 paged KV cache，系统通常按 block 分配，而不是按单个 token 分配。设 block size 是 `B`，一个 block 的大小是：

```text
kv_block_bytes = B * kv_bytes_per_token
```

可分配 block 数：

```text
num_blocks = floor(kv_budget / kv_block_bytes)
```

总 token slots：

```text
token_slots = num_blocks * B
```

这里的 `token_slots` 不是“最多能服务多少请求”，而是所有活跃请求的上下文 token 总和上限。比如 `token_slots = 131072`，如果每个请求最多 4096 token，则满长并发上限大约是：

```text
131072 / 4096 = 32 requests
```

但如果大多数请求只有 512 token，理论并发就会高得多。容量规划必须说明 workload 假设：平均 prompt 多长，平均输出多长，最大上下文多长，是否存在少量超长请求。

## 4. 不能只看显存，还要看吞吐和延迟

显存能放下，只说明系统不一定 OOM；它不说明速度够不够。推理服务还要看计算吞吐和延迟。

prefill 阶段处理整个 prompt，矩阵乘和 attention 规模较大，GPU 并行度通常比较高。长 prompt 的 prefill 会显著影响 TTFT，因为用户要等完整 prompt 处理完后才能看到第一个 token。

decode 阶段每轮为每个活跃 sequence 生成一个 token。单个 sequence 的 decode 计算量不大，但要反复执行很多轮，并且每轮要读取历史 KV cache。decode 常常受内存带宽、kernel launch、调度开销、batch size 和通信延迟影响。

因此容量规划至少要分开问：

- 显存是否能容纳权重和 KV cache？
- prefill 的 TTFT 是否能接受？
- decode 的 TPOT 是否能接受？
- 服务整体 tokens/s 是否达到目标？
- 高并发时尾延迟是否爆炸？

一个系统可能显存足够，但 decode 太慢；也可能单请求很快，但高并发时队列等待导致尾延迟很差。

## 5. 什么情况下需要多卡

多卡不是自动加速按钮。引入多卡通常是因为遇到以下几类瓶颈：

第一，权重放不下。模型权重本身超过单卡可用显存时，必须把权重切到多张卡，常见方式是张量并行或流水并行。

第二，KV cache 放不下。长上下文和高并发让 KV cache 超出单卡容量时，可以增加显存，也可以降低上下文/并发，或者考虑 KV cache 量化、分页管理、跨卡切分等方案。

第三，吞吐不够。单卡能放下模型，但 tokens/s 不够，可能需要更多 GPU 分摊计算。

第四，延迟不达标。某些情况下多卡能降低单层计算时间，但通信也会增加延迟，所以低延迟服务并不总是因为加卡而变好。

做判断时要先区分是容量问题还是速度问题。容量问题看显存构成，速度问题看 profiler、tokens/s、TTFT、TPOT 和尾延迟。

## 6. Tensor Parallelism

Tensor parallelism, TP, 把同一层内部的矩阵乘切到多张 GPU 上。以线性层为例，可以按输出维切分 weight，每张卡计算一部分输出；也可以按输入维切分，再把部分结果相加。

TP 的优点是能分摊单层权重和计算。对于大模型，单层矩阵很大，切到多卡后每张卡计算量下降，权重显存也下降。

TP 的代价是层内通信。常见通信包括：

- all-reduce: 多张卡各自得到部分结果，需要求和后让每张卡拿到完整结果。
- all-gather: 每张卡持有一部分 tensor，需要拼成完整 tensor。
- reduce-scatter: 先归约再分发，每张卡只保留自己需要的部分。

这些 collective 通信通常由 NCCL 这类库完成。通信不是免费的，尤其 decode 阶段每 token 都可能走很多层，每层都可能触发通信。batch 很小或延迟要求很严时，TP 通信开销会非常明显。

可以把 TP 理解为：

```text
减少单卡计算和权重压力
换来层内通信开销
```

## 7. Pipeline Parallelism

Pipeline parallelism, PP, 把不同 transformer layers 放到不同 GPU。比如 GPU0 负责前 12 层，GPU1 负责后 12 层。这样每张卡只保存一部分层的权重。

PP 的优点是直观，能分摊权重显存。它特别适合模型层数多、单卡放不下所有层的情况。

PP 的问题是 pipeline bubble。一个请求必须先经过前面的 stage，后面的 stage 才能工作。如果 microbatch 数不足，部分 GPU 会等待。训练中可以用很多 microbatch 填满流水线，但在线推理请求常常不稳定，decode 又是逐 token 进行，流水线利用率不一定好。

对推理来说，PP 还会带来跨 stage activation 传输。每层边界传输的数据量不一定像权重那么大，但在低延迟场景中仍然需要考虑。

可以把 PP 理解为：

```text
按层分摊权重
换来流水线气泡和 stage 间传输
```

## 8. Expert Parallelism 与 Context Parallelism

Expert parallelism, EP, 常用于 MoE 模型。MoE 的 FFN 部分由多个 expert 组成，每个 token 只路由到其中少数 expert。EP 可以把不同 expert 放到不同 GPU。

EP 的主要通信是 all-to-all：token 要根据 router 结果发送到对应 expert，计算完后再收集回来。它适合 MoE 模型，但对 dense transformer 不是主要策略。

Context parallelism, CP, 面向超长上下文。它把 sequence/context 维度切到多张 GPU 上，减轻单卡 KV cache 和 attention 负担。代价是 attention 需要跨卡交换 K/V 或中间统计量。长上下文下 CP 有意义，短上下文下通信复杂度可能不划算。

这两类并行说明一个共同点：并行策略必须和模型结构、瓶颈类型匹配。不是所有模型都适合 EP，也不是所有上下文长度都值得做 CP。

## 9. 一次容量估算应该怎么做

做容量估算时，可以按下面顺序推导。

第一步，列出模型结构和数据类型：

```text
num_params
num_layers
num_kv_heads
head_dim
weight_dtype_bytes
kv_dtype_bytes
```

第二步，估算权重显存：

```text
weight_memory = num_params * weight_dtype_bytes
```

如果有量化权重，则用量化后的平均 bytes/parameter，但要额外考虑 scale、zero point、packing 和 kernel workspace。

第三步，估算每 token KV cache：

```text
kv_bytes_per_token =
num_layers * 2 * num_kv_heads * head_dim * kv_dtype_bytes
```

第四步，估算可用 KV 显存：

```text
kv_budget =
gpu_memory * usable_ratio
- weight_memory
- runtime_overhead
- safety_margin
```

第五步，算 token slots：

```text
token_slots = floor(kv_budget / kv_bytes_per_token)
```

如果使用 paged KV，则再按 block size 向下取整：

```text
num_blocks = floor(kv_budget / (block_size * kv_bytes_per_token))
token_slots = num_blocks * block_size
```

第六步，根据 workload 假设换算并发：

```text
max_concurrency ≈ token_slots / average_active_context_length
```

这里的 `average_active_context_length` 不能只看 prompt 长度，还要考虑生成过程中输出 token 也会占 KV cache。一个请求生成得越久，它占用的 KV cache 越大。

第七步，再判断速度：

```text
prefill tokens/s 是否足够？
decode tokens/s 是否足够？
TTFT 是否满足？
TPOT 是否满足？
p90/p99 latency 是否满足？
```

第八步，判断多卡策略：

```text
权重放不下 -> 优先考虑 TP/PP
KV cache 放不下 -> 优先考虑减少上下文/并发、paged KV、KV 量化，必要时考虑跨卡方案
计算吞吐不够 -> 看 prefill 还是 decode 瓶颈，再考虑 TP、batching 或 kernel 优化
通信开销过大 -> 减少并行粒度、优化 collective、重新评估是否值得多卡
```

容量规划的结论应该是解释性的，而不是只给一个数字。好的结论会说清楚：在什么 workload 假设下，瓶颈来自权重、KV cache、计算还是通信；如果需求变化，哪个参数最敏感。

## 10. 本节在实验中的少量对照

实验中会用一个容量估算脚本帮助检查上述公式。它只是把本节的估算步骤程序化：输入模型结构、GPU 显存、block size、上下文长度和并行配置，输出 KV budget、token slots、满长并发和通信量估算。

阅读脚本时不要把重点放在代码行数上，而要确认每个输出数字来自哪一个公式、依赖哪一个假设。
