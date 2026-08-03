# Week 15: 前沿 Serving 策略

## 1. 本周核心问题

当前沿 serving 论文或系统说自己“更快”时，它通常不是只改了一个 CUDA kernel，而是在请求调度、cache 复用、模型拆分、资源隔离和解码策略上做组合优化。本周的重点是学会看懂这些策略解决的瓶颈，而不是只记住名字。

本周要回答四个问题：

- 为什么 prefix cache 和 cache-aware scheduling 能减少重复 prefill？
- 为什么 prefill/decode 解耦适合长 prompt 和高并发场景？
- 为什么 attention 和 FFN 可能需要不同的资源调度方式？
- 投机解码为什么能减少大模型 decode 的串行步数，又为什么不一定总是加速？

## 2. 背景知识：Prefix cache

很多请求共享前缀。例如同一个 system prompt、同一份工具说明、同一段检索文档。prefix cache 的目标是复用这些相同前缀的 KV cache。

如果两个请求的前 512 个 token 完全相同，那么第二个请求不必重新 prefill 这 512 个 token，可以复用已有 KV block。

这要求系统能：

- 对 prompt token block 做 hash。
- 判断完整 block 是否可复用。
- 管理引用计数。
- 在请求结束时正确释放。

prefix cache 的难点是“完整 block”复用。为了避免复杂的部分 token 对齐，入门实现通常只复用完整 block：只有当一段 token 正好覆盖一个 block，并且这个 block 的 token 序列完全相同，才复用对应 KV。这样实现简单，也和 paged KV 的 block table 自然结合。

prefix cache 的收益取决于 workload。如果所有请求 prompt 都完全不同，prefix cache 几乎没有收益。如果大量请求共享 system prompt 或检索模板，收益会很明显。

要理解 prefix cache，必须先理解 prefill 的成本结构。prefill 需要让 prompt 中每个 token 经过所有 transformer layer，并生成对应 KV cache。prompt 越长，prefill 成本越高。如果许多请求的开头完全相同，重复计算这部分前缀就是浪费。

例如很多应用会把同一段 system prompt 放在每个用户请求前面：

```text
system prompt: 你是一个严谨的助手...
user prompt A: 请总结文档 A
user prompt B: 请总结文档 B
```

如果 system prompt 对应的 token 完全一致，那么它产生的 KV 可以复用。第二个请求不需要重新计算 system prompt 的 KV，只需要从分叉点之后继续 prefill。

但是 prefix cache 不是普通字符串缓存。它缓存的是模型内部状态，所以必须满足严格条件：

- tokenizer 后的 token id 必须相同。
- token 顺序必须相同。
- model weights、position encoding、dtype 等上下文必须一致。
- cache 里的 KV 不能被提前释放。

只要其中一个条件不满足，复用就可能导致错误结果。

## 3. 原理详解：Cache-aware scheduling

prefix cache 只有存在复用机会还不够，调度也会影响命中率。cache-aware scheduling 会把共享前缀的请求排得更近，减少 cache 被淘汰的概率，提高复用。

它是一种策略层优化，不改变模型数学，也不改变 attention 公式。它改变的是请求顺序。

这类策略体现了 serving 系统的一个重要思想：同一个模型、同一个 kernel，不同调度顺序也会影响系统表现。调度器不只是 FIFO 队列，它可以利用请求结构信息，例如 prompt token、prefix hash、预计 decode 长度、请求年龄。

代价是公平性问题。过度偏向共享前缀请求，可能让没有共享前缀的请求等待更久。因此真实系统需要在 cache 命中率、等待时间和公平性之间平衡。

可以把 cache-aware scheduling 看成“调度器知道 cache 的存在”。普通 FIFO 只看请求到达顺序：

```text
A -> B -> C -> D
```

cache-aware scheduling 会额外看请求之间是否有共享前缀：

```text
A 和 C 共享 prefix
B 和 D 共享 prefix
```

如果把 A 和 C 排得更近，C 复用 A 的 prefix cache 的概率就更高。如果中间插入太多其他请求，cache 可能被淘汰，复用机会消失。

这类策略体现一个重要思想：serving 系统里，请求顺序本身就是性能变量。两个系统使用同一个模型、同一张 GPU、同一个 kernel，只要调度顺序不同，就可能出现不同的 TTFT、吞吐和 cache 命中率。

## 4. 原理详解：Prefill/decode 解耦

prefill 和 decode 的资源特征不同。prefill 计算大、并行度高；decode 每步小、延迟敏感、依赖 KV cache。

prefill-decode disaggregation 的思想是把 prefill-heavy 和 decode-heavy 工作拆开，甚至放到不同 worker 或不同 GPU 组。这样可以为两类 workload 使用不同 batch 策略。

代价是系统复杂度上升，KV cache 传递和调度协调更难。

这类方案背后的观察是：prefill 和 decode 混在一起时，长 prompt prefill 可能阻塞 decode，导致正在生成的请求 TPOT 变差。把 prefill 和 decode 拆开后，可以让 prefill worker 专注处理大块 prompt，把 decode worker 维持在稳定的 token iteration 节奏。

但拆分后必须解决 KV cache 从 prefill worker 到 decode worker 的交接问题。如果 KV cache 传输代价太大，收益可能被抵消。

更细地看，prefill 和 decode 对硬件的压力不同：

- prefill 一次处理多个 prompt token，矩阵乘规模较大，容易形成较高并行度。
- decode 每轮每个 sequence 只处理一个 token，更强调低延迟和稳定节奏。
- prefill 产生大量新的 KV，decode 反复读取历史 KV。
- 长 prompt prefill 可能占用很长时间，影响正在 decode 的请求。

prefill/decode 解耦的目标是让两类工作不要互相拖累。可以想象有两类 worker：

```text
prefill worker: 专门处理长 prompt，把 KV cache 计算出来
decode worker: 专门按稳定节奏生成下一个 token
```

难点在交接。prefill worker 算完后，decode worker 需要拿到正确的 KV cache。如果两者在不同 GPU 上，就涉及 GPU 间传输；如果传输开销太高，解耦的好处会被抵消。这也是为什么 serving 策略不能只看算法流程，还要看硬件拓扑和通信成本。

## 5. 原理详解：Attention-FFN 解耦

Transformer layer 通常包含 attention 和 FFN 两个主要计算部分。decode 阶段中，attention 需要读取历史 KV cache，访存压力很强；FFN 更像逐 token 的大矩阵计算，计算密度更高。

Attention-FFN 解耦的思想是把这两类资源特征不同的工作拆开调度，甚至放到不同 worker、不同 GPU 或不同执行队列上。这样做的动机和 prefill/decode 解耦类似：如果两类工作瓶颈不同，就不一定要让它们被同一个调度节奏绑定。

它带来的问题也很直接：中间 activation 要在两类 worker 之间传递，调度依赖更复杂，故障恢复和性能分析也更难。学习这类系统时，不要只记住名字，要看它试图解决哪个瓶颈：是 attention 的 KV 访存，是 FFN 的计算吞吐，还是两者混合导致的资源利用不均。

为什么 attention 和 FFN 的资源特征不同？可以从计算和访存看：

- decode attention 需要读取历史 K/V，历史越长，读的 KV 越多。
- FFN 主要是对当前 token 的 hidden state 做大矩阵变换，通常计算密度更高。
- attention 更容易受 memory bandwidth 和 KV cache layout 影响。
- FFN 更容易受矩阵乘吞吐、tensor core 利用率和 batch size 影响。

如果两类工作放在同一执行节奏里，某一类资源可能空闲，另一类资源可能拥堵。Attention-FFN 解耦就是尝试把不同瓶颈拆开调度。但它不是免费优化，因为 transformer layer 内部存在严格依赖：

```text
input hidden state
-> attention
-> residual / norm
-> FFN
-> residual
-> next layer
```

拆得越细，中间状态传输和同步越多。收益来自资源利用率提高，代价来自通信、调度和系统复杂度。

## 6. 原理详解：Token-level scheduling

一种适合教学演示的 token-level scheduling 启发式，是关注 decode 过程中每个 sequence 的进度和剩余工作量。它不使用简单 FIFO，而是根据已生成长度、剩余 token 估计和请求年龄等因素决定谁进入下一轮 decode batch。

如果优先完成剩余工作量较小的请求，这种策略接近 shortest-remaining-work 思路，有机会降低平均完成时间；同时加入请求年龄作为排序条件，可以缓解长请求长期得不到调度的问题。它依赖对剩余生成长度的估计，估计不准时收益可能消失，甚至损害公平性。

token-level scheduling 比 request-level scheduling 更细。request-level 只问“这个请求什么时候进入系统”，token-level 会问“这个 sequence 的下一个 token 什么时候生成”。LLM decode 天然是一轮一轮 token iteration，因此 token-level 策略有发挥空间。

学习时不需要把某篇论文完整复现，但需要学会把论文机制映射到工程对象：sequence、scheduler、KV cache handle、batch、queue、policy score。

request-level scheduling 和 token-level scheduling 的区别可以这样理解：

```text
request-level: 这个请求什么时候开始？
token-level: 这个 sequence 的下一个 token 什么时候算？
```

LLM decode 是天然逐 token 的，所以 token-level scheduling 能更细地控制延迟。例如，一个请求已经等待很久但还没生成完，另一个请求刚进入系统。调度器可能根据请求年龄、已生成长度、剩余长度估计、优先级等信息决定谁进入下一轮 decode batch。

这类策略的难点是目标不唯一。系统可能想优化平均延迟，也可能想优化 p99，也可能要保证不同用户公平。不同目标会导致不同调度策略。

## 7. 背景知识：投机解码

投机解码在 Week12 已经作为系统优化方向简要出现，Week15 从 serving 策略角度再看一次。它的目标是减少大模型 decode 的串行步数：小模型先草拟多个 token，大模型批量验证。

它影响的不是某一个 CUDA kernel，而是整个 decode 流程：

- 需要 draft model 生成候选 token。
- 需要 target model 验证候选 token。
- 需要根据接受/拒绝结果更新 sequence。
- 需要维护两套或多套 KV cache 状态。
- 需要统计接受率、吞吐、TTFT、TPOT 等指标。

如果 draft model 太慢，或者候选经常被拒绝，投机解码可能没有收益。因此它是典型的系统策略：理论上减少大模型步数，实际收益取决于 workload、模型组合和实现开销。

自回归 decode 慢的根本原因是串行依赖：

```text
生成 token t
-> 把 token t 放回输入
-> 生成 token t+1
-> 再生成 token t+2
```

大模型每次只前进一步，无法直接跳过多个 token。投机解码的思路是让小模型先猜一段：

```text
draft model 猜: x1, x2, x3, x4
target model 验证这些候选
接受前 k 个
如果某个位置不接受，从 target model 的结果继续
```

它的收益取决于接受率。接受率高，说明小模型猜得准，大模型一次验证可以推进多个 token；接受率低，说明经常回退，小模型的额外工作可能变成负担。

投机解码还会让系统状态更复杂：

- draft model 有自己的 KV cache。
- target model 有自己的 KV cache。
- 被接受的 token 要同步到最终 sequence。
- 被拒绝的位置要正确回退。
- 采样分布要保持正确或满足近似要求。

所以它不是简单“加一个小模型就会快”，而是用额外计算和额外状态管理换取更少的大模型串行步数。

## 8. 本节小结：如何把论文机制映射到系统

读 serving 论文时，不要停留在“提出了一个调度策略”这种摘要层面。要继续问：

- 这个机制改变了哪个对象：request、sequence、KV block、batch、queue，还是 worker？
- 它新增了什么状态：prefix hash、引用计数、优先级分数、剩余长度估计，还是接受率统计？
- 它改变了哪个决策：谁先 prefill，谁进入 decode batch，哪个 cache 保留，哪个请求被抢占？
- 它影响哪些指标：TTFT、TPOT、吞吐、显存占用、cache 命中率、p99 latency？
- 它的代价是什么：通信、额外模型、复杂调度、状态同步，还是公平性问题？

这样的分析方法比复述论文摘要更重要。前沿策略本质上是在已有 serving 结构上改变状态、队列和调度决策。学完本节后，应能看到一个新的 serving 方法时，判断它改变了什么状态、什么队列、什么资源分配策略，以及它用哪些指标证明有效。
