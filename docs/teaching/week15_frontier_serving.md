# Week 15: 前沿 Serving 策略

## 0. 本节学习目标

Week15 不再讲基础算子，而是讲工业级 serving 策略。前面已经学习了 KV cache、paged KV、scheduler 和 batching，本周要理解更高层的策略如何利用这些机制：prefill/decode 解耦、Attention-FFN 解耦、cache-aware serving、TokenDance-style scheduling，以及投机解码的基本方向。

## 1. Prefix cache

很多请求共享前缀。例如同一个 system prompt、同一份工具说明、同一段检索文档。prefix cache 的目标是复用这些相同前缀的 KV cache。

如果两个请求的前 512 个 token 完全相同，那么第二个请求不必重新 prefill 这 512 个 token，可以复用已有 KV block。

这要求系统能：

- 对 prompt token block 做 hash。
- 判断完整 block 是否可复用。
- 管理引用计数。
- 在请求结束时正确释放。

prefix cache 的难点是“完整 block”复用。为了避免复杂的部分 token 对齐，入门实现通常只复用完整 block：只有当一段 token 正好覆盖一个 block，并且这个 block 的 token 序列完全相同，才复用对应 KV。这样实现简单，也和 paged KV 的 block table 自然结合。

prefix cache 的收益取决于 workload。如果所有请求 prompt 都完全不同，prefix cache 几乎没有收益。如果大量请求共享 system prompt 或检索模板，收益会很明显。

## 2. Cache-aware scheduling

prefix cache 只有存在复用机会还不够，调度也会影响命中率。cache-aware scheduling 会把共享前缀的请求排得更近，减少 cache 被淘汰的概率，提高复用。

它是一种策略层优化，不改变模型数学，也不改变 attention 公式。它改变的是请求顺序。

这类策略体现了 serving 系统的一个重要思想：同一个模型、同一个 kernel，不同调度顺序也会影响系统表现。调度器不只是 FIFO 队列，它可以利用请求结构信息，例如 prompt token、prefix hash、预计 decode 长度、请求年龄。

代价是公平性问题。过度偏向共享前缀请求，可能让没有共享前缀的请求等待更久。因此真实系统需要在 cache 命中率、等待时间和公平性之间平衡。

## 3. Prefill/decode 解耦

prefill 和 decode 的资源特征不同。prefill 计算大、并行度高；decode 每步小、延迟敏感、依赖 KV cache。

prefill-decode disaggregation 的思想是把 prefill-heavy 和 decode-heavy 工作拆开，甚至放到不同 worker 或不同 GPU 组。这样可以为两类 workload 使用不同 batch 策略。

代价是系统复杂度上升，KV cache 传递和调度协调更难。

这类方案背后的观察是：prefill 和 decode 混在一起时，长 prompt prefill 可能阻塞 decode，导致正在生成的请求 TPOT 变差。把 prefill 和 decode 拆开后，可以让 prefill worker 专注处理大块 prompt，把 decode worker 维持在稳定的 token iteration 节奏。

但拆分后必须解决 KV cache 从 prefill worker 到 decode worker 的交接问题。如果 KV cache 传输代价太大，收益可能被抵消。

## 4. Attention-FFN 解耦

Transformer layer 通常包含 attention 和 FFN 两个主要计算部分。decode 阶段中，attention 需要读取历史 KV cache，访存压力很强；FFN 更像逐 token 的大矩阵计算，计算密度更高。

Attention-FFN 解耦的思想是把这两类资源特征不同的工作拆开调度，甚至放到不同 worker、不同 GPU 或不同执行队列上。这样做的动机和 prefill/decode 解耦类似：如果两类工作瓶颈不同，就不一定要让它们被同一个调度节奏绑定。

它带来的问题也很直接：中间 activation 要在两类 worker 之间传递，调度依赖更复杂，故障恢复和性能分析也更难。学习这类系统时，不要只记住名字，要看它试图解决哪个瓶颈：是 attention 的 KV 访存，是 FFN 的计算吞吐，还是两者混合导致的资源利用不均。

## 5. Token-level scheduling

TokenDance 类策略关注 decode 过程中每个 sequence 的进度和剩余成本。它不是简单 FIFO，而是根据生成长度、年龄、剩余 token 估计等因素决定谁进入 decode batch。

这类策略的目标可能是降低 tail latency，也可能是改善公平性。

token-level scheduling 比 request-level scheduling 更细。request-level 只问“这个请求什么时候进入系统”，token-level 会问“这个 sequence 的下一个 token 什么时候生成”。LLM decode 天然是一轮一轮 token iteration，因此 token-level 策略有发挥空间。

学习时不需要把某篇论文完整复现，但需要学会把论文机制映射到工程对象：sequence、scheduler、KV cache handle、batch、queue、policy score。

## 6. 投机解码

投机解码在 Week12 已经作为系统优化方向简要出现，Week15 从 serving 策略角度再看一次。它的目标是减少大模型 decode 的串行步数：小模型先草拟多个 token，大模型批量验证。

它影响的不是某一个 CUDA kernel，而是整个 decode 流程：

- 需要 draft model 生成候选 token。
- 需要 target model 验证候选 token。
- 需要根据接受/拒绝结果更新 sequence。
- 需要维护两套或多套 KV cache 状态。
- 需要统计接受率、吞吐、TTFT、TPOT 等指标。

如果 draft model 太慢，或者候选经常被拒绝，投机解码可能没有收益。因此它是典型的系统策略：理论上减少大模型步数，实际收益取决于 workload、模型组合和实现开销。

## 7. 如何把论文机制映射到系统

读 serving 论文时，不要停留在“提出了一个调度策略”这种摘要层面。要继续问：

- 这个机制改变了哪个对象：request、sequence、KV block、batch、queue，还是 worker？
- 它新增了什么状态：prefix hash、引用计数、优先级分数、剩余长度估计，还是接受率统计？
- 它改变了哪个决策：谁先 prefill，谁进入 decode batch，哪个 cache 保留，哪个请求被抢占？
- 它影响哪些指标：TTFT、TPOT、吞吐、显存占用、cache 命中率、p99 latency？
- 它的代价是什么：通信、额外模型、复杂调度、状态同步，还是公平性问题？

这样的分析方法比复述论文摘要更重要。前沿策略本质上是在已有 serving 结构上改变状态、队列和调度决策。
