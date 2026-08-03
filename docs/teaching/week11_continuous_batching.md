# Week 11: Continuous Batching

## 1. 本周核心问题

在线 LLM 服务不是一次只处理一个请求。多个用户的请求会在不同时间到达，每个请求 prompt 长度不同，生成长度也不同。系统如果处理得不好，就会出现 GPU 空转、短请求被长请求拖慢、吞吐上不去、用户等待变长。

本周要回答四个问题：

- 为什么传统 fixed batching 不适合自回归生成？
- continuous batching 为什么要在每个 decode iteration 重新组 batch？
- prefill 和 decode 的资源特征有什么不同？
- 调度器如何在吞吐、TTFT、TPOT 和公平性之间取舍？

## 2. 背景知识：为什么普通 batching 不够

普通 fixed batching 的思路是收集一批请求，一起开始，一起结束。它适合输入输出长度相近的任务，例如图像分类。

LLM 生成不适合 fixed batching：

- prompt 长度不同。
- 输出长度不同。
- 有的请求很快 EOS，有的请求生成很久。
- decode 是逐 token 循环。

如果固定 batch 必须等最慢请求结束，短请求完成后也占着 batch 位置，GPU 利用率下降。

举一个简单例子。假设有 3 个请求一起组成一个 fixed batch：

```text
请求 A: 需要生成 4 个 token
请求 B: 需要生成 30 个 token
请求 C: 需要生成 6 个 token
```

如果 batch 必须一起运行到最慢请求结束，那么 A 在第 4 步后已经结束，C 在第 6 步后已经结束，但它们原来占用的位置不能立刻被新请求充分利用。后面的 24 轮里，实际只有 B 还在生成。GPU 看到的 batch 变小，吞吐下降。

LLM serving 的关键难点就在这里：请求不是一个固定大小的矩阵乘任务，而是一组不断变化的 sequence。每个 sequence 可能在不同时间进入系统，也可能在不同时间结束。

## 3. 原理详解：Continuous batching

Continuous batching 的思想是：batch 不是一次性固定到请求结束，而是在每个 iteration 重新组织。

大致流程：

```text
new requests -> waiting queue
prefill completed -> running queue
each iteration:
    choose some waiting requests for prefill
    choose running requests for decode
    finished requests leave
    new requests can join later iterations
```

Orca 论文中常用 iteration-level scheduling 描述这类思想。它把生成过程拆成迭代，每轮可以重新选择参与计算的 sequence。

可以把 decode 想成很多轮“下一 token 计算”：

```text
iteration 1: A, B, C 生成各自下一个 token
iteration 2: A, B, C 生成各自下一个 token
iteration 3: A 结束，B, C 继续；新请求 D 可以加入
iteration 4: B, C, D 生成各自下一个 token
```

continuous batching 的重点不是“客户端并发连接很多”，而是“模型执行层每一轮能把仍然活跃的 sequence 重新组成 batch”。如果服务端只是开了多个线程，但模型仍然一个请求一个请求跑，那不叫真正的 continuous batching。

iteration-level scheduling 让调度器每轮都重新回答：

- 哪些 sequence 还活着？
- 哪些 sequence 已经 EOS 或达到最大长度？
- 哪些新请求的 prefill 可以插入？
- 当前 GPU token budget 还能容纳多少工作？
- 是否要把长 prompt 拆成 chunk，避免阻塞 decode？

## 4. 原理详解：Prefill 和 decode 的调度差异

prefill 请求的 token 数可能很大。一个长 prompt prefill 会占用大量计算，可能阻塞 decode 请求，导致正在生成的用户 token 变慢。

decode 请求每个 sequence 通常只贡献一个 token，但数量可能很多。decode batch 越大，GPU 利用率通常越好。

prefill 和 decode 的差异可以从 shape 上理解。假设 batch size 是 B，prompt 长度是 S，hidden size 是 H：

```text
prefill 输入大致是 [B, S, H]
decode 每轮输入大致是 [B, 1, H]
```

prefill 一次处理很多 prompt token，矩阵乘规模较大，并行度比较好；decode 每轮只处理每个 sequence 的一个新 token，单轮工作量小，但要重复很多轮。decode 的串行性更强，因为第 t+1 个 token 必须等第 t 个 token 采样出来之后才能计算。

这也是为什么用户体验会同时关心 TTFT 和 TPOT：

- TTFT 主要受排队、tokenization、prefill、第一次 decode 影响。
- TPOT 主要受 decode iteration 的稳定性影响。
- 一个长 prefill 插进来，可能让很多正在 decode 的请求下一 token 变慢。

调度器必须在两者之间平衡：

- 优先 prefill 可以降低新请求 TTFT。
- 优先 decode 可以降低正在生成请求的 TPOT。
- chunked prefill 可以把长 prompt 切开，避免一次 prefill 占用过久。

## 5. 原理详解：队列策略、抢占与 chunked prefill

continuous batching 通常会维护 waiting queue 和 running queue。waiting queue 保存还没有完成 prefill 的新请求，running queue 保存已经进入 decode 循环的请求。

队列策略决定“下一轮算谁”。最简单的是 FIFO，但 FIFO 不一定最优。一个很长的 prompt 如果排在前面，可能让后面很多短请求等待；一批 decode 请求如果长期得不到执行，用户会感到输出卡顿。

抢占 preemption 指调度器临时暂停某些 sequence，把预算让给更紧急或更合适的请求。它的代价是状态管理更复杂：被抢占的请求是否保留 KV cache，是否释放 block，恢复时如何继续生成，都必须定义清楚。

chunked prefill 是处理长 prompt 的常见方法。它不一次性 prefill 完所有 prompt token，而是把 prompt 分成多段：

```text
long prompt
-> chunk 1 prefill
-> allow decode / other prefill
-> chunk 2 prefill
-> ...
```

这样可以避免一个超长 prompt 长时间占住 GPU，让 decode 请求保持更稳定的输出节奏。代价是调度器要维护“这个请求的 prefill 做到哪里了”。

抢占和 chunked prefill 都说明调度器管理的是 sequence 状态，而不只是 HTTP 请求。一个请求可能处在这些状态之一：

```text
waiting: 已到达，但还没有足够资源开始 prefill
prefilling: prompt 正在被分段处理
running: 已经完成 prefill，正在 decode
paused: 暂时不参与本轮计算，但状态保留
finished: 已经结束，资源可以释放
```

状态转换必须非常谨慎。比如一个 running sequence 被暂停，如果释放了它的 KV cache，那么恢复时就需要重新 prefill 历史 token；如果保留 KV cache，就会继续占显存。调度策略本质上是在时间、显存和公平性之间做取舍。

## 6. 原理详解：Batching window

服务端常用一个很短的 batching window，例如几毫秒。第一个请求到达后，服务端稍等一小段时间，收集更多请求一起处理。

窗口太短，合批效果弱；窗口太长，TTFT 增加。这个参数体现了吞吐和延迟的 tradeoff。

continuous batching 的效果通常要同时看三类指标：

- 吞吐：requests/s 或 output tokens/s 是否提高。
- 首 token 延迟：新请求进入系统后多久看到第一个 token。
- 每 token 延迟：已经在生成的请求是否稳定输出。

一个策略可能提高吞吐，但让 TTFT 变差；也可能降低 TTFT，但牺牲 decode batch size，导致总体 tokens/s 下降。调度策略没有脱离 workload 的绝对最优。

还有一个常见误解：客户端看到多个 streaming 连接同时存在，不等于模型内部做了 token-level continuous batching。真正的 continuous batching 要看模型执行时是否把多个 sequence 放进同一轮 prefill/decode batch。

举例说明 batching window 的取舍。假设请求到达时间如下：

```text
t = 0 ms: 请求 A 到达
t = 1 ms: 请求 B 到达
t = 2 ms: 请求 C 到达
```

如果 batching window 是 0 ms，A 可能立刻开始，B 和 C 赶不上这一批，batch size 较小。A 的 TTFT 低，但整体吞吐可能低。

如果 batching window 是 3 ms，A 会等到 B 和 C，一起组成更大的 batch。吞吐可能提高，但 A 的 TTFT 至少多等了几毫秒。

这几毫秒看起来很小，但在线系统的延迟是很多部分累加起来的：排队、batching window、prefill、decode、网络传输都会影响用户感知。理解 continuous batching 时，不能只看单次 kernel 快不快，还要看整个请求生命周期。

## 7. 本节小结

需要理解 waiting/running 队列、sequence budget、token budget。连续批处理的第一目标不是追求某个固定最优策略，而是让 prefill/decode 的批处理语义正确，并能解释吞吐和延迟之间的取舍。

调度器每一轮至少要回答：

- 哪些新请求进入 prefill？
- 哪些已有请求进入 decode？
- 本轮 token budget 是否够？
- 长 prompt 是否需要 chunked prefill？
- 已完成请求什么时候离开 running queue？
- 被抢占的请求如何保留或释放状态？

这些问题共同决定吞吐和延迟。如果只追求更大的 batch，可能让新请求 TTFT 变差；如果只追求低 TTFT，可能让 GPU 利用率下降。学完本节后，应能解释 waiting queue、running queue、prefill batch、decode batch、token budget 和 batching window 分别解决什么问题。
