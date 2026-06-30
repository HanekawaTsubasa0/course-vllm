# Week 11: Continuous Batching

## 0. 本节学习目标

Week11 专门讲请求调度和合批。前面 Week01 提过 batching 的动机，本周要把 continuous batching 的机制讲细：waiting queue、running queue、prefill batch、decode batch、iteration-level scheduling，以及吞吐和延迟的 tradeoff。

## 1. 为什么普通 batching 不够

普通 fixed batching 的思路是收集一批请求，一起开始，一起结束。它适合输入输出长度相近的任务，例如图像分类。

LLM 生成不适合 fixed batching：

- prompt 长度不同。
- 输出长度不同。
- 有的请求很快 EOS，有的请求生成很久。
- decode 是逐 token 循环。

如果固定 batch 必须等最慢请求结束，短请求完成后也占着 batch 位置，GPU 利用率下降。

## 2. Continuous batching

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

## 3. Prefill 和 decode 的调度差异

prefill 请求的 token 数可能很大。一个长 prompt prefill 会占用大量计算，可能阻塞 decode 请求，导致正在生成的用户 token 变慢。

decode 请求每个 sequence 通常只贡献一个 token，但数量可能很多。decode batch 越大，GPU 利用率通常越好。

调度器必须在两者之间平衡：

- 优先 prefill 可以降低新请求 TTFT。
- 优先 decode 可以降低正在生成请求的 TPOT。
- chunked prefill 可以把长 prompt 切开，避免一次 prefill 占用过久。

## 4. 队列策略、抢占与 chunked prefill

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

## 5. Batching window

服务端常用一个很短的 batching window，例如几毫秒。第一个请求到达后，服务端稍等一小段时间，收集更多请求一起处理。

窗口太短，合批效果弱；窗口太长，TTFT 增加。这个参数体现了吞吐和延迟的 tradeoff。

continuous batching 的效果通常要同时看三类指标：

- 吞吐：requests/s 或 output tokens/s 是否提高。
- 首 token 延迟：新请求进入系统后多久看到第一个 token。
- 每 token 延迟：已经在生成的请求是否稳定输出。

一个策略可能提高吞吐，但让 TTFT 变差；也可能降低 TTFT，但牺牲 decode batch size，导致总体 tokens/s 下降。调度策略没有脱离 workload 的绝对最优。

还有一个常见误解：客户端看到多个 streaming 连接同时存在，不等于模型内部做了 token-level continuous batching。真正的 continuous batching 要看模型执行时是否把多个 sequence 放进同一轮 prefill/decode batch。

## 6. 实现调度器时要想清楚什么

需要理解 waiting/running 队列、sequence budget、token budget。实现目标不是工业最优策略，而是让 prefill/decode 批处理语义正确。

调度器每一轮至少要回答：

- 哪些新请求进入 prefill？
- 哪些已有请求进入 decode？
- 本轮 token budget 是否够？
- 长 prompt 是否需要 chunked prefill？
- 已完成请求什么时候离开 running queue？
- 被抢占的请求如何保留或释放状态？

这些问题共同决定吞吐和延迟。如果只追求更大的 batch，可能让新请求 TTFT 变差；如果只追求低 TTFT，可能让 GPU 利用率下降。

## 7. 实验中的少量对照

实验会通过多个并发请求观察合批效果。阅读代码时重点看 waiting queue、running queue、prefill batch、decode batch 和统计指标之间的关系。
