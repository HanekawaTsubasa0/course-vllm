# Week 09: 推理 Engine 与请求生命周期

## 0. 本节学习目标

Week09 聚焦 engine 如何管理一次生成请求。前面周次已经分别讲了模型算子、采样和 KV cache；本周把这些组件组织成状态机。重点不是新的数学公式，而是请求状态、sequence 状态、停止条件和 streaming event。

## 1. 为什么需要 engine

模型 backend 只知道“给 token，跑 forward”。但在线生成请求需要更多逻辑：

- prompt encode。
- prefill。
- decode loop。
- sampling。
- max_tokens。
- EOS/stop token。
- streaming 输出。
- cache release。
- batch 请求保持原始顺序。

这些逻辑不应该散落在 HTTP 层，也不应该塞进模型层，因此需要 engine。

## 2. Request 和 Sequence

Request 表示用户提交的一次任务，包含 prompt 和 sampling 参数。Sequence 表示这次任务的 token 状态。

Sequence 需要保存：

- prompt token ids。
- generated token ids。
- past_key_values handle。
- next token id。
- finish reason。
- prefill progress。

把 Request 和 Sequence 分开，是为了后续支持更复杂的调度，例如一个 request 可能有多个候选 sequence，或者 scheduler 只关心 sequence 的长度和状态。

## 3. 停止条件

生成不可能无限循环。常见停止条件：

- 生成了 EOS token。
- 生成了用户指定 stop token。
- 达到 `max_tokens`。
- 服务端因为错误或资源限制中止。

停止条件必须在每次 append token 后检查。否则可能多生成 token，或者无法及时释放 cache。

## 4. Streaming event

streaming 模式下，每生成一个 token 就产生一个 event。这个 event 至少包含：

```text
event type
request_id
token_id
text
```

最后还需要 finished event，告诉客户端结束原因和完整 token 列表。

## 5. 一次请求的完整状态变化

单请求可以抽象成这样的状态变化：

```text
waiting
-> prefill
-> first token sampled
-> decode loop
-> token appended
-> finished
-> cache released
```

流式返回时，每生成一个 token 就可以产生一个 event；非流式返回时，系统可以把这些 token 收集起来，最后一次性返回完整文本。

batch 请求只是把多个 sequence 放在同一个调度周期里处理。概念上仍然要保持每个 sequence 自己的状态：

```text
sequence A: prefill -> decode -> finished
sequence B: prefill -> decode -> decode -> finished
sequence C: waiting -> prefill -> decode -> finished
```

## 6. 实现主循环时要想清楚什么

需要保证：

- token id 被追加到 generated list。
- request/sequence 状态正确更新。
- finish reason 能被上层看到。

调度策略会在 Week11 展开，这里先把单个请求的生命周期和状态变化理清楚。

## 7. 实验中的少量对照

实验会把 request、sequence、sampling、finish reason 和 streaming event 串起来。阅读代码时重点看状态是否单调推进：token 追加、结束条件、cache 释放和返回事件不能互相矛盾。
