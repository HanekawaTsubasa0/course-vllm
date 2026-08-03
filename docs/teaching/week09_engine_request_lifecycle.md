# Week 09: 推理 Engine 与请求生命周期

## 1. 本周核心问题

一个 LLM 服务不只是“调用一次模型 forward”。真实生成请求包含 prompt、采样参数、KV cache、生成 token、停止条件、streaming 事件和资源释放。engine 的作用就是把这些环节组织成清晰的状态机。

本周要回答四个问题：

- 为什么模型 backend 和 serving engine 不能混成一层？
- Request 和 Sequence 分别表示什么？
- 一个请求从 waiting 到 finished 会经历哪些状态？
- streaming 输出为什么需要事件模型？

## 2. 背景知识：为什么需要 engine

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

可以把 engine 理解成“生成过程的控制中心”。模型 backend 负责数值计算，sampler 负责从 logits 选 token，KV cache 负责保存历史状态，而 engine 负责安排这些组件按正确顺序工作。

没有 engine 时，常见混乱包括：

- HTTP 层直接管理生成循环，协议逻辑和模型逻辑耦合。
- 模型层知道太多请求状态，难以支持 streaming、取消和批处理。
- 停止条件分散在多个地方，容易多生成或漏释放。
- 请求完成后资源释放不清楚。

## 3. 原理详解：Request 和 Sequence

Request 表示用户提交的一次任务，包含 prompt 和 sampling 参数。Sequence 表示这次任务的 token 状态。

Sequence 需要保存：

- prompt token ids。
- generated token ids。
- past_key_values handle。
- next token id。
- finish reason。
- prefill progress。

把 Request 和 Sequence 分开，是为了后续支持更复杂的调度，例如一个 request 可能有多个候选 sequence，或者 scheduler 只关心 sequence 的长度和状态。

更直观地说：

```text
Request: 用户想要什么
Sequence: 当前已经生成到哪里
```

Request 更接近外部接口，包含用户输入和生成配置。Sequence 更接近模型执行，包含 token ids、KV cache handle、当前位置和是否结束。调度器通常更关心 Sequence，因为 batch 里的基本执行单位是“还需要生成下一个 token 的序列”。

## 4. 原理详解：停止条件

生成不可能无限循环。常见停止条件：

- 生成了 EOS token。
- 生成了用户指定 stop token。
- 达到 `max_tokens`。
- 服务端因为错误或资源限制中止。

停止条件必须在每次 append token 后检查。否则可能多生成 token，或者无法及时释放 cache。

停止条件的顺序也要清楚。一次 decode step 通常是：

```text
模型输出 logits
-> sampler 选出 next token
-> next token 追加到 sequence
-> 检查 EOS / stop / max_tokens
-> 如果结束，产生 finish reason
```

如果在追加 token 前检查停止条件，就可能看不到刚生成的 EOS。如果检查太晚，又可能多生成一个不该返回的 token。状态机的价值就是让这些边界行为明确。

## 5. 原理详解：Streaming event

streaming 模式下，每生成一个 token 就产生一个 event。这个 event 至少包含：

```text
event type
request_id
token_id
text
```

最后还需要 finished event，告诉客户端结束原因和完整 token 列表。

streaming 的本质是把一次完整响应拆成多个增量事件。用户不必等所有 token 生成完才看到结果，这会显著改善感知延迟。非 streaming 模式下，服务端也可能内部逐 token 生成，只是最后一次性返回。

事件模型常见内容包括：

- token event: 新生成一个 token 或一段文本。
- error event: 请求中途失败。
- finished event: 正常结束，并给出 finish reason。

event 必须带 request_id，因为多个请求可能同时生成，客户端或上层协议需要知道每个增量属于哪个请求。

## 6. 原理详解：一次请求的完整状态变化

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

## 7. 本节小结

调度策略会在 Week11 展开，这里先把单个请求的生命周期和状态变化理清楚。学完本节后，应能解释一个请求从 waiting 到 prefill、decode、finished 的状态变化，也能解释为什么 token 追加、结束条件、cache 释放和返回事件不能互相矛盾。
