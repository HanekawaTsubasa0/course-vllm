# 第 9 章：在线推理引擎的请求状态与资源生命周期

## 9.1 模型执行与请求编排的职责分离

模型 backend 最擅长回答：

```text
给定 token IDs 和 cache，如何运行 prefill/decode 并返回 logits？
```

但一个在线请求还包含：

```text
prompt 文本与 chat template
sampling 参数
生成 token 列表
EOS / stop / max_tokens
streaming
取消与错误
cache release
请求 ID 与返回顺序
```

如果这些逻辑散落在 HTTP route、模型类和客户端中，单请求可能勉强运行，一旦加入 batch、streaming 和取消，状态很容易互相矛盾。

Engine 的职责是把模型的一步计算组织成完整请求生命周期。它是模型数值执行与服务协议之间的编排层。[vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html)同样区分 online/offline entrypoints、LLM engine、worker、model runner 和模型本体；具体进程与类会随版本改变，本章关注这些边界所承担的稳定职责。

---

## 9.2 Protocol、Engine、Scheduler 与 Backend

```mermaid
flowchart LR
    P["Protocol / HTTP\n解析请求与返回事件"] --> E["Engine\n请求生命周期"]
    E --> S["Scheduler\n选择本轮 sequence"]
    E --> B["Model Backend\nprefill/decode"]
    S --> B
    B --> C["KV Cache\n保存历史状态"]
    B --> E
```

### 9.2.1 Protocol 层

负责 JSON schema、HTTP 状态、SSE 格式、chat messages 等外部契约。它不应实现 token-by-token 模型循环。

### 9.2.2 Engine

负责编码 prompt、创建请求状态、调用 backend、sampling、检查结束、发出 token/finished event 和清理资源。

### 9.2.3 Scheduler

决定本轮哪些 sequence prefill/decode。Week 09 先关注单请求状态；Week 11 再展开多请求调度。

### 9.2.4 Backend

封装模型与 cache 的数值执行：

```text
encode/decode text
prefill
decode_step / decode_batch
release_cache
```

边界清晰后，reference backend 与 course backend 可以替换，而 Request 语义不必重写。

## 9.3 Request 与 Sequence 的状态粒度

### 9.3.1 Request

表示用户提交的一次任务：

```text
request_id
prompt / messages
sampling_params
arrival metadata
```

### 9.3.2 Sequence

表示某一条候选 token 路径：

```text
prompt_token_ids
generated_token_ids
next_token_id
cache_handle
prefill progress
status
finish_reason
```

当前课程一次 request 通常只有一条 sequence，但分开仍有价值：beam search、parallel sampling 或 best-of 可能让一个 request 派生多条候选 sequence；scheduler 更关心 sequence 的长度、cache 和状态，而 HTTP 更关心 request。

### 9.3.3 Prompt Tokens 与 Generated Tokens 的独立计数

停止长度通常指生成 token 数，不是总上下文长度；输出给用户也通常只包含 generated 部分。分开存储能避免：

```text
max_tokens 误把 prompt 算进去
返回文本重复 prompt
位置与 cache length 计算混乱
```

总上下文可按需得到：

```text
context = prompt_token_ids + generated_token_ids
```

## 9.4 输入规范化与边界检查

### 9.4.1 普通 Prompt

```text
string -> tokenizer.encode -> prompt_token_ids
```

### 9.4.2 Chat Messages

```text
messages
-> apply_chat_template
-> rendered prompt string or token IDs
-> model input
```

Engine 必须明确 tokenizer/backend 边界：不同模型可能需要不同 chat template、BOS/EOS 处理和特殊 token。

### 9.4.3 空输入与上下文长度约束

应定义：

- 空字符串是否允许；
- tokenizer 后为空怎样处理；
- prompt 超过上下文上限时拒绝还是截断；
- 截断方向和用户可见提示。

静默截断可能改变问题语义，在线服务通常更适合明确报错或由调用者选择策略。

## 9.5 Prefill 输出与首 Token 采样

Prefill 处理整个 prompt，并返回最后位置 logits。Sampling 这份 logits 就得到第一个生成 token。

因此主循环不是：

```text
prefill
-> decode
-> first token
```

而是：

```text
prefill
-> sample first token
-> append/check/emit
-> 如果未结束，再用 first token 做 decode
```

这个顺序直接影响 `max_tokens=1`：合理行为是返回 prefill logits 采出的一个 token，然后结束；如果 prefill 后先检查 generated length，可能错误返回空结果。

## 9.6 单请求生成循环的状态转换

```mermaid
flowchart TB
    A["创建 Request/Sequence"] --> B["encode prompt"]
    B --> C["backend.prefill"]
    C --> D["sample first token"]
    D --> E["append token"]
    E --> F["检查停止条件"]
    F -- "结束" --> G["emit finished"]
    F -- "继续" --> H["emit token"]
    H --> I["backend.decode_step(latest token, cache)"]
    I --> J["sample next token"]
    J --> E
    G --> K["release cache"]
```

伪代码：

```python
handle = None
try:
    prompt_ids = backend.encode(prompt)
    logits, handle = backend.prefill(prompt_ids)

    while True:
        token = sampler.sample(logits, params)
        sequence.append(token)

        reason = sequence.check_finish(token, params)
        if reason:
            yield finished_event(reason)
            break

        yield token_event(token)
        logits, handle = backend.decode_step(token, handle)
finally:
    if handle is not None:
        backend.release_cache(handle)
```

真实实现可能在 finished 前仍发送最后 token，也可能 event schema 不同，但状态顺序必须自洽。

## 9.7 Token 追加与终止条件

### 9.7.1 EOS

EOS 是模型词表中的特殊 token。是否把 EOS 文本返回给用户要由协议定义，通常它只作为结束标记，不显示。

### 9.7.2 Stop Token IDs

用户或模型配置可以指定多个 stop token。采样到其中之一后结束。

### 9.7.3 `max_tokens`

通常限制 generated token 数：

```text
len(generated_token_ids) >= max_tokens
```

`max_tokens=None` 表示不按生成长度停止，但仍受 EOS、stop、上下文上限、超时和资源保护约束。生产服务不应允许完全无界请求破坏容量。

### 9.7.4 Stop String 与 Stop Token

字符串可能跨多个 token；只检查最新 token ID 无法识别任意 stop string。课程核心协议使用 stop token IDs，若扩展字符串停止，需要维护增量解码缓冲和跨 token 匹配。[Transformers `StopStringCriteria`](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.StopStringCriteria)列出的跨 token “overhang”情形说明，停止字符串匹配必须基于 tokenizer 可能产生的多种分段，而不能假定一个字符串对应一个 token。

### 9.7.5 Finish Reason

至少区分：

```text
stop       EOS 或 stop token
length     达到 max_tokens/context limit
cancelled  客户端取消
error      执行失败
```

调用者需要知道回答是否被截断。

## 9.8 Streaming 事件模型

Token event：

```json
{"event":"token","request_id":"r1","token_id":42,"text":"..."}
```

Finished event：

```json
{"event":"finished","request_id":"r1","finish_reason":"stop"}
```

最后协议层可以发送：

```text
data: [DONE]
```

### 9.8.1 Token 边界与增量文本边界

一个 token 可能是单词片段或字节片段。单独 decode 每个 token 再拼接，可能与一次 decode 完整 token 列表不同，尤其涉及空格、Unicode 和 tokenizer cleanup。

可靠方式之一是维护已解码前缀：

```text
new_full_text = decode(all_generated_ids)
delta = new_full_text[len(previous_text):]
```

代价是重复 decode；更高效实现使用 tokenizer 的增量解码能力。课程应明确当前选择和边界。

### 9.8.2 终止 Token 与 Finished Event 的顺序

如果 token 同时触发长度结束，它通常仍是有效生成结果，应先产生 token delta，再 finished。EOS/stop token 是否隐藏则由协议定义。顺序必须测试。

## 9.9 Streaming 与 Non-streaming 的共享生成内核

Engine 可以内部产生相同 token events：

- streaming consumer 逐个转成 SSE；
- non-streaming consumer 收集 token，最后组装 response。

```mermaid
flowchart LR
    E["Engine token events"] --> S["Streaming adapter\n立即 SSE"]
    E --> N["Non-stream adapter\n收集到结束"]
    S --> C1["客户端逐步显示"]
    N --> C2["客户端一次收到完整文本"]
```

这样停止条件和模型逻辑只有一份，减少两种接口行为不一致。

## 9.10 取消传播与资源回收

用户关闭页面、客户端超时、上游取消请求都很常见。若服务端继续生成：

- 浪费 GPU 时间；
- KV cache 继续增长；
- 占用 batch 位置；
- 最终结果无人消费。

取消需要从协议层传播到 engine/scheduler，并最终 release cache。

### 9.10.1 不同执行阶段的取消语义

```text
waiting：从队列删除，不应分配 cache
prefill 中：当前 GPU 工作可能无法立刻中断，完成后不再继续
decode 间：下一调度迭代移除并释放 cache
stream write：客户端断开时触发上游取消
```

状态转换应幂等：重复收到取消不能释放其他请求资源或发送多个 finished。

## 9.11 异常安全与统一清理路径

Cache 可能在 prefill 成功后创建，随后 sampling、detokenize 或网络发送失败。只在正常循环末尾 release 会泄漏。

资源管理应遵循：

```text
谁创建，谁保证最终释放
```

或由统一 owner 管理，并在所有终止路径调用幂等 release。

测试可以注入异常：

```text
prefill 后 sampler 抛错
第 3 个 token 时取消
stream consumer 抛错
backend decode 抛错
```

每次都确认 cache usage 和 sequence registry 回到基线。

## 9.12 Batch 内 Sequence 的独立终止

Batch A/B/C 输出长度不同：

```text
A 第 2 轮 EOS
B 第 10 轮 length
C 第 5 轮 cancelled
```

不能等全部结束后才释放 A，也不能因 A 结束而终止 B/C。

Batch result 还要恢复原请求顺序。Scheduler 为效率可能重排、按长度分桶，但 API response 必须通过 request ID 或原始 index 对齐。

## 9.13 单调状态转换与非法操作

一种简单状态：

```text
WAITING -> PREFILL -> RUNNING -> FINISHED
                       |            |
                       -> CANCELLED -
                       -> ERROR -----
```

合法性规则：

- FINISHED 不再 append token；
- release 后不能 decode；
- prefill 只发生一次，除非明确支持 chunk/resume；
- generated length 只增加；
- finish_reason 一旦设置不随意覆盖。

显式状态能让错误尽早暴露，而不是依赖多个布尔值的偶然组合。

## 9.14 Engine 与 Scheduler 的接口边界

单请求 `generate_stream` 可以自己循环；多请求 continuous batching 时，scheduler 每轮选择一组 sequences，engine 执行 batch step 并更新每条状态。

稳定接口可以是：

```text
schedule() -> batch metadata
execute(batch) -> logits/cache updates
sample/update sequences
release finished
```

Week 09 要把 sequence 状态做正确，Week 11 才能安全地改变选择顺序。

## 9.15 请求生命周期的 Profiling 边界

本周建立单请求基线：

```text
encode
prefill
first sampling
each decode step
detokenize/event
release
```

不要为了计时在每个 CUDA op 后同步。可以用 profiler range 标注阶段，再观察真实异步 timeline。

基线要固定 prompt、output tokens、dtype、backend 和 sampling，作为 Week 10/11 更换 cache/调度后的对照。

## 9.16 请求状态机测试矩阵

### 9.16.1 正常终止

- 第一个 token EOS；
- 多步后 EOS；
- stop token；
- max_tokens=1；
- max_tokens=N。

### 9.16.2 Streaming

- token event 顺序；
- finished 恰好一次；
- `[DONE]` 在 finished 后；
- 最后 token 的 delta 正确。

### 9.16.3 错误与取消

- prefill 异常；
- decode 异常；
- consumer 取消；
- release 恰好且幂等。

### 9.16.4 Batch

- 不同结束轮次；
- 输出恢复输入顺序；
- A 结束不影响 B；
- 每条 finish_reason 正确。

## 9.17 请求语义的常见混淆

### 9.17.1 Prefill 已经产生首 Token 所需 Logits

Prefill 最后位置 logits 已能采样首 token。

### 9.17.2 返回方式不应复制生成循环

共享 event 生成逻辑更容易保持语义一致。

### 9.17.3 生成上限与上下文上限

API 中通常指生成 token 数；模型上下文上限是另一约束。

### 9.17.4 客户端断开不会自动停止模型执行

不会。取消必须传播到 engine/scheduler。

### 9.17.5 Generator 退出不构成 Cache 释放协议

只有资源 owner 在 finally 或统一生命周期中明确 release 才可靠。

### 9.17.6 Batch 不共享单一终止条件

每条 sequence 都有独立 EOS、长度和取消状态。

## 9.18 本章小结与思考题

1. Protocol、Engine、Scheduler、Backend 各负责什么？
2. Request 与 Sequence 为什么分开？
3. Prefill 后首 token 的采样顺序是什么？
4. max_tokens=1 应怎样工作？
5. Stop token 与 stop string 有何区别？
6. Streaming 增量文本为什么可能跨 token？
7. 取消如何传播并保证 cache release？
8. Batch 重排后怎样恢复原请求顺序？
9. 哪些测试能证明状态机没有泄漏？

## 参考资料

- vLLM：[Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html)
- vLLM：[Engine API](https://docs.vllm.ai/en/latest/api/vllm/index.html)
- Hugging Face Transformers：[Stopping Criteria](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.StoppingCriteria)
- Yu 等：[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- 课程工程：[`engine.py`](../../../course_vllm/engine/engine.py)、[`request.py`](../../../course_vllm/engine/request.py)、[`protocol.py`](../../../course_vllm/server/protocol.py) 与 [`api.py`](../../../course_vllm/server/api.py)
