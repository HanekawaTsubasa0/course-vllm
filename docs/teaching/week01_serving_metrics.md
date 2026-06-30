# Week 01: LLM Serving 流程与指标

## 0. 本节学习目标

学习 LLM serving 不能停留在“调用一个模型 API”。训练好的模型只是服务系统里的一个组件。真正的 LLM serving 系统还要处理请求协议、tokenizer、prefill、decode、KV cache、batching、sampling、streaming、队列、指标、显存管理和故障隔离。

第一周先建立一个完整图景：用户发来的不是“一个函数调用”，而是一个在线系统任务。这个任务既有模型计算，也有系统调度；既有数学上的自回归生成，也有工程上的吞吐和尾延迟；既要考虑单个请求能不能返回，也要考虑很多请求同时进来时服务端是否稳定。

本周最重要的结论是：

```text
LLM serving = 自回归生成算法 + 在线服务系统 + GPU 资源管理。
```

如果只理解自回归生成，不理解 serving 指标，就无法解释为什么需要 batching、paged KV、admission control。如果只理解 HTTP 服务，不理解 prefill/decode，就无法解释为什么 LLM 的请求形态和普通 Web API 完全不同。

## 1. 从语言模型到在线服务

### 1.1 语言模型本身在做什么

自回归语言模型的基本任务是根据已有 token 预测下一个 token。给定上下文：

```text
x_1, x_2, ..., x_t
```

模型输出一个 logits 向量：

```text
logits_t in R^{vocab_size}
```

这个向量的每个位置对应词表中一个 token 的分数。经过 softmax 后得到概率分布：

```text
P(x_{t+1} | x_1, ..., x_t)
```

生成时，系统从这个分布里选择一个 next token。选择方式可能是 greedy decoding，也可能是带随机性的 sampling。选出 `x_{t+1}` 后，把它拼回上下文，再预测 `x_{t+2}`。这个循环不断重复，直到遇到 EOS、stop token，或者达到 `max_tokens`。

所以 LLM 的一次生成不是一次 forward，而是：

```text
encode prompt
-> forward
-> sample next token
-> forward
-> sample next token
-> ...
-> stop
```

这就是它和普通分类模型、embedding 模型、图像分类服务的根本区别。

### 1.2 在线服务比离线推理复杂在哪里

离线推理只关心“给一个 prompt，最后生成什么”。在线 serving 还要关心：

- 用户何时看到第一个 token。
- 同时有多少请求在排队。
- GPU 是否被充分利用。
- 请求太长时是否拒绝。
- 某个请求失败是否影响其他请求。
- 多个请求能否合批。
- 服务端是否能输出 streaming response。
- 显存中的 KV cache 是否会爆。

普通 Web 服务经常是 CPU 处理、数据库查询、返回 JSON。LLM 服务则是 GPU-bound 和 memory-bound 混合问题。每个请求的运行时间也不是固定的：prompt 长度不同，输出长度不同，停止时间不同，KV cache 增长速度也不同。

因此，学习 LLM serving 不能只从模型结构讲起，还必须从服务指标和请求生命周期讲起。

## 2. Prompt、Token、Tokenizer

### 2.1 为什么服务端首先要分词

用户发来的 prompt 是字符串，但 transformer 模型不直接处理字符串。模型处理的是 token id 序列。tokenizer 负责把文本切分成 token，并映射到整数 id。

例如用户输入：

```text
Explain KV cache.
```

服务端大致做：

```text
"Explain KV cache."
-> tokenizer.encode(...)
-> [token_id_1, token_id_2, ..., token_id_L]
```

模型输出的也是 token id。最后需要用 tokenizer decode 回文本。

这个过程对 serving 很重要，因为很多指标按 token 计算，而不是按字符计算。两个 prompt 字符数接近，token 数可能不同；不同语言、空格、标点、代码片段都会影响 token 数。

### 2.2 Chat template

chat 模型通常不是直接把用户消息拼进去，而是用 chat template 把多轮消息转换成模型训练时熟悉的格式。比如：

```text
system: ...
user: ...
assistant: ...
```

会被转换成一个完整 prompt。服务端 `/v1/chat/completions` 的职责之一就是把 messages 转换成模型 prompt。

服务端通常会把 chat messages 先交给 chat template，得到普通 prompt，再进入普通生成路径。这个转换会影响 token 序列，因此也会影响模型输出。

## 3. Prefill 和 Decode

### 3.1 为什么要分成两个阶段

LLM serving 的核心概念是 prefill 和 decode。它们不是两个随便起的名字，而是由自回归 transformer 的计算形态决定的。

假设 prompt 有 `L` 个 token：

```text
x_1, x_2, ..., x_L
```

生成第一个新 token 前，模型需要理解整个 prompt。这个阶段可以一次性把 `L` 个 token 输入模型，执行完整 forward。每一层 attention 会为这 `L` 个 token 产生 key 和 value。这个阶段叫 prefill。

prefill 的输出包括：

- 最后一个位置的 logits，用于采样第一个输出 token。
- 每层的 KV cache，用于后续 decode。

得到第一个输出 token `y_1` 后，下一步不需要重新计算整个 prompt 的 K/V。服务端只输入新 token `y_1`，并把它和历史 KV cache 一起送入模型。这个阶段叫 decode。

decode 每一步产生一个 token：

```text
step 1: input y_1, use KV(prompt), output y_2
step 2: input y_2, use KV(prompt + y_1), output y_3
...
```

### 3.2 Prefill 的计算特点

prefill 的输入长度可能很长，计算形态接近一次普通 transformer forward。它的特点是：

- prompt 内所有 token 可以并行计算。
- 矩阵乘规模较大，GPU 利用率通常较高。
- attention 需要处理 prompt 内 token 之间的 causal attention。
- prefill 直接影响 TTFT。

prefill 越长，用户等待第一个 token 的时间通常越长。长 prompt 请求会显著拖慢 TTFT，也会占用更多 KV cache。

### 3.3 Decode 的计算特点

decode 每一步只处理一个新 token，但需要读取完整历史 KV cache。它的特点是：

- 每步计算量小，但要循环很多次。
- 对内存访问和调度开销敏感。
- 单个请求 decode 时 GPU 可能利用率不高。
- 多请求合批可以提高 decode 阶段吞吐。
- decode 直接影响 TPOT 和 token throughput。

这也是为什么 LLM serving 系统强调 continuous batching。单个请求每次只 decode 一个 token，太小；多个请求合成一个 decode batch，GPU 才更容易被喂满。

## 4. KV Cache 为什么是 serving 的核心

### 4.1 Attention 里的 K/V

Transformer attention 中，每个 token 会产生 query、key、value：

```text
Q = X W_q
K = X W_k
V = X W_v
```

当前位置的 query 会和历史 token 的 key 做点积，得到注意力权重，再加权求和历史 value。简化公式：

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

在自回归 decode 中，历史 token 不会变。历史 token 的 K/V 也不会变。因此，如果每一步都重新计算全部历史 K/V，就是浪费。

### 4.2 KV cache 的作用

KV cache 保存每一层历史 token 的 key/value。这样 decode 时只需要为新 token 计算新的 K/V，再把它 append 到 cache 里。

没有 KV cache：

```text
step t: 重新计算 prompt + y_1 + ... + y_t 的全部 K/V
```

有 KV cache：

```text
step t: 只计算 y_t 的 K/V，并读取历史 KV cache
```

这大幅减少重复计算。但代价是显存占用随上下文长度和并发请求数增长。

每个 token 的 KV cache 大小近似和下面这些量成正比：

```text
num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
```

其中 `2` 表示 K 和 V。上下文越长、并发越高，KV cache 越容易成为显存瓶颈。这就是后续 Week08、Week10 要讲连续 KV cache 和 paged KV cache 的原因。

## 5. Streaming 为什么重要

用户体验上，LLM 服务通常不等完整回答生成完才返回，而是边生成边返回 token。这叫 streaming。

如果不 streaming，用户看到的是：

```text
等待完整生成 10 秒
-> 一次性显示答案
```

如果 streaming，用户看到的是：

```text
等待 1 秒看到第一个 token
-> 后续 token 持续出现
```

即使总生成时间一样，streaming 也会显著改善主观体验，因为 TTFT 更短。

HTTP 上常见实现是 Server-Sent Events, SSE。服务端不断写：

```text
data: {"event": "token", "text": "..."}
```

最后写：

```text
data: [DONE]
```

实际服务中的 streaming 响应通常就是这种形式。

需要注意：streaming 是“返回方式”，continuous batching 是“模型执行调度方式”。二者不是同一个概念。一个服务可以支持 streaming，但 streaming 请求内部仍然串行执行；也可以 non-streaming 请求合批执行。

## 6. Sampling 为什么会导致同样输入输出不同

模型输出 logits 后，服务端要决定下一个 token。常见方式有：

### 6.1 Greedy decoding

选择概率最大的 token：

```text
next = argmax(logits)
```

当 `temperature=0` 时，很多系统会走 greedy path。同样输入通常输出相同。

### 6.2 Temperature sampling

当 `temperature > 0` 时，logits 会被缩放：

```text
probs = softmax(logits / temperature)
```

temperature 越高，分布越平，低概率 token 更容易被采到；temperature 越低，分布越尖锐，输出更确定。

### 6.3 Top-k / Top-p

top-k 只在概率最高的 k 个 token 里采样。top-p，也叫 nucleus sampling，会选出累计概率达到 p 的最小候选集合，然后在这个集合里采样。

这些方法会引入随机数。因此同样 prompt 在 `temperature > 0` 时输出不同是正常现象。要做可复现调试和测试，应优先使用：

```text
temperature = 0
```

或固定 seed。

## 7. Serving 指标详解

### 7.1 TTFT

TTFT 是 Time To First Token。它衡量用户多久能看到第一个输出 token。

TTFT 包含：

- HTTP 请求进入服务端。
- 排队等待。
- tokenizer encode。
- prefill forward。
- 第一次 sampling。
- 第一个 token 写回客户端。

如果 prompt 很长，prefill 时间会增加；如果队列很深，排队时间会增加；如果 batch 策略等待太久，batching window 也会增加 TTFT。

### 7.2 TPOT

TPOT 是 Time Per Output Token。它衡量 decode 持续生成速度。

粗略地说：

```text
TPOT = decode 阶段耗时 / 输出 token 数
```

TPOT 受以下因素影响：

- decode batch size。
- KV cache 读取效率。
- attention kernel。
- sampler 开销。
- GPU 利用率。
- 是否有同步或 CPU 调度瓶颈。

### 7.3 End-to-end latency

end-to-end latency 是请求总耗时：

```text
latency = 请求完成时间 - 请求发出时间
```

它可以近似拆成：

```text
queueing time
+ prefill time
+ decode time
+ response overhead
```

### 7.4 Throughput

request throughput:

```text
requests_per_s = completed_requests / elapsed_s
```

token throughput:

```text
output_tokens_per_s = generated_output_tokens / elapsed_s
```

LLM serving 中 token throughput 通常更有解释力。因为一个请求生成 8 个 token 和生成 512 个 token，成本完全不同。

### 7.5 Tail latency

p50 是中位数，p90 表示 90% 请求不超过这个延迟，p99 表示 99% 请求不超过这个延迟。在线系统非常关注 p99，因为用户往往会感知最慢的那批请求。

平均值可能掩盖问题。例如 99 个请求 0.5 秒完成，1 个请求 30 秒完成，平均值看起来可能还能接受，但 p99 会暴露尾部问题。

### 7.6 elapsed_s

`elapsed_s` 是一次压测从开始到所有请求完成的墙钟时间。它不是单请求 latency，而是一组请求的总完成时间。

例如：

```text
9 个请求顺序执行 elapsed_s = 8.2
9 个请求并发合批 elapsed_s = 3.1
```

说明整组 workload 完成更快。再结合 `requests_per_s` 和 `output_tokens_per_s`，就能说明吞吐提升。

## 8. Batching 的基本动机

GPU 擅长大规模并行计算。单个请求 decode 时，每一步只处理一个 token，计算粒度太小，GPU 可能吃不满。把多个请求合成 batch，可以让同一次 forward 处理更多 token，提高 GPU 利用率。

普通 batching 的问题是请求长度不同、结束时间不同。固定 batch 里如果一个请求很早结束，另一个请求很慢，资源会浪费。LLM serving 里常用更细粒度的 scheduling，例如 iteration-level scheduling 或 continuous batching。它们的思想是：每一轮 decode 都重新组织 batch，让完成的请求退出，让新请求加入。

Week11 会详细讲 continuous batching。Week01 只需要先知道 batching 是为了吞吐，但可能带来排队和尾延迟。

## 9. 在线推理服务的分层视角

一个 LLM serving 系统通常可以分成几层理解。

最外层是协议层。它负责接收 HTTP 请求、解析 JSON、处理 chat message 或普通 prompt，并把请求转换成模型能理解的文本。

第二层是请求管理层。它负责记录一次请求的状态：prompt token 是什么，已经生成了哪些 token，是否已经结束，结束原因是 EOS、长度上限还是异常。

第三层是调度层。它决定哪些请求进入 prefill，哪些请求进入 decode，哪些请求等待，哪些请求可以合成 batch。调度层直接影响 TTFT、TPOT、吞吐和尾延迟。

第四层是模型执行层。它负责真正执行 transformer forward，包括 RMSNorm、RoPE、linear、attention、softmax、KV cache 读写等。

第五层是返回层。它决定一次性返回完整文本，还是边生成边流式返回 token。streaming 改变的是用户看到结果的方式，不等于模型内部一定做了 continuous batching。

用分层视角看系统，后续每周内容就不会散：算子优化发生在模型执行层，KV cache 影响模型执行和显存管理，continuous batching 属于调度层，指标和 profiling 贯穿所有层。

## 10. 本节和后续知识的关系

Week01 建立 serving 全局图景。后续周次会逐步把这条链拆开：

- Week02: 怎么量化性能和定位瓶颈。
- Week03-07: 怎么把关键算子写成 CUDA kernel。
- Week08: 为什么 KV cache 能加速 decode。
- Week10: 为什么连续 KV cache 不够，需要 paged KV。
- Week11: 为什么 continuous batching 能提高吞吐。
- Week12: 怎么做 admission control 和传输优化。
- Week13: 怎么估算多卡容量和通信成本。

如果没有理解 prefill、decode、KV cache、TTFT、TPOT，后面所有系统设计都会变成孤立技巧。因此 Week01 的学习重点不是写代码，而是建立正确的 serving 心智模型。

## 11. 实验中的少量对照

第一周的实验会启动一个最小推理服务，并通过一次请求观察 streaming 输出。读代码时只需要把现象和本节概念对应起来：请求如何进入服务、prompt 如何变成 token、prefill 和 decode 如何交替发生、token 如何返回给客户端。
