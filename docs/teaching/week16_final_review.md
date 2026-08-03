# Week 16: 系统总复习

## 1. 本周核心问题

最后一周不引入新的 isolated concept，而是把整门课的知识串成一个完整推理服务系统。复习重点不是背术语，而是能讲清楚：一个用户请求进入服务后，为什么会经过 tokenizer、prefill、decode、KV cache、sampling、scheduler、CUDA kernel、显存管理和系统保护机制。

本周要回答三个问题：

- 一次 LLM serving 请求从输入文本到输出 token，经历了哪些阶段？
- correctness、latency、throughput、capacity 这些指标分别证明什么？
- 算子、缓存、调度和多卡策略如何共同决定系统表现？

## 2. 背景知识：系统主线

完整路径：

```text
HTTP request
-> API protocol
-> batching/admission
-> engine
-> scheduler
-> model backend
-> Qwen3 model
-> CUDA ops
-> KV cache / paged KV cache
-> sampler
-> response
```

这条路径对应前面各周的知识：

- Week01 建立 serving 和指标。
- Week02 建立 profiling 方法。
- Week03-07 实现 CUDA 算子和 attention。
- Week08-10 实现 KV cache 和 paged KV。
- Week11 实现 continuous batching。
- Week12 实现系统边界优化。
- Week13 学会容量估算。
- Week15 理解前沿策略。

复习时要避免按文件夹机械记忆。更好的方式是沿真实请求理解：

1. HTTP 请求进来，协议层如何解析。
2. batching 层如何 admission 和排队。
3. engine 如何创建 Request/Sequence。
4. scheduler 如何决定 prefill/decode。
5. backend 如何调用 Qwen3 模型。
6. 模型层如何调用 RMSNorm、RoPE、Linear、Attention。
7. CUDA wrapper 如何进入 `.cu` kernel。
8. KV cache 如何保存和释放历史 K/V。
9. sampler 如何把 logits 变成 token。
10. response 如何返回给客户端。

这条线能证明自己不是只实现了孤立函数，而是理解了完整 serving 系统。

复习时可以把系统分成四层：

```text
协议层: 接收请求，解析参数，返回 streaming 或非 streaming 响应
调度层: 排队、admission、batching、prefill/decode 调度
模型层: tokenizer、transformer layer、attention、FFN、sampling
硬件层: CUDA kernel、显存、KV cache、数据传输、多卡通信
```

每层都解决不同问题。协议层关注接口语义，调度层关注多请求并发，模型层关注数学计算，硬件层关注性能和资源。一个 serving 系统慢，可能不是模型公式错了，而是排队策略、KV cache 管理、H2D copy、batch size 或多卡通信出了问题。

## 3. 原理详解：如何判断一个系统是否真的正确

复习时不能只说“模型能输出文本”。能输出文本只是最表层现象，还需要区分不同证据：

- correctness evidence: 数学结果是否和 reference 对齐。
- performance evidence: 延迟、吞吐、GPU timeline 是否符合预期。
- capacity evidence: 显存预算和 KV cache token slots 是否足够。
- serving evidence: 多请求并发、streaming、batching、admission 是否按预期工作。

correctness evidence 说明计算结果是对的，例如一个算子的输出能和数学 reference 对齐。performance evidence 说明系统表现如何，例如吞吐、延迟分位数、GPU timeline 和资源利用率。

这两类证据不能互相替代。模型能输出文本，不等于底层计算一定正确；正确性检查通过，也不等于服务吞吐一定好。

要养成这种判断习惯：

```text
结果对不对？       -> correctness
单个请求快不快？   -> latency
并发时吞吐高不高？ -> throughput
显存够不够？       -> capacity
高负载会不会崩？   -> admission / overload behavior
```

这就是工程系统和单个算法函数的区别。单个函数主要验证输入输出；服务系统还要验证多请求、多阶段、多资源之间的相互影响。

## 4. 原理详解：系统复习路线

可以按真实请求路径复习：

```text
HTTP request
-> tokenizer
-> request / sequence state
-> scheduler
-> prefill
-> KV cache write
-> sampler
-> decode loop
-> KV cache read/write
-> streaming or final response
```

这比按模块名称机械记忆更清楚，因为它符合真实请求的执行过程。

每个模块复习时只需要抓住三个问题：

- 它收到什么输入？
- 它维护什么状态或做什么计算？
- 它把什么输出交给下一个模块？

这样可以防止复习变成背名词。学习目标是理解数据如何流动、状态如何变化、资源如何被占用和释放。

也可以按性能瓶颈复习：

```text
prefill 慢
-> prompt 很长？
-> batch 太小？
-> matmul/attention kernel 利用率低？
-> 是否有 prefix cache 复用机会？

decode 慢
-> batch size 太小？
-> KV cache 读取效率低？
-> scheduler 是否让 decode 被长 prefill 阻塞？
-> sampling 是否引入额外开销？

TTFT 高
-> 请求排队久？
-> batching window 太长？
-> prefill 量太大？
-> admission control 是否允许超长请求挤占系统？

吞吐低
-> GPU 利用率低？
-> continuous batching 没有形成大 batch？
-> H2D copy 或 CPU 调度造成空洞？
-> 多卡通信开销是否过高？
```

最后还要按资源复习：

- 计算资源：矩阵乘、attention、FFN、CUDA kernel。
- 显存资源：weights、KV cache、temporary buffers。
- 带宽资源：HBM 读写、KV cache 访问、H2D copy、多卡通信。
- 队列资源：waiting queue、running queue、batch token budget。

一个成熟的 serving 分析通常会把“请求路径”“性能指标”“资源瓶颈”三条线合在一起看。

## 5. 本节小结

最后复习时，应能把 correctness、CUDA、serving benchmark、batching stats、容量规划和后续优化方向串成一个完整故事：系统为什么这样设计，怎么证明结果正确，性能瓶颈在哪里，优化带来了什么变化，还有哪些限制。整门课的核心不是某一个孤立技巧，而是理解算子、缓存、调度、服务接口和 GPU 资源管理如何共同决定 LLM serving 的正确性与性能。
