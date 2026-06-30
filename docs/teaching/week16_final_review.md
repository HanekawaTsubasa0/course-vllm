# Week 16: 系统总复习

## 0. 本节学习目标

Week16 不引入新理论，而是把整个工程重新组织成一条完整 serving 路径。需要能讲清楚自己实现的每个模块在系统里的位置，以及如何用测试和指标证明它工作。

## 1. 系统主线

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

## 2. 最终整理时要说明什么

最终整理时不能只说明模型能聊天，还要回答：

- 请求从 HTTP 到 token 输出经过哪些模块？
- 哪些代码是自己补的？
- CUDA kernel 是否真实运行？
- 哪些测试证明 correctness？
- batching 是否真的发生？
- 吞吐和延迟如何变化？
- 当前实现和工业系统差距在哪里？

整理结果时要特别区分 correctness evidence 和 performance evidence。correctness evidence 说明代码算得对，例如 CUDA kernel 对齐 PyTorch reference、paged attention 对齐 dense reference。performance evidence 说明系统表现如何，例如 batching 后 requests/s 和 output_tokens/s 提升、p90 latency 变化、CUDA profiler 显示 kernel 真正运行。

这两类证据不能互相替代。模型能输出文本，不等于 CUDA kernel 正确；pytest 通过，也不等于服务吞吐好。

## 3. 系统复习路线

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

这比按文件夹机械记忆更清楚，因为它符合真实请求调用链。

每个模块复习时只需要抓住三个问题：

- 它收到什么输入？
- 它维护什么状态或做什么计算？
- 它把什么输出交给下一个模块？

这样可以防止讲解变成逐行念代码。

## 4. 复习建议

最后复习时，应能把 correctness、CUDA、serving benchmark、batching stats、容量规划和后续优化方向串成一个完整故事：系统为什么这样设计，怎么证明结果正确，性能瓶颈在哪里，优化带来了什么变化，还有哪些限制。
