# Week 06: Softmax 与 Sampling

## 0. 本节学习目标

Week06 聚焦 logits 如何变成下一个 token。前面几周讲的是模型内部算子，本周讲生成决策：softmax、greedy decoding、temperature、top-k、随机采样，以及 softmax CUDA kernel 的 row-wise reduction。

本周不展开完整 attention，也不重复 serving 指标。

## 1. Logits 到概率分布

语言模型最后输出 logits：

```text
logits: [batch, vocab_size]
```

logits 不是概率，可以是任意实数。要把它变成概率分布，通常使用 softmax：

```text
prob_i = exp(logit_i) / sum_j exp(logit_j)
```

softmax 输出满足：

```text
prob_i >= 0
sum_i prob_i = 1
```

每一行对应一个请求或一个 sequence 的下一个 token 分布。

## 2. 数值稳定 softmax

直接计算 `exp(logit)` 可能溢出。例如 logit 很大时，`exp(logit)` 超过浮点可表示范围。softmax 有一个重要性质：对所有 logits 同时减去同一个常数，结果不变。

因此稳定实现会先减去最大值：

```text
m = max_i logit_i
prob_i = exp(logit_i - m) / sum_j exp(logit_j - m)
```

这样最大的指数是 `exp(0)=1`，避免溢出。

CUDA softmax 通常包含三个阶段：

1. 对每行求 max。
2. 对每行求 `sum(exp(x - max))`。
3. 写出归一化结果。

这两个“求 max / 求 sum”都是 reduction。Week06 的 CUDA 重点就是 row-wise reduction。

## 3. 并行归约与 warp 原语

reduction 是把一组数合成一个数的操作，例如求和、求最大值。softmax 至少需要两次 reduction：一次求每行最大值，一次求每行指数和。

在 CUDA 中，reduction 的关键问题是让多个线程协作，同时避免过多同步和低效访存。最基础的写法是每个线程负责若干元素，先把局部结果写入 shared memory，再做树形归约：

```text
thread local result
-> shared memory
-> block-level reduction
-> row max / row sum
```

warp 原语是同一个 warp 内线程通信的指令族，例如 shuffle。它可以让线程直接交换寄存器里的值，减少 shared memory 和 `__syncthreads()` 的使用。对于 vocab 维度不太大、或者每行被一个 block/若干 warp 处理的 softmax，warp-level reduction 是常见优化方向。

学习时先掌握 block-level reduction 的正确性，再理解 warp primitive 为什么能减少同步开销。不要一开始就追求最复杂的 kernel；softmax 的第一目标是数值稳定和结果正确。

## 4. Greedy decoding 与 sampling

如果选择概率最大的 token：

```text
next_token = argmax(logits)
```

这叫 greedy decoding。它通常是确定性的，适合测试和可复现调试。

如果按概率分布随机抽样：

```text
next_token ~ Categorical(probs)
```

这叫 sampling。它会引入随机性。同样 prompt 输出不同，很多时候不是 bug，而是 sampling 的正常结果。

## 5. Temperature

temperature 调整分布尖锐程度：

```text
probs = softmax(logits / temperature)
```

当 temperature 较低时，大 logit 更突出，输出更保守、更确定。当 temperature 较高时，分布更平，模型更容易选到低概率 token，输出更多样但也更不稳定。

很多系统把 `temperature=0` 特殊处理为 greedy decoding，因为除以 0 没有数学意义。

## 6. Top-k

top-k sampling 只保留概率最高的 k 个 token，其余 token 概率置零，再重新归一化。它的作用是限制采样空间，避免模型选到长尾低质量 token。

入门实现通常先做 top-k，再把 top-p 作为相邻概念理解。top-p，也叫 nucleus sampling，是保留累计概率达到 p 的最小 token 集合。

## 7. 实现时最容易错的地方

输入输出：

```text
logits: [batch, vocab]
probs: [batch, vocab]
```

需要保证每行概率和接近 1，并且 CUDA 输出和 PyTorch softmax 对齐。

sampling 还要注意随机性。`temperature=0` 通常走 greedy，因此更适合正确性测试；`temperature>0` 会引入随机采样，同一个 prompt 输出不同是正常现象。固定 seed 只能让某一次采样过程可复现，不代表采样本身变成确定性算法。

## 8. 实验中的少量对照

实验会把 logits 经过稳定 softmax，再根据 greedy 或 top-k sampling 选择下一个 token。阅读代码时重点看三个边界：softmax 是否数值稳定、概率是否归一化、采样参数是否按预期影响输出。
