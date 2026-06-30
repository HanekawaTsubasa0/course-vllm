# Week 04: RMSNorm 与 RoPE

## 0. 本节学习目标

Week04 讲两个 transformer 模型内部的基础算子：RMSNorm 和 RoPE。它们都出现在 Qwen3 的每层 forward 中，但作用不同：RMSNorm 控制 hidden states 的尺度，RoPE 给 attention 注入位置信息。

本周不深入讲完整 attention，也不讲 matmul 优化。重点是把数学公式、tensor shape、CUDA dispatch 三件事讲清楚。

## 1. RMSNorm 的背景

Transformer 层之间不断做线性变换、attention、MLP 和残差连接。如果 hidden states 的尺度在层间不断漂移，模型计算会不稳定。归一化层的作用就是稳定激活尺度。

LayerNorm 的公式通常是：

```text
mean = average(x)
var = average((x - mean)^2)
y = (x - mean) / sqrt(var + eps) * weight + bias
```

RMSNorm 去掉了减均值，只使用 root mean square：

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight
```

它计算更简单，很多 LLM 使用 RMSNorm 或其变体。RMSNorm 对每个 token 的 hidden vector 独立计算。假设输入：

```text
x: [tokens, hidden]
```

那么每一行单独求均方：

```text
mean_square[row] = sum_j x[row, j]^2 / hidden
```

再对该行每个 hidden 位置缩放。

## 2. RMSNorm 的 CUDA 思路

RMSNorm 是 row-wise reduction 加 elementwise scaling。它天然分两步：

1. 对一行 hidden 维做平方和 reduction。
2. 用 `rsqrt(mean + eps)` 缩放这一行每个元素。

基础 CUDA 实现可以让一个 block 负责一行，block 内线程分担 hidden 维的累加。更高性能实现会考虑 warp-level reduction、向量化 load/store、half/bfloat16 accumulate 到 float32 等细节。

数值上要注意：即使输入是 fp16/bfloat16，平方和通常用 float32 累加，避免误差过大。

## 3. RoPE 的背景

Attention 本身只看 token 内容，不天然知道位置。位置编码的作用是告诉模型 token 的顺序。

早期 transformer 使用绝对位置 embedding，把位置向量加到 token embedding 上。RoPE, Rotary Position Embedding, 采用另一种方式：对 Q 和 K 做位置相关旋转，使得 Q/K 点积中包含相对位置信息。

RoPE 的常见实现可以写成：

```text
rope(x) = x * cos(pos) + rotate_half(x) * sin(pos)
```

`rotate_half` 把最后一维拆成两半或成对维度后做旋转。直观上，二维平面里的向量乘以旋转矩阵：

```text
[x1'] = [ cos  -sin ] [x1]
[x2']   [ sin   cos ] [x2]
```

多维 RoPE 可以看成很多二维子空间同时旋转。

RoPE 只作用于 Q 和 K，不作用于 V。原因是 attention score 来自 Q 和 K 的相似度，位置信息应该影响“看哪里”，而 V 是被加权汇总的内容。

## 4. RoPE 的 CUDA 思路

RoPE 是 elementwise 变换。每个输出元素依赖：

- 原始 `x` 的对应元素。
- `rotate_half(x)` 的配对元素。
- 对应位置的 cos。
- 对应位置的 sin。

它没有跨行 reduction，比 RMSNorm 更像纯 elementwise kernel。关键是正确处理最后一维配对关系，以及 cos/sin 的广播 shape。

基础实现通常把输入展平成二维：

```text
rows x head_dim
```

每个线程处理一个或多个元素。

## 5. RMSNorm 和 RoPE 在推理中的位置

RMSNorm 通常出现在 transformer block 的 attention 前和 MLP 前，用来稳定 hidden states 的尺度。它不改变 token 数，也不改变 hidden dimension，只对每个 token 的 hidden 向量做归一化和缩放。

RoPE 出现在 attention 计算 Q/K 之后、打分之前。它把位置信息注入 Q 和 K，因此会影响 attention score。RoPE 不作用于 V，因为 V 表示被加权汇总的内容，位置信息主要应该影响“query 看哪些 key”。

从 shape 上看，这两个算子都比较适合按行并行：RMSNorm 的一行是一个 token 的 hidden 向量，RoPE 的一行可以看作某个 token、某个 head 的 head_dim 向量。

## 6. 实现时最容易错的地方

RMSNorm 主要 shape：

```text
x: [tokens, hidden]
weight: [hidden]
out: [tokens, hidden]
```

RoPE 输出 shape 和输入 Q/K 相同。需要保证 Q 和 K 都被同样的位置旋转。

## 7. 实验中的少量对照

实验会把 RMSNorm 和 RoPE 的 CUDA 实现与 PyTorch reference 对齐。阅读代码时重点关注输入输出 shape、每个线程负责的数据范围、累加精度和位置索引是否一致。
