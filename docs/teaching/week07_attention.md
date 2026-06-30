# Week 07: Attention

## 0. 本节学习目标

Week07 聚焦 attention 计算本身：Q/K/V 从哪里来，attention score 怎么算，为什么要做 causal mask，prefill attention 和 decode attention 为什么不是同一种工程形态，以及 FlashAttention 的基本思想是什么。

KV cache 的生命周期会在 Week08 讲，paged KV 的 block 管理会在 Week10 讲。本周只把它们当作 attention 读取历史 K/V 的背景，不提前展开分页管理。

## 1. Self-attention 基本公式

Transformer attention 从 hidden states 计算 Q、K、V：

```text
Q = X W_q
K = X W_k
V = X W_v
```

attention 输出：

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

其中 `d` 是 head dimension。`Q K^T` 计算 query 和 key 的相似度，softmax 得到权重，再对 value 加权求和。

多头 attention 把 hidden dimension 分成多个 head，每个 head 独立做 attention，最后拼接。

## 2. Causal mask

自回归语言模型不能看未来 token。prefill 时，prompt 中所有 token 一次输入模型，但第 i 个 token 只能看第 1 到 i 个 token，不能看后面的 token。

这需要 causal mask：

```text
score[i, j] = -inf if j > i
```

decode 时每步只有一个新 token，它天然只看历史和自己，因此不需要构造完整的上三角 mask，但仍然要保证只能读合法历史长度。

## 3. Prefill attention 和 decode attention 的差异

prefill attention 输入通常是：

```text
Q, K, V: [batch, heads, seq_len, head_dim]
```

它要计算 `seq_len x seq_len` 的 attention score。

decode attention 输入通常是：

```text
Q: [batch, heads, head_dim]
K/V cache: 历史所有 token
```

它只为当前 token 算一次 query 对历史 keys 的 attention。每步 score 长度等于当前 context length。

这导致两类 kernel 的形态不同。prefill 更像大矩阵/块状 attention，decode 更像按请求读取变长 KV cache。

## 4. GQA

Grouped-query attention, GQA, 指 Q heads 数量大于 KV heads 数量。多个 Q heads 共享同一个 KV head。这样可以减少 KV cache 大小和带宽需求。

如果：

```text
num_heads = 16
num_kv_heads = 8
```

那么每 2 个 query heads 共享 1 个 KV head。实现 paged attention 时必须正确把 query head 映射到 KV head。

## 5. FlashAttention 的基本思想

普通 attention 的直接实现会显式构造 `Q K^T` 矩阵。对于长度为 `seq_len` 的序列，每个 head 的 score 矩阵大小是：

```text
[seq_len, seq_len]
```

当序列很长时，这个矩阵会占用大量显存，并且会带来大量 HBM 读写。FlashAttention 的核心思想是 IO-aware attention：不把完整 score 矩阵写回显存，而是在片上 SRAM/shared memory 中分块计算，并用在线 softmax 维护每一行的最大值和归一化分母。

可以把它理解为三件事：

第一，分块读取 Q、K、V。每次只处理一小块 query 和一小块 key/value，让数据尽可能在片上复用。

第二，边算 score 边更新 softmax 的统计量。由于 softmax 需要全行的最大值和指数和，不能简单把每块独立 softmax 后拼起来。FlashAttention 使用 online softmax，在看到新块时修正旧块的归一化比例。

第三，减少显存中间结果。普通实现可能写出 attention score、softmax probability，再读回来乘 V。FlashAttention 尽量只写最终输出，减少 HBM traffic。

FlashAttention 没有改变 attention 的数学定义，改变的是计算顺序和内存访问方式。学习顺序上，先把 dense prefill attention 和 decode attention 写正确，再理解 FlashAttention 为什么能减少显存读写。

## 6. Decode attention 与历史 K/V

decode 阶段每次只有一个新 query，但它要看完整历史 K/V。历史 K/V 可以来自连续 KV cache，也可以来自后续 Week10 的 paged KV cache。

无论 K/V 怎么存，attention 数学仍然是：

```text
score = q k^T / sqrt(d)
prob = softmax(score)
out = prob v
```

区别在于 kernel 如何找到第 `t` 个历史 token 的 K/V 地址。本节先理解 decode attention 要按有效历史长度读取 K/V；分页寻址的细节留到 Week10。

## 7. 实现时最容易错的地方

需要保证 dense attention 对齐 PyTorch reference，paged decode attention 对齐把 K/V 读成 dense 后的 reference。

attention 的错误通常来自四类地方：

- mask 错误：prefill 中未来 token 没有被正确屏蔽。
- head 映射错误：GQA 中 query head 没有映射到正确的 KV head。
- scale 错误：score 没有除以 `sqrt(head_dim)`。
- 长度错误：decode 时读取了超过有效历史长度的 K/V。

## 8. 实验中的少量对照

实验会分别对齐 dense prefill attention 和 decode attention。阅读代码时重点看 Q/K/V shape、causal mask、GQA head 映射、历史长度和 reference 对齐。
