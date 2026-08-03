# Week 08: KV Cache

## 1. 本周核心问题

自回归生成每次只多生成一个 token，但这个 token 需要看见前面所有历史。KV cache 的目标就是保存历史 token 在 attention 中产生的 K/V，避免每一步都重新计算整段上下文。

本周先学习最容易理解的连续 KV cache：每个 sequence 的 K/V 沿 token 维连续增长。更复杂的分块和分页管理放到后面学习。

## 2. 背景知识：为什么没有 KV cache 会很慢

自回归生成第 t 步需要看前面所有 token。如果没有 KV cache，每一步都要把完整上下文重新送进模型，重新计算所有历史 token 的 K/V。

假设 prompt 长度是 `L`，要生成 `T` 个 token。没有 cache 时，第 1 步处理 `L` 个 token，第 2 步处理 `L+1` 个 token，第 T 步处理 `L+T-1` 个 token。大量历史计算被重复。

有 KV cache 后：

```text
prefill: 计算 prompt 的 K/V 并缓存
decode step: 只计算新 token 的 K/V，读取历史 K/V
```

这把 decode 的重复计算大幅减少。

可以用一个小例子理解。prompt 长度是 3，要继续生成 3 个 token：

```text
没有 KV cache:
step 1 计算 [prompt 3 tokens]
step 2 重新计算 [prompt 3 tokens + generated 1 token]
step 3 重新计算 [prompt 3 tokens + generated 2 tokens]

有 KV cache:
prefill 计算并保存 prompt 3 tokens 的 K/V
step 1 只计算新 token 的 Q/K/V，并读取已有 K/V
step 2 只计算新 token 的 Q/K/V，并读取已有 K/V
step 3 只计算新 token 的 Q/K/V，并读取已有 K/V
```

注意：KV cache 不会消除 attention 对历史 K/V 的读取。它减少的是“重新为历史 token 计算 K/V”的成本。随着上下文变长，decode attention 仍然要读越来越长的历史 K/V，所以 KV cache 同时带来加速和显存/带宽压力。

## 3. 原理详解：KV cache 存什么

每个 transformer layer 都有自己的 attention。每层都要保存历史 token 的 K/V。

一个 cache 可以概念化为：

```text
cache[sequence_id][layer_id] = (key, value)
```

key/value 的常见 shape：

```text
key:   [batch, num_kv_heads, seq_len, head_dim]
value: [batch, num_kv_heads, seq_len, head_dim]
```

对于单个 sequence，`seq_len` 会随着生成增长。

更完整地说，每生成一个新 token，每一层都会产生一份新的 K 和 V。假设有 `num_layers` 层，那么一个 token 不是只占一份 cache，而是在每一层都占一份 K/V。也正因为如此，KV cache 的显存成本会随层数、KV head 数、head_dim、token 数线性增长。

K 和 V 的含义也不同：

- K, key: 用来和当前 query 计算相似度，决定注意力权重。
- V, value: 根据注意力权重被加权求和，形成输出内容。

decode 时当前 token 会产生 query，然后这个 query 会和历史所有 key 打分，再用权重汇总历史 value。

## 4. 原理详解：KV cache 的显存成本

KV cache 的显存成本通常近似：

```text
num_tokens
* num_layers
* 2
* num_kv_heads
* head_dim
* dtype_bytes
```

其中 `2` 是 K 和 V。这个公式非常重要，因为它解释了为什么长上下文和高并发会迅速吃掉显存。

KV cache 是 LLM serving 的核心资源之一。很多 serving 系统的容量上限不是权重，而是 KV cache。

还要注意 batch 内请求长度不同。一个 batch 里可能有的 sequence 长度是 128，有的是 2048。attention 读取 K/V 时不能简单假设所有 sequence 有相同历史长度，否则短请求会读到无效位置，长请求会被截断。

因此 KV cache 通常要同时维护两类信息：

- K/V tensor 本身。
- 每个 sequence 当前有效长度。

有效长度决定 decode attention 该读多少历史 token，也决定新 K/V 应该 append 到哪里。

显存估算时要注意几个常见误区。

第一，batch size 不是唯一因素。一个 batch 里 8 个短请求和 8 个长请求的 KV cache 成本完全不同。

第二，prompt token 和 generated token 都占 KV cache。长 prompt 本身就会在 prefill 后占用大量 cache，不是只有生成出来的 token 才占 cache。

第三，GQA/MQA 会影响 KV cache 大小。如果 `num_kv_heads` 小于 `num_heads`，KV cache 会比普通多头 attention 更省。

第四，dtype 会直接影响容量。FP16/BF16 通常是 2 bytes，FP32 是 4 bytes，KV cache 量化则可能进一步降低 bytes。

## 5. 原理详解：KV cache 生命周期

KV cache 的生命周期包括：

1. prefill 创建 cache。
2. decode 每步 append 新 K/V。
3. attention 每步读取历史 K/V。
4. 请求结束时 release。

如果 release 失败，服务运行一段时间后显存会被已结束请求占满。在线 serving 系统必须严肃处理 cache 生命周期。

连续 KV cache 的一个朴素实现是每次 append 时做 concat：

```text
old: [num_heads, old_len, head_dim]
new: [num_heads, 1, head_dim]
cat -> [num_heads, old_len + 1, head_dim]
```

这种实现容易理解，但频繁 concat 会带来内存分配和拷贝开销。真实系统通常会预分配、分块或使用 paged KV 来避免每步搬移历史数据。

生命周期里最重要的是“所有权”。一个 sequence 的 cache 只能在它还需要继续生成时保留；一旦请求完成，系统必须知道哪些 cache 属于它，并释放这些资源。如果多个请求共享 prefix cache，那么释放还要看引用计数，不能简单地请求结束就删除共享块。

连续 KV cache 容易理解，但它把 sequence 的历史放在连续空间里。这个假设在教学上清楚，在高并发服务中却会带来扩容和碎片问题。后续 Paged KV 会把这个连续空间拆成固定大小 block，让逻辑连续和物理连续解耦。

## 6. 本节小结

连续 KV cache 的主要问题不是数学公式，而是生命周期。一个请求结束后必须释放 cache；一个请求继续 decode 时必须接着原来的位置追加；batch 内不同请求长度不同，不能把它们的历史长度混在一起。

复习本节时要抓住四个动作：

- create: prefill 后为 prompt 建立历史 K/V。
- append: 每个 decode step 把新 token 的 K/V 加到末尾。
- read: attention 根据有效长度读取历史 K/V。
- release: 请求完成后释放不再需要的 cache。

学完本节后，应能解释 KV cache 为什么能减少重复计算、为什么会消耗大量显存、为什么每个 sequence 必须维护自己的有效长度，以及为什么请求结束后必须释放 cache。
