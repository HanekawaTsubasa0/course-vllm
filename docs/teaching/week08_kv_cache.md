# Week 08: KV Cache

## 0. 本节学习目标

Week08 聚焦 KV cache 的基本原理和生命周期。本周先讲连续 KV cache，也就是每个 sequence 的 K/V 按 token 维连续 append。Paged KV 的 block 化存储放到 Week10。

## 1. 为什么没有 KV cache 会很慢

自回归生成第 t 步需要看前面所有 token。如果没有 KV cache，每一步都要把完整上下文重新送进模型，重新计算所有历史 token 的 K/V。

假设 prompt 长度是 `L`，要生成 `T` 个 token。没有 cache 时，第 1 步处理 `L` 个 token，第 2 步处理 `L+1` 个 token，第 T 步处理 `L+T-1` 个 token。大量历史计算被重复。

有 KV cache 后：

```text
prefill: 计算 prompt 的 K/V 并缓存
decode step: 只计算新 token 的 K/V，读取历史 K/V
```

这把 decode 的重复计算大幅减少。

## 2. KV cache 存什么

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

## 3. 显存成本

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

## 4. 生命周期

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

## 5. 实现时最容易错的地方

需要处理：

- 首次 append。
- 后续沿 sequence length 维 concat。
- 按 sequence id release。

block table 会在 Week10 进入视野，本节先把连续 KV cache 的 append、fetch、release 理清楚。

连续 KV cache 的主要问题不是数学公式，而是生命周期。一个请求结束后必须释放 cache；一个请求继续 decode 时必须接着原来的位置追加；batch 内不同请求长度不同，不能把它们的历史长度混在一起。

## 6. 实验中的少量对照

实验会先使用连续 KV cache，让模型能连续生成多个 token。阅读代码时重点看 append、fetch、release 三个动作是否和 sequence 生命周期一致。
