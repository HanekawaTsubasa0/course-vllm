# Week 10: Paged KV 与 Block Manager

## 1. 本周核心问题

KV cache 让 decode 不必反复计算历史 token，但它也带来新的问题：历史越长，占用显存越多；并发请求越多，显存分配越复杂。如果每个请求都要求一段连续显存，系统很快会遇到浪费、碎片和扩容困难。

本周要回答四个问题：

- 为什么连续 KV cache 在高并发 serving 中不够好？
- Paged KV 为什么要借鉴操作系统里的分页思想？
- block table 和 slot mapping 如何把“逻辑 token 位置”翻译成“物理显存位置”？
- prefix cache 为什么只能安全复用完全相同的前缀？

## 2. 背景知识：连续 KV cache 的问题

先回忆 KV cache 的基本用途：每生成一个新 token，模型会为这个 token 在每一层 attention 中计算 K 和 V，并把它们保存下来。后续 token 做 attention 时，需要读取前面所有 token 的 K/V。也就是说，KV cache 的长度会随着生成过程不断增长。

如果每个请求都需要一段连续 KV cache，系统会遇到几个问题。

第一，预分配浪费。请求最终生成多长不确定。如果按最大长度预分配，大量空间不用。

第二，动态增长困难。decode 每步增长一个 token，如果底层 tensor 需要不断 concat 或搬移，开销很大。

第三，碎片。请求长度不同、结束时间不同，显存里会出现很多大小不一的空洞。

第四，并发请求之间很难复用。很多请求可能共享同一个 system prompt 或文档前缀，但如果 KV cache 只是一整段连续空间，就不容易把相同前缀拆出来独立复用。

Paged KV 的思想就是借鉴虚拟内存分页：逻辑上连续，物理上分块。操作系统里，进程看到的虚拟地址空间可以是连续的，但真实物理内存可以分散在不同页框中。Paged KV 也是类似思想：一个 sequence 看到的 token position 是连续的，但真实 K/V 可以放在不同物理 block 里。

这个类比很重要，但不能混淆：

- 操作系统分页管理的是通用内存地址。
- Paged KV 管理的是 attention 要读写的 K/V 张量位置。
- 操作系统通过页表做虚拟地址到物理地址的翻译。
- Paged KV 通过 block table 做 logical block 到 physical block 的翻译。

## 3. 原理详解：Block table

把 KV cache 切成固定大小 block。例如 block size 是 16，逻辑 token 0-15 在 logical block 0，16-31 在 logical block 1。

block table 保存：

```text
logical_block_id -> physical_block_id
```

访问某个 token position 时：

```text
logical_block = position // block_size
offset = position % block_size
physical_block = block_table[logical_block]
slot = physical_block * block_size + offset
```

attention kernel 根据 slot 找到 K/V。

block table 的意义是把“逻辑连续”和“物理连续”解耦。对模型来说，一个 sequence 的历史 token 仍然是 0、1、2、3 这样连续增长；对显存管理来说，这些 token 可以散落在不同物理 block 中。只要 block table 能把逻辑位置翻译成物理位置，attention 就能读到正确 K/V。

举一个具体例子。假设 block size 是 4，一个请求已经有 10 个 token，那么逻辑上它占用：

```text
logical block 0: token 0, 1, 2, 3
logical block 1: token 4, 5, 6, 7
logical block 2: token 8, 9
```

如果物理 block 分配结果是：

```text
logical block 0 -> physical block 12
logical block 1 -> physical block 7
logical block 2 -> physical block 30
```

那么 token 6 的位置计算是：

```text
position = 6
logical_block = 6 // 4 = 1
offset = 6 % 4 = 2
physical_block = block_table[1] = 7
slot = 7 * 4 + 2 = 30
```

这里的 slot 可以理解为“在所有物理 KV block 展平以后，第几个 token 槽位”。真正的 K/V 张量通常还会有 layer、head、head_dim 等维度，所以完整访问还要加上这些维度：

```text
K[layer, physical_block, offset, kv_head, head_dim]
V[layer, physical_block, offset, kv_head, head_dim]
```

初学时最容易错的是把 `position`、`logical_block`、`physical_block`、`offset` 混成一个概念。它们分别回答不同问题：

- `position`: 这是 sequence 里的第几个 token。
- `logical_block`: 这个 token 属于这个 sequence 的第几个逻辑块。
- `physical_block`: 这个逻辑块实际被放到显存里的哪个物理块。
- `offset`: 这个 token 在块内部的偏移。

## 4. 原理详解：Paged KV 的收益和代价

Paged KV 带来几个好处：

- 按需分配 block，减少预分配浪费。
- 请求结束后 block 可以回收。
- 每个 sequence 的 KV 在逻辑上连续，物理上可以分散在多个 block 中。
- prefix cache 可以复用相同 prompt 的完整 block。

代价是：

- 需要 block table metadata。
- attention kernel 访问 K/V 时多一次间接寻址。
- block size 选择会影响碎片和调度开销。

block size 是一个重要 tradeoff。block 太大，最后一个 block 可能浪费很多未使用 slot，内部碎片更严重；block 太小，block table 更长，metadata 和寻址开销更大，kernel 访问也可能更复杂。

例如 block size 为 16 时，一个 17 token 的 sequence 需要 2 个 block，其中第二个 block 只用了 1 个 token，浪费 15 个 slot。block size 为 4 时，同样 17 token 需要 5 个 block，只浪费 3 个 slot，但 block table 更长。

因此 block size 不是越小越好，也不是越大越好。它要在碎片、metadata、kernel 访存模式和调度复杂度之间折中。

还要区分两类碎片：

- 内部碎片：已经分配的 block 里有一部分 slot 没有使用，例如最后一个 block 没填满。
- 外部碎片：系统里有很多零散空洞，虽然总空闲空间不少，却难以满足连续大块分配。

Paged KV 主要缓解外部碎片，因为它不要求一个 sequence 的所有 KV 连在一起；但它会引入内部碎片，因为每个 sequence 的最后一个 block 可能没有填满。

Paged KV 还要求系统维护 free block list。可以把它想成一个“空闲物理 block 池”：

```text
新 token 需要写入，但当前最后一个 block 满了
-> 从 free block list 取一个 physical block
-> 写入 block table
-> 后续 attention 通过 block table 找到它

请求结束
-> 找到它独占的 physical block
-> 清理引用关系
-> 放回 free block list
```

这个流程看起来像内存管理，而不是矩阵计算。它说明 LLM serving 不只是模型数学问题，也包含显存资源管理问题。

## 5. 原理详解：Prefix cache

很多 serving workload 有共享前缀。例如同一个 system prompt、同一个文档前缀、同一个工具描述。prefix cache 通过 hash 完整 block 的 token，把相同前缀的 KV block 复用。

prefix cache 必须处理引用计数。多个 sequence 共享一个 block 时，不能因为其中一个请求结束就释放物理 block。

prefix cache 只在“前缀完全相同”时可靠。原因是 KV cache 不是普通文本缓存，它保存的是模型每一层对历史 token 的中间状态。如果 token 序列有一个位置不同，后续 attention 状态就不能直接复用。

为什么强调“前缀”？因为因果语言模型中，第 t 个 token 的 hidden state 只能看见 0 到 t 的历史。如果两个请求前 128 个 token 完全相同，那么这 128 个位置在每一层产生的 K/V 也是相同的，可以被复用。可是如果第 10 个 token 不同，那么从第 10 个位置开始，后续 hidden state 都可能不同，后面的 KV 就不能直接共享。

prefix cache 通常按 block 粒度复用，而不是按任意 token 粒度复用。原因是 block 粒度更容易管理：

- block table 本来就按 block 记录映射。
- 引用计数按 block 维护比较简单。
- 完整 block 的 hash 更容易作为 cache key。
- 释放时可以直接按 block 回收。

一个常见过程如下：

```text
请求 A: tokens [a, b, c, d, e, f, g, h]
block size = 4

block 0 的 token 是 [a, b, c, d]
block 1 的 token 是 [e, f, g, h]

请求 B: tokens [a, b, c, d, x, y]

请求 B 的 block 0 和请求 A 完全相同，可以复用
请求 B 后面的 token 不同，需要新算新分配
```

prefix cache 的收益也不是免费的。系统需要保存 hash 表、引用计数和淘汰策略。如果 cache 永远不淘汰，显存会被旧 prefix 占满；如果淘汰太激进，又会降低命中率。因此 prefix cache 也是一个调度和资源管理问题。

## 6. 本节小结

理解 paged KV 时，先把基本分配和 slot mapping 想清楚，再理解 prefix cache 的复用与引用计数。

paged KV 最容易错的是逻辑位置和物理位置混淆。逻辑 token position 是 sequence 里的第几个 token；物理位置是某个 block 的某个 offset。attention kernel 读取 K/V 时必须用 block table 完成这次映射。

另一个容易错的地方是释放。请求结束后可以释放它独占的 block，但共享 prefix block 只有引用计数归零时才能释放。学完本节后，应能解释 logical token position、logical block、physical block、offset、slot mapping 和 prefix cache 的关系。
