# Week 10: Paged KV 与 Block Manager

## 0. 本节学习目标

Week10 聚焦 KV cache 的内存管理。Week08 的连续 KV cache 容易理解，但不适合高并发长上下文 serving。Week10 讲为什么需要 paged KV、block table、slot mapping 和 prefix cache。

## 1. 连续 KV cache 的问题

如果每个请求都需要一段连续 KV cache，系统会遇到几个问题。

第一，预分配浪费。请求最终生成多长不确定。如果按最大长度预分配，大量空间不用。

第二，动态增长困难。decode 每步增长一个 token，如果底层 tensor 需要不断 concat 或搬移，开销很大。

第三，碎片。请求长度不同、结束时间不同，显存里会出现很多大小不一的空洞。

Paged KV 的思想就是借鉴虚拟内存分页：逻辑上连续，物理上分块。

## 2. Block table

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

## 3. Paged KV 的收益

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

## 4. Prefix cache

很多 serving workload 有共享前缀。例如同一个 system prompt、同一个文档前缀、同一个工具描述。prefix cache 通过 hash 完整 block 的 token，把相同前缀的 KV block 复用。

prefix cache 必须处理引用计数。多个 sequence 共享一个 block 时，不能因为其中一个请求结束就释放物理 block。

prefix cache 只在“前缀完全相同”时可靠。原因是 KV cache 不是普通文本缓存，它保存的是模型每一层对历史 token 的中间状态。如果 token 序列有一个位置不同，后续 attention 状态就不能直接复用。

## 5. 实现时最容易错的地方

先实现基本分配和 slot mapping，再理解 prefix cache 的复用与引用计数。

paged KV 最容易错的是逻辑位置和物理位置混淆。逻辑 token position 是 sequence 里的第几个 token；物理位置是某个 block 的某个 offset。attention kernel 读取 K/V 时必须用 block table 完成这次映射。

另一个容易错的地方是释放。请求结束后可以释放它独占的 block，但共享 prefix block 只有引用计数归零时才能释放。

## 6. 实验中的少量对照

实验会把连续 KV cache 替换成 paged KV cache。阅读代码时重点看三件事：block 如何分配，slot mapping 如何从逻辑位置映射到物理位置，prefix cache 如何避免错误释放共享 block。
