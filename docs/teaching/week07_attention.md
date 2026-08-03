# Week 07：Attention 从公式、Shape 到 IO-Aware 实现

## Attention 到底在做什么

一句话概括：每个 query 位置根据与所有可见 key 的匹配程度，从对应 value 中汇总信息。

例如模型处理“巴黎是法国的首都”。当当前位置需要更新“首都”的表示时，它可能更关注“巴黎”和“法国”。Attention 不会直接操作文字，而是通过向量点积学习这种关联。

这句话听起来简单，真正实现时至少要处理：

```text
Q/K/V 从哪里来
head 怎样拆分
为什么除以 sqrt(d)
未来位置怎样 mask
softmax 怎样稳定计算
prefill 与 decode 的 shape 为什么不同
GQA 的 Q head 怎样找到 KV head
为什么不能总把完整 score matrix 写到显存
```

本周先把 dense attention 数学与 reference 做牢，再理解 FlashAttention 的 IO 思路。Paged KV 的分配生命周期放到 Week 10；本周只把它视为历史 K/V 的一种存储方式。

---

## 一、从 Hidden State 得到 Q、K、V

输入 hidden states：

```text
X [batch, seq_len, hidden_size]
```

三组 Linear：

```text
Q = X Wq
K = X Wk
V = X Wv
```

可以用检索系统建立直觉：

- **Query**：当前位置想找什么；
- **Key**：每个位置拿什么特征供匹配；
- **Value**：匹配后真正取回的内容。

Q/K/V 都来自同一个 X，因此叫 self-attention。Cross-attention 中 Q 与 K/V 可以来自不同序列。

### 为什么不直接用 X 做点积

可学习投影允许模型把“用来匹配的特征”和“被取回的内容”分开。某个维度可能适合判断语法关系，却不一定适合作为最终汇总内容。

## 二、多头 Attention

设：

```text
hidden_size = num_heads * head_dim
```

投影后 reshape：

```text
Q [B, Hq, S, D]
K [B, Hkv, S, D]
V [B, Hkv, S, D]
```

每个 head 独立计算 attention，再把结果拼接并经过输出 projection。

不同 head 可以学习不同关系：局部邻近、长距离指代、语法、格式等。这里的解释只是直觉；head 并没有被人工指定固定语义。

```mermaid
flowchart LR
    X["X [B,S,hidden]"] --> Q["Wq -> Q heads"]
    X --> K["Wk -> K heads"]
    X --> V["Wv -> V heads"]
    Q --> A["per-head attention"]
    K --> A
    V --> A
    A --> C["concat heads"]
    C --> O["Wo -> output"]
```

## 三、Scaled Dot-Product Attention

单个 head：

```text
scores = Q K^T / sqrt(D)
weights = softmax(scores + mask)
output = weights V
```

### 点积表示什么

Query 和 Key 越同向，点积越大；越不相关或方向相反，点积越小。Softmax 再把一行 scores 变成对所有 key 位置的权重。

### 为什么除以 sqrt(D)

假设 Q/K 每个分量均值约 0、方差约 1，并近似独立。D 个乘积相加后，点积方差会随 D 增长，标准差约为 `sqrt(D)`。

不缩放时，D 较大可能让 scores 绝对值很大，softmax 变得极尖，梯度和低精度数值都更不稳定。除以 `sqrt(D)` 把典型尺度拉回相对稳定范围。

这不是为了让所有 score 落入 [-1,1]，而是控制随 head_dim 增长的统计尺度。

## 四、用三 Token 手算一行 Attention

设某 query 对三个 key 的缩放后分数：

```text
scores = [2, 1, 0]
```

Softmax：

```text
weights ≈ [0.665, 0.245, 0.090]
```

如果 value 为一维：

```text
V = [10, 4, -2]
```

输出：

```text
0.665*10 + 0.245*4 + 0.090*(-2)
≈ 7.45
```

Attention 输出不是“复制最大权重 token”，而是对 value 的加权和。权重由 Q/K 决定，汇总内容来自 V。

## 五、Causal Mask

自回归模型不能让位置 i 读取未来位置 j>i：

```text
          Key position
Query     0   1   2   3
0         ✓   ×   ×   ×
1         ✓   ✓   ×   ×
2         ✓   ✓   ✓   ×
3         ✓   ✓   ✓   ✓
```

实现通常在被屏蔽位置加 `-inf`，softmax 后概率为 0：

```text
masked_scores[i,j] = -inf, if j > i
```

### Mask 顺序

Scale 和 mask 都应在 softmax 前完成。若先 softmax 再把未来概率置 0，却不重新归一化，剩余概率和将小于 1。

### Padding Mask 与 Causal Mask

Batch 内不同长度可能还需要 padding mask。Causal mask 处理“未来不可见”，padding/context-length mask 处理“这个位置根本不属于有效序列”。两者语义不同。

## 六、MHA、MQA 与 GQA

### MHA

```text
Hq = Hkv
```

每个 query head 有自己的 K/V head。

### MQA

```text
Hkv = 1
```

所有 query heads 共享一组 K/V，显著减少 KV cache。

### GQA

```text
1 < Hkv < Hq
```

每组 query heads 共享一个 KV head。例如：

```text
Hq=16, Hkv=8
group_size=2
kv_head = q_head // 2
```

如果 q head 7 错映射到 KV head 7，而正确值应为 3，shape 仍可能合法，结果却错误。

### KV Cache 为什么因此变小

KV cache 按 KV heads 保存。其他参数相同时，Hkv 从 16 降到 8，K/V 存储约减半。Q 不跨 decode step 缓存，因此 Q head 数不直接决定 KV cache 大小。

## 七、Prefill Attention 的 Shape 与复杂度

Prefill：

```text
Q [B,Hq,S,D]
K [B,Hkv,S,D]
V [B,Hkv,S,D]
```

每个 query position 对所有可见 key position 打分。Dense score matrix 每 head 是：

```text
[S,S]
```

计算量随 `S^2*D` 增长；显式 score/prob 中间量随 `S^2` 增长。S 从 2k 增到 4k，score 元素数约变为 4 倍。

### 一个显存数量感

若：

```text
B=1, H=16, S=4096, dtype=bf16
```

只计算一个 `[B,H,S,S]` score tensor：

```text
1 * 16 * 4096 * 4096 * 2 bytes
= 512 MiB
```

还没算 softmax probability、Q/K/V、输出和反向图。推理没有训练反向，但显式中间矩阵仍会带来显存和 HBM 流量。

## 八、Decode Attention 的 Shape

Decode 每个请求当前只有一个新 query：

```text
Q [B,Hq,1,D]
K/V cache [B,Hkv,context_len,D]
```

Score 每请求每 head 只有一行：

```text
[1, context_len]
```

它不再构造 SxS，但每一步都要读取完整历史 K/V。随着 context_len 增长，读取量持续增加。

Prefill 和 decode 因此不是同一个 kernel 改个参数那么简单：

| 维度 | Prefill | Decode |
| --- | --- | --- |
| Query positions | 多 | 每请求通常 1 |
| Score 形状 | SxS | 1xcontext |
| 并行来源 | token/head/batch | batch/head/history |
| 主要数据问题 | 中间矩阵与 IO | 读取变长 KV |
| 典型优化 | tiled/FlashAttention | batching、KV layout、paged decode |

## 九、朴素 Attention 的 IO 问题

直接实现可能分三步：

```text
1. S = QK^T，写到 HBM
2. P = softmax(S)，再写到 HBM
3. O = PV，从 HBM 读 P/V
```

S 和 P 都是 O(seq^2) 中间量。即使 GPU 计算很快，反复写回和读取 HBM 也很昂贵。

FlashAttention 的关键不是近似，也不是改变公式，而是重排计算，让 score/prob tile 尽量留在片上存储，并最终只写输出和必要统计量。

## 十、为什么 Softmax 不能简单分块后各算各的

假设一行 scores 被拆成两个 block：

```text
block A = [2,1]
block B = [4,0]
```

分别 softmax：

```text
softmax(A) ≈ [0.731,0.269]
softmax(B) ≈ [0.982,0.018]
```

直接拼接后总和为 2，不是全行 softmax。因为两个块共享同一个全局最大值和归一化分母。

要分块，必须在看到新块时修正旧结果。

## 十一、Online Softmax 的核心推导

处理到某一块后，维护：

```text
m  已见 scores 的最大值
l  sum(exp(score - m))
o  以当前尺度累积的加权 value
```

新块有最大值 `m_b`，新的全局最大值：

```text
m_new = max(m, m_b)
```

旧分母原来基于 `m`，要转换到 `m_new` 尺度：

```text
l_old_rescaled = l * exp(m - m_new)
```

新块分母：

```text
l_block = sum(exp(score_b - m_new))
```

更新：

```text
l_new = l_old_rescaled + l_block
```

旧输出累加器同样要乘 `exp(m-m_new)`，再加入新块的 `exp(score_b-m_new)V_b`。

最后：

```text
output = o / l
```

```mermaid
flowchart LR
    A["已有 m,l,o"] --> M["读取新 K/V tile，算 score tile"]
    M --> N["m_new=max(m,m_tile)"]
    N --> R["重缩放旧 l/o"]
    R --> U["加入新 tile 的 exp 与 value"]
    U --> C{"还有 tile?"}
    C -- "是" --> M
    C -- "否" --> O["output=o/l"]
```

这就是分块时仍保持精确 softmax 的关键。它没有把每块独立归一化，而是不断维护全局一致的尺度。

## 十二、FlashAttention 为什么减少 HBM Traffic

FlashAttention 风格计算：

1. 把 Q tile 载入片上存储；
2. 逐块读取 K/V；
3. 计算小块 scores；
4. 在线更新 softmax 统计量和输出；
5. 不把完整 score/prob matrix 写回 HBM；
6. 写出最终 O。

```mermaid
flowchart TB
    H["HBM: Q,K,V"] --> Q["片上 Q tile"]
    H --> KV["片上 K/V tile"]
    Q --> S["score tile"]
    KV --> S
    S --> OS["online softmax + output accumulation"]
    OS --> H2["HBM: final O"]
```

优化的核心是 IO-aware：根据存储层次容量安排 tile 和计算顺序。作者的 [FlashAttention V1](https://zhuanlan.zhihu.com/p/669926191) 与 [V2](https://zhuanlan.zhihu.com/p/691067658) 分别适合建立 IO 动机和并行组织直觉。

## 十三、Decode Attention 与 Paged KV

Attention 数学只要求按逻辑顺序得到历史 K/V。它们可以连续存储，也可以分散在固定大小物理 block 中。

Paged decode 额外输入：

```text
block_tables [B,max_blocks]
context_lens [B]
block_size
physical K/V cache
```

逻辑位置 `t` 映射：

```text
logical_block = t // block_size
offset        = t % block_size
physical_block = block_table[seq, logical_block]
slot = physical_block * block_size + offset
```

Week 07 只要求 attention kernel 按有效 context 读取正确 K/V；block 如何分配、共享和释放留到 Week 10。

## 十四、Reference 实现为什么必须保留

高性能 attention 涉及 mask、head 映射、online softmax、变长 context 和地址计算。Reference 应优先清晰：

```text
把逻辑 K/V 读成 dense
-> repeat/map GQA heads
-> scores / scale / mask
-> torch.softmax
-> weighted V
```

它可能慢，却是 correctness oracle。优化 kernel 不应同时修改 reference 和 expected，否则容易让相同错误互相“对齐”。

## 十五、最常见的四类错误

### Scale 错

忘记 `/sqrt(D)` 或用 hidden_size 代替 head_dim。

### Mask 错

未来 token 未屏蔽，或 padding/context 外位置被读取。

### GQA 映射错

Q head 到 KV head 的分组关系错误。

### Length/Address 错

Decode 读取超过 context_len，或 paged slot 计算错误。

这些错误都可能不产生崩溃，只产生“看起来还能生成”的错误文本，因此必须做张量级对齐。

## 十六、正确性测试

### Dense Prefill

- 小 S 手工 mask；
- 与 PyTorch attention 对齐；
- 检查未来位置概率为 0；
- 覆盖多 batch/head 与不同 D。

### Dense Decode

- Q 长度 1；
- 不同 context_len；
- 与 prefill 最后位置或 dense reference 对齐。

### GQA

- Hq 与 Hkv 不等；
- 构造每个 KV head 明显不同的数据，防止错误映射被随机值掩盖。

### Paged Decode

- block table 打乱物理 block 顺序；
- context 跨 block 边界；
- 最后 block 仅部分有效；
- 与 gather 成 dense 后的 reference 对齐。

### Online Softmax

- 一块与多块结果一致；
- 后续 tile 出现更大 max 时旧累加器正确重缩放；
- 极端 logits 不出现 NaN。

## 十七、本周实验

### 实验 1：三 Token 手算

手算 scores、mask、softmax 和 weighted V，作为所有复杂实现的最小 oracle。

### 实验 2：Prefill vs Decode Shape

给出同一请求在 prefill 和第 3 个 decode step 的 Q/K/V shape 与 score 元素数。

### 实验 3：GQA 映射

画出 Hq=16、Hkv=4 时每个 Q head 对应的 KV head。

### 实验 4：Score Matrix 显存

计算不同 S、H、dtype 下显式 score tensor 大小，解释 S 翻倍为何变 4 倍。

### 实验 5：Online Softmax

把一行 scores 拆成多个 tile，逐步记录 m/l/o，并与一次性 softmax 对齐。

### 实验 6：CUDA/Reference

报告 dense/paged attention 最大误差，并用严格模式证明 CUDA dispatch。

## 十八、教学 Kernel 与生产 Kernel 的差距

至少包括：

```text
tile 与 warp 专项优化
Tensor Core / MMA 使用
异步 copy 与流水
多 shape/dtype dispatch
更复杂的变长序列与分页布局
架构专项调优
融合 RoPE、cache write 等操作
```

课程 kernel 的目标是让数学、地址和 online softmax 可验证，不以超过 FlashAttention、cuDNN 或 vLLM 生产 kernel 为验收标准。

## 十九、常见误区

### Attention 就是 QK 点积

还包括 scale、mask、softmax 和 V 汇总，任何一步错误都会改变语义。

### FlashAttention 是近似算法

核心版本通过重排和 online softmax 计算精确 attention，主要减少 IO 和中间存储。

### Decode 没有 SxS，所以几乎免费

它每个 token 都要执行，并读取增长的历史 K/V。高并发和长 context 下仍是核心成本。

### GQA 只是 repeat K/V

数学上可用 repeat 理解，生产实现通常避免真的复制，以节省带宽和显存。

### PagedAttention 改变了 Attention 公式

分页改变 K/V 地址组织，不改变 QK-softmax-V 数学。

## 二十、学完本周，应能回答

1. Q/K/V 各自承担什么角色？
2. 为什么 score 除以 sqrt(head_dim)？
3. Causal mask 与 padding/context mask 有何区别？
4. MHA、MQA、GQA 如何影响 KV cache？
5. Prefill 与 decode attention 的 shape 和瓶颈有何差异？
6. 为什么不能对每个 score tile 独立 softmax？
7. Online softmax 中 m、l、o 分别表示什么？
8. FlashAttention 减少了哪些 HBM 中间读写？
9. Paged KV 为什么不改变 attention 数学？

## 参考与素材说明

- 猛猿：[Self-Attention](https://zhuanlan.zhihu.com/p/455399791)
- 猛猿：[FlashAttention V1：从硬件到计算逻辑](https://zhuanlan.zhihu.com/p/669926191)
- 猛猿：[FlashAttention V2：从原理到并行计算](https://zhuanlan.zhihu.com/p/691067658)
- 猛猿：[再读 MLA](https://zhuanlan.zhihu.com/p/19585986234)
- 课程工程：dense/paged attention reference、CUDA kernels 与 Week 07 grader

正文、推导、数值例子和图示均为课程原创组织。参考文章用于补充直觉和技术背景；具体实现以课程指定模型、dtype 和 shape 为准。
