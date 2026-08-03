# 猛猿文章与 LLM Serving 课程素材映射

更新时间：2026-07-16

## 使用边界

本文档只保存文章标题、链接、结构统计和课程映射，不复制文章正文或图片。课程讲义应基于技术事实重新组织和表达，图示优先原创重绘；如后续取得作者明确授权，再按授权范围使用原文或原图，并在使用位置标明作者和原始链接。

## 素材盘点

- 作者：猛猿（知乎账号 `lemonround`）
- 主页公开文章：70 篇，已全部枚举
- 直接服务于本课程的 serving 主线文章：16 篇
- serving 主线审计总量：289,374 个可见文本字符、120 张正文图，120 张均可加载
- 课程补充素材：CUDA/GEMM、Attention、RoPE、LayerNorm、TP/PP/CP、ZeRO、MoE、RLHF 等

## 课程映射

| 周次 | 课程主题 | 首要参考 | 补充参考 | 使用建议 |
| --- | --- | --- | --- | --- |
| Week 01 | Serving 流程与指标 | [vLLM V1：整体流程](https://zhuanlan.zhihu.com/p/1900126076279160869) | [vLLM 源码解析1：整体架构](https://zhuanlan.zhihu.com/p/691045737)、[DistServe](https://zhuanlan.zhihu.com/p/706761664) | 用 offline/online 两条调用链建立全局视角，再引出 TTFT、TPOT、吞吐和延迟。 |
| Week 02 | Profiling 与性能分析 | [FlashAttention V1](https://zhuanlan.zhihu.com/p/669926191) | [CUDA GEMM 优化](https://zhuanlan.zhihu.com/p/703256080)、[KV Cache 初始化](https://zhuanlan.zhihu.com/p/1900932850829730567) | 从 memory-bound、compute-bound、IO 次数和显存预算建立性能分析方法。 |
| Week 03 | CUDA extension 入门 | [CUDA GEMM 优化](https://zhuanlan.zhihu.com/p/703256080) | [FlashAttention V1](https://zhuanlan.zhihu.com/p/669926191) | 只借鉴 grid/block、访存和验证思路，课程中重新设计更小的 vector-add 闭环。 |
| Week 04 | RMSNorm 与 RoPE | [RoPE 直觉解释](https://zhuanlan.zhihu.com/p/863378538) | [LayerNorm/BatchNorm](https://zhuanlan.zhihu.com/p/456863215) | 分开讲归一化与位置编码的数学目标，再落到张量形状和 kernel 映射。 |
| Week 05 | Matmul 与 Linear | [CUDA GEMM 优化](https://zhuanlan.zhihu.com/p/703256080) | [FlashAttention V2](https://zhuanlan.zhihu.com/p/691067658) | 形成 naive -> tiled -> shared memory -> 性能瓶颈的递进线。 |
| Week 06 | Softmax 与 Sampling | [Self-Attention](https://zhuanlan.zhihu.com/p/455399791) | [FlashAttention V1](https://zhuanlan.zhihu.com/p/669926191) | 作者没有专门的 sampling 教程；softmax 归约可参考 Attention 文章，采样部分需原创补足。 |
| Week 07 | Attention | [Self-Attention](https://zhuanlan.zhihu.com/p/455399791) | [FlashAttention V1](https://zhuanlan.zhihu.com/p/669926191)、[FlashAttention V2](https://zhuanlan.zhihu.com/p/691067658)、[MLA](https://zhuanlan.zhihu.com/p/19585986234) | 先标准 Attention，再从 IO 复杂度解释 FlashAttention，最后把 MLA 作为扩展。 |
| Week 08 | KV Cache | [vLLM V1：KV Cache 初始化](https://zhuanlan.zhihu.com/p/1900932850829730567) | [PagedAttention 原理](https://zhuanlan.zhihu.com/p/691038809) | 先解释为什么 decode 必须缓存，再讲容量估算和初始化。 |
| Week 09 | Engine 与请求生命周期 | [vLLM V1：整体流程](https://zhuanlan.zhihu.com/p/1900126076279160869) | [vLLM 源码解析1](https://zhuanlan.zhihu.com/p/691045737)、[AsyncLLM](https://zhuanlan.zhihu.com/p/1916187125931554299) | 围绕 add request、step、finish/abort 和同步/异步两条路径组织。 |
| Week 10 | Paged KV 与 Block Manager | [PagedAttention 原理](https://zhuanlan.zhihu.com/p/691038809) | [BlockManager](https://zhuanlan.zhihu.com/p/700780161)、[Prefix Caching](https://zhuanlan.zhihu.com/p/707228704) | 用操作系统分页类比逻辑块/物理块，再讲 slot mapping、分配、释放和共享。 |
| Week 11 | Continuous Batching | [vLLM V1：Scheduler](https://zhuanlan.zhihu.com/p/1908153627639551302) | [旧版 Scheduler 深入解析](https://zhuanlan.zhihu.com/p/692540949)、[chunked-prefills](https://zhuanlan.zhihu.com/p/710165390) | 用 waiting/running、单步 schedule、token budget 和抢占构造状态机。 |
| Week 12 | 系统优化与 Admission | [chunked-prefills](https://zhuanlan.zhihu.com/p/710165390) | [DistServe](https://zhuanlan.zhihu.com/p/706761664)、[TP 通信计算 overlap](https://zhuanlan.zhihu.com/p/16594218518) | 区分调度优化、传输/通信重叠、prefill/decode 干扰和准入保护。 |
| Week 13 | 容量规划与并行 | [DistServe](https://zhuanlan.zhihu.com/p/706761664) | [TP](https://zhuanlan.zhihu.com/p/622212228)、[PP](https://zhuanlan.zhihu.com/p/613196255)、[Context Parallel](https://zhuanlan.zhihu.com/p/5502876106)、[MoE](https://zhuanlan.zhihu.com/p/681154742) | 从显存和 SLO 约束出发选择 TP/PP/CP/EP，不把训练并行结论直接照搬到推理。 |
| Week 14 | AscendC 对照 | 暂无直接对应文章 | [CUDA GEMM 优化](https://zhuanlan.zhihu.com/p/703256080) | 只抽象 host/device、并行粒度、存储层次和 profiling 的对照框架。 |
| Week 15 | 前沿 Serving | [Prefix Caching](https://zhuanlan.zhihu.com/p/707228704) | [DistServe](https://zhuanlan.zhihu.com/p/706761664)、[chunked-prefills](https://zhuanlan.zhihu.com/p/710165390)、[MLA](https://zhuanlan.zhihu.com/p/19585986234) | 用“问题 -> 机制 -> 工程位置 -> 指标”的统一模板讲前沿系统。 |
| Week 16 | 综合复盘 | [历史技术文章导航](https://zhuanlan.zhihu.com/p/654910335) | vLLM V1 系列 1-7 | 让学生用同一张系统图串联请求、调度、KV、kernel、worker、指标和证据。 |

## Serving 主线结构审计

| 文章 | 字符数 | 正文图 | 主要课程位置 |
| --- | ---: | ---: | --- |
| [vLLM V1：AsyncLLM](https://zhuanlan.zhihu.com/p/1916187125931554299) | 3,572 | 3 | Week 09 |
| [vLLM V1：KVCacheManager 与 PrefixCaching](https://zhuanlan.zhihu.com/p/1916181593229334390) | 11,204 | 8 | Week 10/15 |
| [vLLM V1：Scheduler](https://zhuanlan.zhihu.com/p/1908153627639551302) | 15,108 | 5 | Week 11 |
| [vLLM V1：加载模型权重](https://zhuanlan.zhihu.com/p/1908151478557839879) | 13,254 | 2 | Week 01/09 |
| [vLLM V1：KV Cache 初始化](https://zhuanlan.zhihu.com/p/1900932850829730567) | 7,342 | 2 | Week 08/13 |
| [vLLM V1：Executor-Workers](https://zhuanlan.zhihu.com/p/1900613601577899465) | 7,984 | 7 | Week 09/13 |
| [vLLM V1：整体流程](https://zhuanlan.zhihu.com/p/1900126076279160869) | 6,058 | 3 | Week 01/09 |
| [MLA](https://zhuanlan.zhihu.com/p/19585986234) | 8,482 | 10 | Week 07/15 |
| [chunked-prefills](https://zhuanlan.zhihu.com/p/710165390) | 30,234 | 14 | Week 11/12/15 |
| [Prefix Caching](https://zhuanlan.zhihu.com/p/707228704) | 42,253 | 7 | Week 10/15 |
| [DistServe](https://zhuanlan.zhihu.com/p/706761664) | 12,426 | 13 | Week 01/12/13/15 |
| [BlockManager](https://zhuanlan.zhihu.com/p/700780161) | 30,762 | 3 | Week 10 |
| [旧版 Scheduler](https://zhuanlan.zhihu.com/p/692540949) | 66,540 | 12 | Week 11 |
| [Mixtral 推理优化](https://zhuanlan.zhihu.com/p/691066049) | 10,364 | 12 | Week 07/08/15 |
| [vLLM 旧版整体架构](https://zhuanlan.zhihu.com/p/691045737) | 11,612 | 7 | Week 01/09 |
| [PagedAttention](https://zhuanlan.zhihu.com/p/691038809) | 12,179 | 12 | Week 08/10 |

## 完整 70 篇文章索引

以下按作者主页的时间顺序排列。

1. [谢谢大家，暂时停更一段日子](https://zhuanlan.zhihu.com/p/1938597050020245547)
2. [万字长文图解Qwen2.5-VL实现细节](https://zhuanlan.zhihu.com/p/1921289925552210138)
3. [请不要在xhs平台转载我的文章](https://zhuanlan.zhihu.com/p/1919431109067912610)
4. [异步RL框架AReaL速览](https://zhuanlan.zhihu.com/p/1916441720817714438)
5. [图解Vllm V1系列7：使用AsyncLLM做异步推理](https://zhuanlan.zhihu.com/p/1916187125931554299)
6. [图解Vllm V1系列6：KVCacheManager与PrefixCaching](https://zhuanlan.zhihu.com/p/1916181593229334390)
7. [图解Vllm V1系列5：调度器策略（Scheduler）](https://zhuanlan.zhihu.com/p/1908153627639551302)
8. [图解Vllm V1系列4：加载模型权重(load_model)](https://zhuanlan.zhihu.com/p/1908151478557839879)
9. [图解Vllm V1系列3：KV Cache初始化](https://zhuanlan.zhihu.com/p/1900932850829730567)
10. [图解Vllm V1系列2：Executor-Workers架构](https://zhuanlan.zhihu.com/p/1900613601577899465)
11. [图解Vllm V1系列1：整体流程](https://zhuanlan.zhihu.com/p/1900126076279160869)
12. [探索一个关于deepspeed zero3的认知误区](https://zhuanlan.zhihu.com/p/20115278338)
13. [记录对DeepSeek-R1的一些理解](https://zhuanlan.zhihu.com/p/19843230707)
14. [再读MLA，还有多少细节是你不知道的](https://zhuanlan.zhihu.com/p/19585986234)
15. [图解Megatron TP中的计算通信overlap](https://zhuanlan.zhihu.com/p/16594218518)
16. [图解OpenRLHF中基于Ray的分布式训练流程](https://zhuanlan.zhihu.com/p/12871616401)
17. [人人都能看懂的RL-PPO理论知识](https://zhuanlan.zhihu.com/p/7461863937)
18. [图解大模型训练系列：序列并行4，Megatron Context Parallel](https://zhuanlan.zhihu.com/p/5502876106)
19. [图解大模型训练系列：序列并行3，Ring Attention](https://zhuanlan.zhihu.com/p/4963530231)
20. [图解大模型训练系列：序列并行2，DeepSpeed Ulysses](https://zhuanlan.zhihu.com/p/4496065391)
21. [图解大模型训练系列：序列并行1，Megatron SP](https://zhuanlan.zhihu.com/p/4083427292)
22. [OpenAI o1技术初探3：如何让模型拥有自我纠错的能力](https://zhuanlan.zhihu.com/p/905620136)
23. [OpenAI o1技术初探2：使用MCTS增强推理能力](https://zhuanlan.zhihu.com/p/864190605)
24. [避开复数推导，我们还可以怎么理解RoPE？](https://zhuanlan.zhihu.com/p/863378538)
25. [OpenAI o1技术初探1：Test-Time Scaling Law](https://zhuanlan.zhihu.com/p/773907223)
26. [人人都能看懂的DPO数学原理](https://zhuanlan.zhihu.com/p/721073733)
27. [分离式推理架构2：chunked-prefills](https://zhuanlan.zhihu.com/p/710165390)
28. [vLLM源码解析3：Prefix Caching](https://zhuanlan.zhihu.com/p/707228704)
29. [分离式推理架构1：从DistServe谈起](https://zhuanlan.zhihu.com/p/706761664)
30. [从啥也不会到CUDA GEMM优化](https://zhuanlan.zhihu.com/p/703256080)
31. [vLLM源码解析3：块管理器BlockManager](https://zhuanlan.zhihu.com/p/700780161)
32. [再读deformable detr](https://zhuanlan.zhihu.com/p/700776674)
33. [vLLM源码解析2：调度器策略](https://zhuanlan.zhihu.com/p/692540949)
34. [Flash Attention V2](https://zhuanlan.zhihu.com/p/691067658)
35. [Mixtral 8x7B推理优化](https://zhuanlan.zhihu.com/p/691066049)
36. [vLLM源码解析1：整体架构](https://zhuanlan.zhihu.com/p/691045737)
37. [vLLM核心技术PagedAttention原理](https://zhuanlan.zhihu.com/p/691038809)
38. [DeepSpeed-Megatron MoE并行训练（源码篇）](https://zhuanlan.zhihu.com/p/681692152)
39. [DeepSpeed-Megatron MoE并行训练（原理篇）](https://zhuanlan.zhihu.com/p/681154742)
40. [PPO原理与源码解读](https://zhuanlan.zhihu.com/p/677607581)
41. [FlashAttention V1](https://zhuanlan.zhihu.com/p/669926191)
42. [再读Swin Transformer](https://zhuanlan.zhihu.com/p/663747861)
43. [再读GAN](https://zhuanlan.zhihu.com/p/663253709)
44. [Megatron源码解读3：分布式混合精度训练](https://zhuanlan.zhihu.com/p/662700424)
45. [多模态经典CLIP](https://zhuanlan.zhihu.com/p/660476765)
46. [再读VIT](https://zhuanlan.zhihu.com/p/657666107)
47. [AdaLoRA](https://zhuanlan.zhihu.com/p/657130029)
48. [DDPM源码解读](https://zhuanlan.zhihu.com/p/655568910)
49. [历史技术文章导航](https://zhuanlan.zhihu.com/p/654910335)
50. [LoRA源码解读与实操](https://zhuanlan.zhihu.com/p/654897296)
51. [DDPM数学原理](https://zhuanlan.zhihu.com/p/650394311)
52. [LoRA原理](https://zhuanlan.zhihu.com/p/646831196)
53. [DDPM模型架构](https://zhuanlan.zhihu.com/p/637815071)
54. [Megatron源码解读2：模型并行](https://zhuanlan.zhihu.com/p/634377071)
55. [Megatron源码解读1：分布式环境初始化](https://zhuanlan.zhihu.com/p/629121480)
56. [张量模型并行TP：Megatron-LM](https://zhuanlan.zhihu.com/p/622212228)
57. [DeepSpeed ZeRO](https://zhuanlan.zhihu.com/p/618865052)
58. [DP、DDP与ZeRO](https://zhuanlan.zhihu.com/p/617133971)
59. [流水线并行PP：GPipe](https://zhuanlan.zhihu.com/p/613196255)
60. [赋予GPT写代码能力的Codex](https://zhuanlan.zhihu.com/p/611313567)
61. [GPT1、GPT2与GPT3](https://zhuanlan.zhihu.com/p/609367098)
62. [强化学习：MDP和有模型学习](https://zhuanlan.zhihu.com/p/607596944)
63. [训练框架InstructGPT](https://zhuanlan.zhihu.com/p/605516116)
64. [BERT模型详解](https://zhuanlan.zhihu.com/p/461267517)
65. [Subword Tokenization](https://zhuanlan.zhihu.com/p/460678461)
66. [图解HDFS文件系统](https://zhuanlan.zhihu.com/p/459921566)
67. [ResNet](https://zhuanlan.zhihu.com/p/459065530)
68. [LayerNorm与BatchNorm](https://zhuanlan.zhihu.com/p/456863215)
69. [Self-Attention](https://zhuanlan.zhihu.com/p/455399791)
70. [Positional Encoding](https://zhuanlan.zhihu.com/p/454482273)

## 下一阶段

1. 深读 Week 01 的三篇首要参考，产出原创试讲义。
2. 为每篇首要参考记录：核心问题、概念图、推导、系统机制、容易误解点、可验证实验和时效性风险。
3. 图示采用原创重绘；如作者授权复用，则增补授权状态、原图链接和逐图署名。
4. 对照当前 vLLM 官方文档和论文，标出旧版源码与 vLLM V1 的差异，避免把历史实现当作当前事实。
