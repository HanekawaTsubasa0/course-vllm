# LLM Serving 课程讲义总目录

这套讲义围绕一个问题展开：怎样把“模型能够生成文本”，逐步建设成一个可解释、可验证、可优化的 LLM Serving 系统。

课程不是按零散技术名词排列，而是沿着一条完整链路推进：先理解请求与指标，再进入 GPU 算子和 Attention，随后管理 KV 状态、请求调度与系统容量，最后学习如何读前沿工作并用证据完成验收。

## 第一阶段：先把请求和性能问题说清楚

### [Week 01：一次回答是怎样被生成出来的](week01_serving_metrics.md)

从 Token、Logits、Sampling 和 Chat Template 开始，完整解释 Prefill、Decode、KV Cache、Streaming、TTFT、TPOT、吞吐、长尾延迟与 Goodput。目标是建立后续所有优化共同使用的语言。

### [Week 02：性能分析不是猜哪里慢](week02_profiling_roofline.md)

建立服务指标、框架算子、系统时间线和单 Kernel 微架构四层证据。重点学习 GPU 异步计时、Warmup、Roofline、Arithmetic Intensity，以及怎样把现象写成可验证的性能假设。

## 第二阶段：看清模型一步怎样在 GPU 上执行

### [Week 03：从 PyTorch 算子走到 CUDA Kernel](week03_cuda_extension.md)

解释 Extension、Binding、Dispatch、Launch Configuration、Grid-Stride Loop、内存合并访问与正确性验收。重点不是“写出能跑的 CUDA”，而是证明 GPU 路径真的被调用，并且数值和边界条件正确。

### [Week 04：RMSNorm 与 RoPE](week04_rmsnorm_rope.md)

拆解归一化和位置编码的数学含义、数值稳定性、Reduction、向量化与融合机会。通过 RMSNorm 和 RoPE 观察公式、张量布局与 GPU 实现之间的对应关系。

### [Week 05：矩阵乘与 Linear](week05_matmul_linear.md)

从朴素 GEMM 进入 Tiling、Shared Memory、数据复用、Tensor Core、算子融合和形状敏感性。解释为什么“矩阵乘法计算量很大”并不自动意味着实现已经接近硬件峰值。

### [Week 06：Softmax、Sampling 与生成策略](week06_softmax_sampling.md)

解释稳定 Softmax、Temperature、Top-k、Top-p、Greedy 与随机采样；把数值变换、概率分布和用户可见生成行为连起来，并讨论 GPU 上 Reduction 与采样流水线的成本。

### [Week 07：Attention 为什么难以高效实现](week07_attention.md)

从标准 Attention 的矩阵公式出发，解释因果掩码、缩放、Online Softmax、FlashAttention、Prefill/Decode 形状差异，以及为什么减少中间张量读写往往比减少少量计算更重要。

## 第三阶段：管理历史状态和请求生命周期

### [Week 08：KV Cache 是怎样工作的](week08_kv_cache.md)

解释 KV Cache 为什么能够消除重复计算、每个 Token 的缓存占用怎样估算、张量布局如何影响访问效率，以及容量、并发和最大上下文之间为什么必须共同规划。

### [Week 09：一个请求如何穿过推理引擎](week09_engine_request_lifecycle.md)

沿 API、Engine、Request、Sequence、Scheduler、Model Runner 和输出处理追踪请求。重点区分控制面与数据面、Waiting 与 Running、逻辑状态与 GPU 执行状态。

### [Week 10：Paged KV 与 Block Manager](week10_paged_kv_block_manager.md)

把连续 KV 预留改写为逻辑块到物理块的映射问题。解释内部碎片、Block Table、分配/释放、引用计数、Copy-on-Write，以及分页机制为何不改变 Attention 的数学语义。

### [Week 11：Continuous Batching](week11_continuous_batching.md)

解释迭代级调度、动态加入与退出、Token Budget、Chunked Prefill、公平性和长尾延迟。通过调度时间线理解为什么连续批处理不是“把 batch size 调大”。

### [Week 12：系统优化必须闭环](week12_system_optimization.md)

把优化写成 Observation、Hypothesis、Change、Evidence、Conclusion 的证据链。覆盖 CPU Overhead、Kernel Launch、同步、内存分配、算子融合、CUDA Graph 和性能回归保护。

## 第四阶段：扩展容量、跨硬件迁移并跟进前沿机制

### [Week 13：容量规划与并行策略](week13_capacity_parallelism.md)

从权重、KV Cache、激活和运行时开销推导显存预算，再比较 Tensor、Pipeline、Data 和 Expert Parallel。解释通信量、负载均衡、扩展效率与 SLO 之间的关系。

### [Week 14：用 Ascend C 重做一次算子推理](week14_ascendc_comparison.md)

不做 CUDA API 的逐词翻译，而是比较硬件层级、执行单元、内存层次、流水与编程模型。通过同一算子的双平台实现，训练从算法语义到设备映射的迁移能力。

### [Week 15：怎样读懂一个新的 LLM Serving 机制](week15_frontier_serving.md)

用 Paper-to-System 五问分析 Prefix Cache、Cache-Aware Scheduling、Prefill/Decode 分离、KV Transfer、Chunked Prefill、Token-Level Scheduling、Speculative Decoding 与 MLA，明确收益、代价、状态和公平对照。

### [Week 16：把整个 LLM Serving 系统讲成一条证据链](week16_final_review.md)

把 Correctness、Dispatch、Integration、Performance、Capacity 和 Reliability 六类证据组织成最终展示。提供 Claim-Evidence Matrix、A/B 实验、故障复盘、复现清单和答辩自检问题。

## 建议使用方式

第一次学习时按 Week 01 到 Week 16 顺序阅读。每周先回答开头提出的问题，再读概念与推导，最后完成实验和自检；不要只看代码片段。

回查问题时可以按对象定位：指标问题看 Week 01-02，算子问题看 Week 03-07，KV 和调度问题看 Week 08-12，容量与前沿机制看 Week 13-15，最终验收看 Week 16。

讲义中的 Mermaid 图是为课程重新设计的结构图与时间线；外部资料用于进一步阅读，并在各周末尾单独列出。知乎文章只作为参考来源之一，不替代课程自身的论证和实验。

## 交付统计

- 讲义数量：16 篇
- 正文总行数：7,613 行
- 正文总大小：222,339 字节
- Mermaid 原创图示：45 张
- 外部链接：73 处，去重后 30 个
- 验证详情：[课程讲义验证报告](course_validation_report.md)
- 资料映射：[猛猿文章与课程素材映射](source_map.md)
