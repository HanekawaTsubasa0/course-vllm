# 《大模型推理服务：从算子到系统》课程讲义

这套讲义回答一个贯穿全书的问题：一个只能离线生成文本的模型，怎样变成可度量、可并发、可扩展、可优化的在线推理服务？

讲义按知识依赖组织为五篇，而不是把十六周写成彼此独立的知识点清单。正文只承担教学任务：解释概念、推导机制、分析代码和讨论工程取舍。环境配置、操作命令、提交要求和评分标准另见实验任务书。

## 阅读地图

### 第一篇：服务与性能基础

- [第 1 章：大语言模型在线推理的请求流程与性能指标](part1_foundations/week01_serving_metrics.md)
- [第 2 章：GPU 性能分析的证据层次与 Roofline 模型](part1_foundations/week02_profiling_roofline.md)

这一篇建立全书共同语言：token、prefill、decode、KV cache、TTFT、TPOT、吞吐、尾延迟、GPU 异步执行和 Roofline。

### 第二篇：GPU 编程与核心算子

- [第 3 章：PyTorch CUDA Extension 的执行模型](part2_gpu_kernels/week03_cuda_extension.md)（来源驱动样章）
- [第 4 章：RMSNorm 与 RoPE 的数学定义及 GPU 映射](part2_gpu_kernels/week04_rmsnorm_rope.md)
- [第 5 章：GEMM 的数据复用与 Linear 算子实现](part2_gpu_kernels/week05_matmul_linear.md)
- [第 6 章：从 Logits 到 Token 的 Softmax 与 Sampling](part2_gpu_kernels/week06_softmax_sampling.md)
- [第 7 章：Attention 的张量语义与 IO-Aware 实现](part2_gpu_kernels/week07_attention.md)

这一篇从最小 CUDA 闭环出发，逐步进入归约、数据复用、矩阵乘、概率计算和 Attention。重点不是背 API，而是把数学、张量布局、线程工作和硬件瓶颈对应起来。

### 第三篇：KV 状态与推理引擎

- [第 8 章：自回归推理中的 KV Cache](part3_serving_engine/week08_kv_cache.md)（来源驱动样章）
- [第 9 章：在线推理引擎的请求状态与资源生命周期](part3_serving_engine/week09_engine_request_lifecycle.md)
- [第 10 章：Paged KV Cache 的地址映射与内存管理](part3_serving_engine/week10_paged_kv_block_manager.md)（来源驱动样章）
- [第 11 章：迭代级调度与 Continuous Batching](part3_serving_engine/week11_continuous_batching.md)
- [第 12 章：系统优化必须闭环](part3_serving_engine/week12_system_optimization.md)

这一篇把视角从单次前向扩展到多请求系统：历史状态怎样存放，请求怎样调度，显存怎样分配，优化结论怎样用证据闭环。

### 第四篇：规模化、异构平台与前沿机制

- [第 13 章：容量规划与并行策略](part4_scale_frontier/week13_capacity_parallelism.md)
- [第 14 章：Ascend C 编程模型导读](part4_scale_frontier/week14_ascendc_comparison.md)（实验暂缓）
- [第 15 章：怎样读懂新的 LLM Serving 机制](part4_scale_frontier/week15_frontier_serving.md)

这一篇讨论单卡之外的问题。Ascend C 当前只做概念对照，不要求硬件实验；前沿专题强调从论文主张追到系统状态、资源变化和评价指标。

### 第五篇：系统综合

- [第 16 章：把 LLM Serving 系统讲成一条证据链](part5_synthesis/week16_final_review.md)

最后一篇把正确性、接入、性能、容量和可靠性证据组织成完整的系统论证。

## 编写与维护

- [全书重构计划](BOOK_PLAN.md)：记录篇章目标、重写顺序和当前状态。
- [讲义写作规范](STYLE_GUIDE.md)：规定章节结构、术语、公式、代码、图表和 PPT 素材规则。
- [全书来源计划](references/source_plan.md)：记录每章优先采用的论文、官方文档和课程材料。
- [图表素材目录](assets/README.md)：保存可复用于讲义和 PPT 的原始素材。
- [旧稿资料映射](references/legacy_source_map.md)：仅用于追溯上一版资料来源，不约束新讲义结构。
- [旧稿校验记录](archive/expanded_draft_validation_report.md)：记录保存提交中的统计结果，不代表新教材已经完成验收。

## 当前状态

第 3 章和第 10 章是第一批来源驱动样章，分别验证“GPU 算子课”和“推理系统课”两种写法。两章以官方文档和原始论文支撑通用结论，以课程源码说明教学实现，并在相关论述之后就近给出链接。其余章节暂时保留上一版正文作为素材，后续按 `BOOK_PLAN.md` 的依赖顺序重写。
