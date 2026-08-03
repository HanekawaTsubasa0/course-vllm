# course-vllm 学习讲义

这个目录放按周学习用的“教材式讲义”。讲义的目标是把每周知识讲清楚，尤其照顾基础薄弱的同学：术语第一次出现要解释，公式要说明每个符号，shape 要说明每一维代表什么，系统机制要按步骤展开。

每周讲义都遵守同一个边界：先学习概念、背景和原理，再去做练习。这里不代替操作手册，也不代替代码说明；阅读时应该先把“为什么需要这个机制”“这个机制解决什么问题”“它的代价是什么”想清楚。

## 周次索引

| 周次 | 主题 | 文档 |
| --- | --- | --- |
| Week01 | LLM serving 流程与指标 | `week01_serving_metrics.md` |
| Week02 | Profiling 与性能分析 | `week02_profiling_roofline.md` |
| Week03 | CUDA extension 入门 | `week03_cuda_extension.md` |
| Week04 | RMSNorm 与 RoPE | `week04_rmsnorm_rope.md` |
| Week05 | Matmul 与 Linear | `week05_matmul_linear.md` |
| Week06 | Softmax 与 Sampling | `week06_softmax_sampling.md` |
| Week07 | Attention | `week07_attention.md` |
| Week08 | KV cache | `week08_kv_cache.md` |
| Week09 | 推理 engine 与请求生命周期 | `week09_engine_request_lifecycle.md` |
| Week10 | Paged KV 与 block manager | `week10_paged_kv_block_manager.md` |
| Week11 | Continuous batching | `week11_continuous_batching.md` |
| Week12 | 系统优化与 admission control | `week12_system_optimization.md` |
| Week13 | 多卡容量规划与并行策略 | `week13_capacity_parallelism.md` |
| Week14 | AscendC 概念导读（实验暂缓） | `week14_ascend_deferred.md` |
| Week15 | 前沿 serving 策略 | `week15_frontier_serving.md` |
| Week16 | 系统总复习 | `week16_final_review.md` |

## 参考资料

- vLLM / PagedAttention: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Orca: [A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- NVIDIA CUDA C++ Programming Guide: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- NVIDIA CUDA C++ Best Practices Guide: [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- NVIDIA Nsight Systems: [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- NVIDIA Nsight Compute: [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- Hugging Face Transformers: [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- Megatron-LM: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
