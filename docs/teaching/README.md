# course-vllm 学习讲义

这个目录放按周学习用的“教材式讲义”，不是实验步骤清单，也不是代码索引。阅读顺序是先理解本周知识，再回到 docs/labs/ 完成对应实验。

每周讲义都遵守同一个边界：详细讲本周主题本身，不把整门课的内容反复重讲。必要的前置知识只做简短衔接，主体放在本周理论、机制和工程取舍上。项目代码只在末尾作为少量对照出现，不能代替知识讲解。

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
| Week14 | AscendC | `week14_ascend_deferred.md` |
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
