# course-vllm 实验进度

这组文档把主线工程拆成 16 个可检查的实验阶段。工程默认保持完整可运行；学生实现可以逐步替换 reference path，并通过对应测试验证。

| 周次 | 主题 | 代码状态 | 主要入口 |
| --- | --- | --- | --- |
| week01 | 课程导论与 baseline serving | implemented | `course_vllm.server.api` |
| week02 | 性能分析 | implemented | `scripts/profile/` |
| week03 | CUDA 入门 | implemented | `kernels/vector_add.cu` |
| week04 | RMSNorm 与 RoPE | implemented | `--kernel-impl auto` |
| week05 | 线性层与矩阵乘 | implemented | `CourseLinear`, `cuda_matmul` |
| week06 | 归约与 Softmax | implemented | `Sampler`, `cuda_softmax` |
| week07 | Attention | implemented | `dense_attention_prefill`, `paged_attention_decode` |
| week08 | KV Cache | implemented | `engine/kv_cache.py` |
| week09 | 推理引擎 | implemented | `engine/engine.py` |
| week10 | 分页 KV Cache | implemented | `engine/paged_kv_cache.py` |
| week11 | 连续批处理 | implemented | `engine/scheduler.py` |
| week12 | 系统优化 | implemented | `--pinned-memory`, `--transfer-stream` |
| week13 | 多卡推理容量规划 | implemented | `benchmarks/capacity_planner.py` |
| week14 | AscendC | deferred | 后续有后端/硬件后再补 |
| week15 | 前沿专题 | implemented | `cache_aware_demo.py` |
| week16 | 总结展示 | implemented | final report |

运行时可通过 `/health` 查看当前 `stage`、`backend` 和 `kernel_impl`。

按周自动检查：

```bash
python -m course_vllm.benchmarks.grader week11
python -m course_vllm.benchmarks.grader week15
```

第十四节 AscendC 当前按项目决策暂缓，仓库内不放本地后端或样例；后续有硬件/后端后再补真实算子和对照文档。
