# course-vllm 实验进度

这组文档把主线工程拆成 16 个可检查的实验阶段。学生分支中的核心实验代码带有 `TODO(labXX)` 标记；你需要逐周补全实现，并通过对应测试验证。

本文档是学生入口，只使用相对路径、示例命令、实验任务和交付物。

Week 03、Week 04、Week 07 已扩成正式 handout 样板，包含“背景概念、读什么、改什么、测什么、报告问题、常见坑、交付物”。其余周次后续按同一结构扩展。

| 周次 | 主题 | 任务状态 | 主要入口 |
| --- | --- | --- | --- |
| week01 | 课程导论与 baseline serving | run and explain | `course_vllm.server.api` |
| week02 | 性能分析 | profile and report | `scripts/profile/` |
| week03 | CUDA 入门 | TODO(lab03) | `kernels/vector_add.cu` |
| week04 | RMSNorm 与 RoPE | TODO(lab04) | `--kernel-impl auto` |
| week05 | 线性层与矩阵乘 | TODO(lab05) | `CourseLinear`, `cuda_matmul` |
| week06 | 归约与 Softmax | TODO(lab06) | `Sampler`, `cuda_softmax` |
| week07 | Attention | TODO(lab07) | `dense_attention_prefill`, `paged_attention_decode` |
| week08 | KV Cache | TODO(lab08) | `engine/kv_cache.py` |
| week09 | 推理引擎 | TODO(lab09) | `engine/request.py` |
| week10 | 分页 KV Cache | TODO(lab10) | `engine/paged_kv_cache.py` |
| week11 | 连续批处理 | TODO(lab11) | `engine/scheduler.py` |
| week12 | 系统优化 | TODO(lab12) | `--pinned-memory`, `--transfer-stream` |
| week13 | 多卡推理容量规划 | run and report | `benchmarks/capacity_planner.py` |
| week14 | AscendC | deferred | 后续有后端/硬件后再补 |
| week15 | 前沿专题 | run and report | `cache_aware_demo.py` |
| week16 | 总结展示 | final report | final report |

运行时可通过 `/health` 查看当前 `stage`、`backend` 和 `kernel_impl`。

按周自动检查：

```bash
python -m course_vllm.benchmarks.grader week11
python -m course_vllm.benchmarks.grader week15
```

CUDA 相关阶段可以先使用 `--kernel-impl auto` 调试。需要确认 CUDA kernel 真正接入时，运行：

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
```

第十四节 AscendC 当前按项目决策暂缓，仓库内不放本地后端或样例；后续有硬件/后端后再补真实算子和对照文档。
