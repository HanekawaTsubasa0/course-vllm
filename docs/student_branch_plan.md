# Student Branch Generation Plan

`main` 保留完整可运行答案和 TA 文档。正式发给学生时，从 `main` 生成 `student` 分支，保留工程可启动、测试可运行和接口稳定，但把每周核心实现替换为明确的 TODO 区域。

## 目标

- 学生看到的是 starter code，不是完整答案。
- 每个 lab 的源码 TODO 与 `docs/labs/week*.md` 对应。
- Grader、测试、报告模板和 demo 命令保留。
- TA 可以从同一个 `main` 重复生成 student branch，避免长期维护两份代码。

## 标记约定

源码中使用统一标记：

```python
# TODO(lab04): implement RMSNorm CUDA dispatch.
# BEGIN SOLUTION
...
# END SOLUTION
```

生成 student branch 时：

- 保留函数签名、参数检查、docstring 和错误信息。
- 将 `BEGIN SOLUTION` 到 `END SOLUTION` 的主体替换为 `raise NotImplementedError("TODO(labXX): ...")` 或最小 reference fallback。
- 对 CUDA `.cu` 文件，保留 kernel launcher 和 shape 注释，把 kernel body 替换为可编译的占位实现或 `// TODO(labXX)`。

## 初始试点

先做 Week 03-07，因为这些最能体现“读什么、改什么、测什么”：

| Lab | 文件 | 目标 |
| --- | --- | --- |
| lab03 | `kernels/vector_add.cu` | vector add kernel body |
| lab04 | `kernels/course_ops.cu`, `course_vllm/model/qwen3_torch.py` | RMSNorm/RoPE kernel 与 dispatch |
| lab05 | `kernels/course_ops.cu`, `course_vllm/model/ops.py` | naive/tiled matmul 与 `CourseLinear` |
| lab06 | `kernels/course_ops.cu`, `course_vllm/engine/sampler.py` | stable softmax 与 sampling dispatch |
| lab07 | `kernels/course_ops.cu`, `course_vllm/model/attention.py`, `course_vllm/model/ops.py` | attention CUDA path |

系统周次随后补：

| Lab | 文件 | 目标 |
| --- | --- | --- |
| lab08 | `course_vllm/engine/kv_cache.py` | continuous KV cache |
| lab09 | `course_vllm/engine/engine.py`, `course_vllm/engine/request.py` | request lifecycle |
| lab10 | `course_vllm/engine/block_manager.py`, `course_vllm/engine/paged_kv_cache.py` | paged KV/block table |
| lab11 | `course_vllm/engine/scheduler.py`, `course_vllm/server/batching.py` | continuous batching |
| lab12 | `course_vllm/model/qwen3_continuous_backend.py`, `course_vllm/server/batching.py` | system optimization/admission |

## Release Procedure

1. Ensure `main` is clean and validation passes.
2. Add or update `BEGIN SOLUTION` / `END SOLUTION` markers on `main`.
3. Run `python scripts/validation/student_branch_preview.py` and review the target list.
4. Create branch:

   ```bash
   git switch -c student
   ```

5. Run the solution-stripping script once implemented.
6. Run student-branch smoke tests:

   ```bash
   pytest -q -rs --ignore=tests/test_kernels.py --ignore=tests/test_attention.py
   python -m course_vllm.benchmarks.grader week01
   python -m course_vllm.benchmarks.grader week02
   ```

7. Push `student` only after TA review.

## Current Status

This repository now contains lab handouts with `TODO(labXX)` scopes. The next implementation step is to add solution markers in source files and implement the stripping script. Until then, `main` remains the authoritative complete version.
