# Week 05 线性层与矩阵乘

目标：理解线性层、朴素 matmul、tiled matmul、cuBLAS/PyTorch 对照和主线投影层接入。

## 代码入口

- `kernels/course_ops.cu`
- `kernels/course_ops.cpp`
- `course_vllm/kernels/cuda_ops.py`
- `course_vllm/model/ops.py`
- `tests/test_kernels.py`

## 当前主路径

`CourseLinear` 替代 Qwen3 中的 Q/K/V/O/MLP 投影层。`--kernel-impl auto|cuda` 在 CUDA tensor 上调用 `cuda_matmul_tiled`，朴素 `cuda_matmul` 保留为 roofline 对照。

## 实验任务

1. 对比 naive 和 tiled matmul 的访存模式。
2. 计算一次 `M*N*K` GEMM 的 FLOPs 与读写字节数。
3. 对照 PyTorch `a @ b`，解释为什么教学 kernel 不追求超过 cuBLAS。
4. 用 `--kernel-impl cuda` 验证投影层确实走课程 kernel。

## 验证

```bash
python -m pytest -q tests/test_kernels.py::test_cuda_matmul_matches_torch tests/test_kernels.py::test_cuda_matmul_tiled_matches_torch -rs
python -m pytest -q tests/test_qwen3_torch.py::test_course_linear_matches_torch_linear
python -m course_vllm.benchmarks.grader week05
```

## 交付物

- naive vs tiled 正确性误差。
- naive vs tiled vs PyTorch 运行时间。
- `CourseLinear` 接入位置说明。
