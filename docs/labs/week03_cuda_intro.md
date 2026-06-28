# Week 03 CUDA 入门

目标：掌握 CUDA kernel 的基本结构、host/device 协同、正确性验证和计时方法。

## 背景概念

本周只做一维 vector add。它不追求性能复杂度，而是建立后续所有 CUDA lab 都会复用的工作流：

- 用 grid/block/thread index 映射数据下标。
- 用 PyTorch extension JIT 编译 `.cu` 文件。
- 用 PyTorch reference 做 correctness oracle。
- 区分首次 JIT 编译时间和热启动 kernel 时间。

## 读什么

- `kernels/vector_add.cu`
- `course_vllm/kernels/harness.py::load_cuda_extension`
- `tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch`

阅读时回答：

1. `blockIdx.x * blockDim.x + threadIdx.x` 对应哪个元素？
2. 为什么 kernel 里需要判断 `idx < n`？
3. `load_cuda_extension` 会把源码编译到哪里？

## 改什么

### TODO(lab03)

- Edit: `kernels/vector_add.cu` 中的 vector add kernel 和 block/thread 配置实验。
- Read-only: `course_vllm/kernels/harness.py::load_cuda_extension`，理解 JIT 编译流程。
- Test target: `tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch`。

改动范围：

- 可以改 kernel body、block size、launch 参数。
- 不要改测试里的 expected 结果。
- 不要把 PyTorch reference 作为 kernel 输出返回。

## 测什么

```bash
python -m pytest -q tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch -rs
python -m course_vllm.benchmarks.grader week03
```

建议额外记录：

```bash
time python -m pytest -q tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch -rs
time python -m pytest -q tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch -rs
```

第一次通常包含 JIT 编译时间，第二次更接近热启动。

## 报告问题

1. 画出 `n=1024`、`block_size=256` 时 grid/block/thread 的覆盖关系。
2. 如果 `n` 不是 block size 的整数倍，会发生什么？
3. 首次运行和第二次运行时间差异来自哪里？

## 常见坑

- 当前 shell 看不到 CUDA 时，测试会 skip；这不是合格 CUDA 验收结果。
- 修改 `.cu` 后如果怀疑 JIT 缓存影响，可以清理对应 torch extension cache。
- kernel 越界通常不会立刻给出清晰 Python traceback，要优先检查 index guard。

## 交付物

- vector add 正确性结果。
- block size 对运行时间的影响。
- 后续 kernel 可复用的测试模板说明。
