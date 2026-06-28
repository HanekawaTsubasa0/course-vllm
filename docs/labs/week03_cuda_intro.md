# Week 03 CUDA 入门

目标：掌握 CUDA kernel 的基本结构、host/device 协同、正确性验证和计时方法。

## 代码入口

- `kernels/vector_add.cu`
- `course_vllm/kernels/harness.py`
- `tests/test_kernels.py`

## 实验任务

1. 阅读 `vector_add.cu`，说明 grid、block、thread index 如何覆盖一维数组。
2. 修改 vector add 的 block size，观察 correctness 和 micro benchmark。
3. 使用 `load_cuda_extension` 编译 kernel。
4. 记录首次 JIT 编译时间和热启动运行时间的区别。

## 验证

```bash
python -m pytest -q tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch -rs
python -m course_vllm.benchmarks.grader week03
```

## 交付物

- vector add 正确性误差。
- block size 对运行时间的影响。
- 后续 kernel 可复用的测试模板说明。
