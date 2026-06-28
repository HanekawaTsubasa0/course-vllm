# Week 06 归约与 Softmax

目标：掌握并行归约、稳定 softmax、log-sum-exp 思路，并把 softmax 接入 sampling 路径。

## 代码入口

- `kernels/course_ops.cu`
- `course_vllm/kernels/cuda_ops.py`
- `course_vllm/engine/sampler.py`
- `tests/test_sampler.py`

## 实验任务

1. 阅读 row-wise softmax kernel 的 max reduction 和 sum reduction。
2. 构造大正数 logits，对比朴素 exp 和稳定 softmax。
3. 在 CUDA 可用时让 sampling softmax 走 `cuda_softmax`。
4. 记录 greedy、temperature、top-k 三种采样路径。

## TODO(lab06)

- Edit: `kernels/course_ops.cu` 中 row-wise stable softmax。
- Edit if needed: `course_vllm/engine/sampler.py` 中 logits softmax dispatch。
- Keep reference: greedy、temperature、top-k 语义不能改变，只替换 softmax 计算路径。

## 验证

```bash
python -m pytest -q tests/test_kernels.py::test_cuda_softmax_matches_torch tests/test_sampler.py -rs
python -m course_vllm.benchmarks.grader week06
```

## 交付物

- softmax 正确性误差。
- 溢出样例与稳定实现解释。
- sampling 路径调用链。
