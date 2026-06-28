# Week 04 RMSNorm 与 RoPE

目标：实现并接入 RMSNorm/RoPE CUDA kernel，理解数值误差、混合精度和主路径 dispatch。

## 背景概念

Qwen3 使用 RMSNorm 和 RoPE。RMSNorm 需要对 hidden dimension 做平方均值归约，再乘以权重；RoPE 按位置把 query/key 的偶奇维旋转。它们都适合作为“从数学公式到 CUDA kernel 再到模型主路径”的第一组算子。

本周关注三件事：

- 数值正确性：CUDA 输出与 PyTorch reference 的误差。
- dispatch 行为：`torch`、`auto`、`cuda` 三种路径的差别。
- 主路径接入：短生成时 kernel 是否真的被模型调用。

## 读什么

- `course_vllm/model/qwen3_torch.py::Qwen3RMSNorm.forward`
- `course_vllm/model/qwen3_torch.py::apply_rotary_pos_emb`
- `course_vllm/kernels/cuda_ops.py::cuda_rms_norm`
- `course_vllm/kernels/cuda_ops.py::cuda_rope`
- `kernels/course_ops.cu`
- `tests/test_kernels.py::test_cuda_rms_norm_matches_torch`
- `tests/test_kernels.py::test_cuda_rope_matches_qwen3_rotate_half`

阅读时回答：

1. RMSNorm 和 LayerNorm 少了哪一步？
2. RoPE 为什么要同时作用在 query 和 key 上？
3. `kernel_impl=auto` 和 `kernel_impl=cuda` 在异常处理上有什么区别？

## 改什么

### TODO(lab04)

- Edit: `kernels/course_ops.cu` 中 RMSNorm/RoPE kernel 对应实现。
- Edit if needed: `course_vllm/kernels/cuda_ops.py::cuda_rms_norm`、`cuda_rope` wrapper。
- Integration check: `course_vllm/model/qwen3_torch.py::Qwen3RMSNorm.forward` 和 `apply_rotary_pos_emb` 的 `kernel_impl` 分支。

改动范围：

- 可以改 CUDA kernel body、launcher 参数和 wrapper 的 shape/dtype 检查。
- 不要改 PyTorch reference 数学公式。
- 不要放宽测试容差来掩盖错误。

## 测什么

```bash
python -m pytest -q \
  tests/test_kernels.py::test_cuda_rms_norm_matches_torch \
  tests/test_kernels.py::test_cuda_rope_matches_qwen3_rotate_half \
  -rs

python examples/offline_generate.py \
  --backend course \
  --kernel-impl auto \
  --stage week04 \
  --max-tokens 8 \
  --temperature 0

python -m course_vllm.benchmarks.grader week04
```

严格 CUDA 接入验收：

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
```

## 报告问题

1. 给出 RMSNorm 和 RoPE 的最大绝对误差。
2. 说明误差主要来自 float32、float16/bfloat16，还是归约顺序。
3. 用一段调用链说明短生成如何从 `offline_generate.py` 走到 CUDA wrapper。

## 常见坑

- RMSNorm 归约维度必须是 hidden dimension，不是 batch dimension。
- RoPE 的 `cos`/`sin` shape 需要能 broadcast 到 query/key。
- `auto` 会 fallback；确认接入时要用 `cuda_smoke` 或 `--kernel-impl cuda`。

## 交付物

- RMSNorm 正确性误差。
- RoPE 正确性误差。
- `torch` 与 `auto/cuda` 路径的性能和行为对比。
