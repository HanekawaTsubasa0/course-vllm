# Week 04 RMSNorm 与 RoPE

目标：实现并接入 RMSNorm/RoPE CUDA kernel，理解数值误差和混合精度。

## 当前代码路径

- PyTorch reference：`course_vllm/model/qwen3_torch.py`
- CUDA wrapper：`course_vllm/kernels/cuda_ops.py`
- CUDA 源码：`kernels/course_ops.cu`
- 主线开关：`--kernel-impl torch|auto|cuda`

`auto` 会在 CUDA tensor 上尝试课程 CUDA kernel，失败时回退 PyTorch；`cuda` 会在 kernel 不可用时直接报错，适合检查是否真的接入。

## TODO(lab04)

- Edit: `kernels/course_ops.cu` 中 RMSNorm/RoPE kernel 对应实现。
- Edit if needed: `course_vllm/kernels/cuda_ops.py::cuda_rms_norm`、`cuda_rope` wrapper。
- Integration check: `course_vllm/model/qwen3_torch.py::Qwen3RMSNorm.forward` 和 `apply_rotary_pos_emb` 的 `kernel_impl` 分支。

## 验证

```bash
python -m pytest -q tests/test_kernels.py::test_cuda_rms_norm_matches_torch tests/test_kernels.py::test_cuda_rope_matches_qwen3_rotate_half -rs
python examples/offline_generate.py --backend course --kernel-impl auto --stage week04 --max-tokens 8 --temperature 0
```

## 交付物

- RMSNorm 正确性误差。
- RoPE 正确性误差。
- `torch` 与 `auto/cuda` 路径的性能和行为对比。
