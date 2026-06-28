# Week 07 Attention

目标：理解 prefill/decode attention 的不同形态，并实现可验证的 CUDA 路径。

## 当前状态

- dense prefill attention 有 online-softmax reference，并通过 `--kernel-impl auto|cuda` 暴露课程 CUDA dispatch。
- paged decode attention 已有 CUDA dispatch。
- 当前 CUDA kernel 是教学版，每个 query row 在线更新 softmax 统计量；优化方向是加入更完整的 tile/block 数据复用。

## 建议拆分

1. 用 PyTorch reference 固定输入输出。
2. 实现 dense decode 或 small prefill CUDA kernel。
3. 改成 tiled online softmax，记录显存访问和中间矩阵大小变化。
4. 与 `paged_attention_decode_reference` 比较误差。

## TODO(lab07)

- Edit: `kernels/course_ops.cu` 中 dense attention 或 paged decode attention 教学 kernel。
- Edit if needed: `course_vllm/model/ops.py` 和 `course_vllm/model/attention.py` 的 `kernel_impl` dispatch。
- Keep reference: `dense_attention_decode_reference` 和 `paged_attention_decode_reference` 不作为学生改动目标。

## 验证

```bash
python -m pytest -q tests/test_attention.py -rs
python -m pytest -q tests/test_qwen3_torch.py::test_dense_attention_prefill_reference_matches_torch_attention
```
