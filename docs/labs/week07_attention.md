# Week 07 Attention

目标：理解 prefill/decode attention 的不同形态，并实现可验证的 CUDA 路径。

## 背景概念

Attention 是 LLM serving 中最核心的算子之一。Prefill 一次处理多个 query position；decode 每步只有一个新 query，但要读完整历史 KV。Paged decode 还需要通过 block table 把逻辑 token 位置映射到物理 KV slot。

本周关注：

- dense attention reference 如何固定 correctness。
- online softmax 如何避免显式保存完整 score matrix。
- paged decode attention 如何读取分块 KV cache。
- 教学 kernel 与 FlashAttention/vLLM 生产 kernel 的差距。

## 读什么

- `course_vllm/model/ops.py::dense_attention_decode_reference`
- `course_vllm/model/ops.py::dense_attention_decode`
- `course_vllm/model/attention.py::paged_attention_decode_reference`
- `course_vllm/model/attention.py::paged_attention_decode`
- `course_vllm/engine/paged_kv_cache.py`
- `kernels/course_ops.cu`
- `tests/test_attention.py`

阅读时回答：

1. Prefill 和 decode 的 query/key/value shape 有什么区别？
2. Paged attention 为什么需要 block table 和 context length？
3. Online softmax 里的 row max、denom、acc 分别保存什么？

## 改什么

### TODO(lab07)

- Edit: `kernels/course_ops.cu` 中 dense attention 或 paged decode attention 教学 kernel。
- Edit if needed: `course_vllm/model/ops.py` 和 `course_vllm/model/attention.py` 的 `kernel_impl` dispatch。
- Keep reference: `dense_attention_decode_reference` 和 `paged_attention_decode_reference` 不作为学生改动目标。

建议拆分：

1. 先让 dense decode CUDA 与 reference 对齐。
2. 再处理 paged decode 的 block table 和 context length。
3. 最后记录 tiled/online softmax 可以如何减少中间矩阵。

改动范围：

- 可以改 CUDA attention kernel、launcher、dispatch。
- 不要改 reference 实现或测试 expected。
- 不要求超过 FlashAttention/cuDNN/vLLM kernel 性能。

## 测什么

```bash
python -m pytest -q tests/test_attention.py -rs
python -m pytest -q tests/test_qwen3_torch.py::test_dense_attention_prefill_reference_matches_torch_attention
python -m course_vllm.benchmarks.grader week07
```

严格 CUDA 接入验收：

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
```

## 报告问题

1. 对比 dense decode 和 paged decode 的输入 shape。
2. 给出 CUDA attention 与 reference 的最大误差。
3. 说明教学 kernel 与 FlashAttention/vLLM 生产 kernel 至少三个差距。
4. 解释为什么 decode attention 通常更受 KV 读取和 batch 组织影响。

## 常见坑

- GQA 下 query heads 和 KV heads 数量可能不同，需要正确 repeat KV。
- block table 不完整时应该报错，而不是静默读错 KV。
- `auto` fallback 会隐藏 CUDA dispatch 问题；阶段验收要跑 `cuda_smoke`。

## 交付物

- attention correctness 测试结果。
- prefill/decode/paged decode 路径图。
- online softmax 或 FlashAttention 风格内存节省说明。
