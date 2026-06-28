# Week 08 KV Cache

目标：理解 KV cache 的作用、连续缓存组织、append、不同长度 batch 的处理和多 token 连续生成。

## 代码入口

- `course_vllm/engine/kv_cache.py`
- `course_vllm/model/qwen3_continuous_backend.py`
- `tests/test_kv_cache.py`
- `tests/test_qwen3_torch.py`

## 实验任务

1. 跟踪单请求 prefill 后的 KV cache handle。
2. 观察 decode 每步只输入最新 token，但 attention 可访问历史 K/V。
3. 对比不开 cache 的全序列 forward 和 cache decode 的数据量。
4. 解释 batch decode 为什么要按历史长度分桶。

## TODO(lab08)

- Edit: `course_vllm/engine/kv_cache.py` 中 cache allocation、append、metadata 维护逻辑。
- Inspect integration: `course_vllm/model/qwen3_continuous_backend.py` 的 prefill/decode cache 使用。
- Keep boundary: 不改 tokenizer、sampling 或 HTTP API。

## 验证

```bash
python -m pytest -q tests/test_kv_cache.py tests/test_qwen3_torch.py
python -m course_vllm.benchmarks.grader week08
```

## 交付物

- KV cache 张量 shape 表。
- 连续生成多个 token 的调用链。
- cache 正确性测试结果。
