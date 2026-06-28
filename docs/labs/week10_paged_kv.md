# Week 10 分页 KV Cache

目标：掌握 block manager、block table、slot mapping、教学版 prefix cache 和碎片统计。

说明：本实验的 prefix cache 是 teaching approximation，用于观察完整 block 前缀复用和碎片统计；它不包含生产系统里的哈希匹配、引用计数、淘汰策略和跨请求生命周期管理。

## 代码入口

- `course_vllm/engine/block_manager.py`
- `course_vllm/engine/paged_kv_cache.py`
- `course_vllm/model/qwen3_paged_backend.py`
- `examples/block_usage.py`

## 实验任务

1. 用固定 block size 分配多个不同长度序列。
2. 输出 block table 和 slot mapping。
3. 构造共享完整 block 前缀的 prompt，观察 prefix cache 复用。
4. 统计 wasted slots 和 fragmentation ratio。

## 验证

```bash
python examples/block_usage.py --num-blocks 8 --block-size 4 --prompt-lens 3,6,9 --decode-steps 2
python -m pytest -q tests/test_block_manager.py tests/test_paged_kv_cache.py
python -m course_vllm.benchmarks.grader week10
```

## 交付物

- block table 示例。
- 碎片率统计。
- prefix cache 复用前后对比。
