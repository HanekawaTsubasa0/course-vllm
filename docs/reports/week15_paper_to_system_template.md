# Week 15 Paper-to-System 映射

## 机制

- 名称：cache-aware serving
- 核心问题：多个请求存在共享 prefix 时，如果调度顺序完全无感知，prefix cache 命中和局部性较差。
- 核心机制：把共享 prefix 更长的请求排得更近，提高复用机会。

## 可复现命令

```bash
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "cache-aware serving" \
  --prompts "1,2,3,4|1,2,3,9|8,7|1,2,5" \
  | tee profiles/reports/week15_cache_aware_demo.json
```

## 本次结果

- baseline order: `[0, 1, 2, 3]`
- cache-aware order: `[0, 1, 3, 2]`
- baseline shared-prefix score: 3
- cache-aware shared-prefix score: 5

## 对应系统模块

- `engine/policies.py`: `cache_aware_order`, `score_order`, `paper_to_system_map`
- `engine/block_manager.py`: block hash table 和 prefix cache reuse
- `engine/engine.py`: `generate_batch(..., cache_aware_scheduling=True)`
- `server/batching.py`: HTTP 参数 `--cache-aware-scheduling`

## 需要修改的数据结构

- prompt token ids: 作为调度排序输入
- block hash table: 记录完整 block prefix hash
- scheduler waiting queue: 决定请求进入 prefill 的顺序
- batching metadata: 暴露 cache-aware scheduling 开关

## 预期影响的指标

- `prefix_cached_blocks`
- TTFT
- requests/s
- KV fragmentation

## 实验设计

- baseline: FIFO 顺序。
- 改造版本：cache-aware greedy ordering。
- 输入负载：包含共享 prefix 的 token id 序列。
- 成功标准：shared-prefix score 不低于 baseline；真实服务中观察 prefix_cached_blocks、TTFT、吞吐变化。

## 从 Demo 推进到主路径

当前 demo 已验证排序分数，主路径已支持 `Engine.generate_batch(..., cache_aware_scheduling=True)` 和 HTTP `--cache-aware-scheduling`。下一步应在真实 prompt workload 下记录 `PagedKVCache.usage_stats()`，把 prefix reuse 和 serving latency 联动起来。
