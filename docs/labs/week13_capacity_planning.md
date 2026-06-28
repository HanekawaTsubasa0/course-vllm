# Week 13 多卡推理与容量规划

目标：在单卡服务上压测并产出容量规划，判断何时需要张量并行、流水并行或更多 KV cache 容量。

## 代码入口

- `course_vllm/benchmarks/capacity_planner.py`
- `course_vllm/benchmarks/bench_server.py`
- `docs/reports/week13_capacity_planning_template.md`

## 实验任务

1. 估算模型权重、KV cache、预留显存。
2. 计算 block bytes、KV blocks、token slots、full-length sequence 上限。
3. 结合目标 concurrency 和 sequence length 判断单卡是否足够。
4. 如果单卡不足，说明应优先增加 KV 容量、张量并行还是流水并行。

## TODO(lab13)

- Edit/report: `docs/reports/week13_capacity_planning_template.md`。
- Run/inspect: `course_vllm/benchmarks/capacity_planner.py`，理解 KV block、token slots、并发上限计算。
- Optional edit: 只允许扩展 planner 参数或报告字段，不改 serving 主路径。

## 建议命令

```bash
python -m course_vllm.benchmarks.capacity_planner \
  --gpu-memory-gb 24 \
  --weight-memory-gb 2 \
  --num-layers 28 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --block-size 16 \
  --max-model-len 2048 \
  --target-concurrency 32 \
  --target-sequence-len 2048 \
  --report
```

## 验证

```bash
python -m course_vllm.benchmarks.grader week13
```

## 交付物

- 容量规划报告。
- 单卡瓶颈判断。
- 多卡触发条件说明。
