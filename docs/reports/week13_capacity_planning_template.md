# Week 13 容量规划报告

## 可复现命令

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
  --report | tee profiles/reports/week13_capacity_report.md
```

## 输入参数

- GPU memory: 24 GiB
- utilization: 0.85
- weight memory: 2 GiB
- safety reserve: 1 GiB
- num layers: 28
- num kv heads: 8
- head dim: 128
- dtype: bfloat16
- block size: 16
- max model len: 2048
- target concurrency: 32
- target sequence length: 2048

## 结果

- KV budget: 17.400 GiB
- KV block bytes: 1,835,008
- KV blocks: 10,181
- total token slots: 162,896
- max full-length sequences: 79
- needed token slots: 65,536

## 判断

- 当前单卡满足 target concurrency=32、sequence length=2048 的 KV 容量目标。
- 当前配置不需要为了 KV 容量引入 tensor/pipeline parallelism。
- 如果实际 latency/SLO 不满足，应先看 kernel 吞吐、batching 策略和排队延迟，而不是直接加卡。
- 当目标并发超过 79 条 full-length sequences，或 token slots 超过 162,896 时，需要降低 `max_model_len`/batch size、增加显存或引入多卡。

## 何时需要多卡

- 张量并行：单卡算力或单层 GEMM latency 成为瓶颈时。
- 流水并行：模型层数/权重无法舒适放入单卡，或需要跨设备分层时。
- 更多 KV 容量：主要由 batch size、sequence length、num layers、num kv heads、head dim、dtype 决定。
