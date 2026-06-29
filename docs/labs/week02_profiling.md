# Week 02 性能分析

目标：建立服务性能基线，能用 nsys/ncu/benchmark 说明瓶颈在哪里。

## 需要改或运行的文件

- `course_vllm/benchmarks/bench_server.py`
- `scripts/profile/nsys_server.sh`
- `scripts/profile/ncu_kernel.sh`
- `scripts/profile/torch_profiler.py`
- `docs/reports/week02_profile_template.md`

## TODO(lab02)

- Edit report: `docs/reports/week02_profile_template.md`。
- Run/inspect: `course_vllm/benchmarks/bench_server.py`、`scripts/profile/torch_profiler.py`、`scripts/profile/nsys_server.sh`。
- Optional edit: 只允许给 benchmark 输出增加字段或格式化，不改模型/engine 主路径。

## 建议命令

```bash
python -m course_vllm.server.api --backend reference --stage week02 --port 18080
python -m course_vllm.benchmarks.bench_server --url http://127.0.0.1:18080/generate --num-requests 16 --concurrency 4 --max-tokens 16 --json
```

有 CUDA profiler 环境时运行：

```bash
bash scripts/profile/nsys_server.sh
KERNEL_SCENARIO=paged_attention bash scripts/profile/ncu_kernel.sh
KERNEL_SCENARIO=matmul KERNEL_NAME=matmul bash scripts/profile/ncu_kernel.sh
python scripts/profile/torch_profiler.py --backend course --kv-mode paged --workload mixed --warmup 1 --repeat 3
```

## 交付物

- baseline 指标：requests/s、tokens/s、p50/p90/p99 latency、估算 TPOT。
- nsys 时间线截图或文字摘要。
- ncu 中至少一个 kernel 的吞吐、访存或 occupancy 摘要。
- 一个瓶颈判断：Python 调度、GPU kernel、数据传输、同步等待中哪个最明显。
