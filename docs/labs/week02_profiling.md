# Week 02 性能分析

目标：建立服务性能基线，能用 nsys/ncu/benchmark 说明瓶颈在哪里。

## 需要改或运行的文件

- `course_vllm/benchmarks/bench_server.py`
- `scripts/profile/nsys_server.sh`
- `scripts/profile/ncu_kernel.sh`
- `scripts/profile/torch_profiler.py`
- `docs/reports/week02_profile_template.md`

## 建议命令

```bash
python -m course_vllm.server.api --backend hf --stage week02 --port 18080
python -m course_vllm.benchmarks.bench_server --url http://127.0.0.1:18080/generate --num-requests 16 --concurrency 4 --max-tokens 16 --json
```

有 CUDA profiler 环境时运行：

```bash
bash scripts/profile/nsys_server.sh
bash scripts/profile/ncu_kernel.sh
```

## 交付物

- baseline 指标：requests/s、tokens/s、p50/p90/p99 latency、估算 TPOT。
- nsys 时间线截图或文字摘要。
- ncu 中至少一个 kernel 的吞吐、访存或 occupancy 摘要。
- 一个瓶颈判断：Python 调度、GPU kernel、数据传输、同步等待中哪个最明显。
