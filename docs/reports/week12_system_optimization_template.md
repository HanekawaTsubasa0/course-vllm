# Week 12 系统优化报告

## 优化项

- pinned memory: enabled in optimized run
- CUDA stream: enabled in optimized run
- non_blocking copy: enabled when moving CPU token tensors to CUDA
- overlap: transfer stream path exists; current implementation waits before consuming moved tensor
- admission control: `--max-queue-size 64`, `--max-prompt-chars 8192`
- scheduler options: `--enable-chunked-prefill`, `--cache-aware-scheduling`

## 可复现命令

Baseline:

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend paged \
  --stage week11 \
  --kernel-impl auto \
  --dtype bfloat16 \
  --max-batch-size 4 \
  --batch-wait-ms 2 \
  --max-queue-size 64 \
  --max-prompt-chars 8192 \
  --port 18081
```

另一个终端：

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18081/generate \
  --num-requests 8 \
  --concurrency 2 \
  --max-tokens 8 \
  --json | tee profiles/reports/bench_baseline_week11.json
```

Optimized:

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend paged \
  --stage week12 \
  --kernel-impl auto \
  --dtype bfloat16 \
  --max-batch-size 4 \
  --batch-wait-ms 2 \
  --max-queue-size 64 \
  --max-prompt-chars 8192 \
  --enable-chunked-prefill \
  --cache-aware-scheduling \
  --pinned-memory \
  --transfer-stream \
  --port 18082
```

另一个终端：

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18082/generate \
  --num-requests 8 \
  --concurrency 2 \
  --max-tokens 8 \
  --json | tee profiles/reports/bench_optimized_week12.json
```

## 对比指标

| 配置 | requests/s | tokens/s | p50 | p90 | p99 | estimated TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before | 4.038430 | 32.307436 | 0.231115 | 1.245282 | 1.248274 | 0.060612 |
| after | 3.981867 | 31.854933 | 0.251096 | 1.216504 | 1.219466 | 0.061504 |

## 解释

这个负载是短 prompt、短 decode、低并发，优化前后差异主要落在运行波动范围内。after 的 p90/p99 略好，但吞吐和 p50 略低。结论不能写成“系统优化显著提升”，应写成“机制已接入，当前小负载收益不明显，需要长 prompt/高并发复测”。

## Admission Control 证据

```bash
python -m course_vllm.benchmarks.system_optimization \
  --pinned-memory \
  --transfer-stream \
  --max-queue-size 64 \
  --max-prompt-chars 8192 \
  | tee profiles/reports/week12_system_optimization_plan.json
```

输出包含：

- accepted request: queue_depth=0, prompt_chars=32
- rejected request: queue_depth=64, prompt_chars=8193, reason=`prompt_too_long`
- overlap plan: `pinned_non_blocking` + `dedicated_transfer_stream`

## nsys 证据

- 产物：`profiles/nsys_server_ready.nsys-rep`
- benchmark during nsys: requests/s=1.430951, output_tokens/s=11.447604, p99=1.710348
- 当前环境 CPU sampling 受限，但 CUDA timeline 文件已生成。

## 风险和取舍

- 正确性风险：pinned/stream 只改变 tensor transfer path，不改变 logits 计算；已通过 Qwen3/HF float32 对齐。
- 稳定性风险：系统优化开关依赖 CUDA device；CPU 环境自动回退普通 `.to(device)`。
- 对长 prompt 的影响：chunked prefill 会改善 token budget 控制，但可能提高调度复杂度。
- 对高并发的影响：admission control 能限制队列爆炸，但参数过严会降低吞吐。
