# Week 12 系统优化

目标：加入 pinned memory、CUDA stream、异步拷贝/执行重叠实验和请求准入控制，并提交优化前后报告。

## 代码入口

- `course_vllm/model/qwen3_continuous_backend.py`
- `course_vllm/server/batching.py`
- `course_vllm/benchmarks/system_optimization.py`
- `docs/reports/week12_system_optimization_template.md`

## 实验任务

1. 用 baseline 参数启动服务，记录 HTTP benchmark。
2. 打开 `--pinned-memory` 和 `--transfer-stream`，再次压测。
3. 设置 `--max-queue-size` 和 `--max-prompt-chars`，验证请求准入。
4. 用 nsys 观察 memcpy、kernel 和同步点。
5. 如果短 prompt、短 decode、小并发下优化收益不明显，需要解释 workload 原因；可补充长 prompt 或更高并发实验观察优化开关更可能生效的场景。

## TODO(lab12)

- Edit: `course_vllm/model/qwen3_continuous_backend.py` 中 pinned memory / transfer stream 辅助路径。
- Edit: `course_vllm/server/batching.py` 中 `max_queue_size`、`max_prompt_chars` 请求准入。
- Edit report: `docs/reports/week12_system_optimization_template.md`，必须解释 workload 与优化效果的关系。

## 建议命令

```bash
python -m course_vllm.benchmarks.system_optimization --pinned-memory --transfer-stream
python -m course_vllm.benchmarks.grader week12
```

更容易观察差异的长负载压测建议使用相同 workload 分别打 baseline 和 optimized 服务。

Baseline 服务使用 Week 11 配置：

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week11 \
  --kernel-impl auto \
  --dtype bfloat16 \
  --max-batch-size 8 \
  --batch-wait-ms 2 \
  --max-queue-size 128 \
  --max-prompt-chars 32768 \
  --port 18081
```

Optimized 服务打开 Week 12 开关：

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week12 \
  --kernel-impl auto \
  --dtype bfloat16 \
  --max-batch-size 8 \
  --batch-wait-ms 2 \
  --max-queue-size 128 \
  --max-prompt-chars 32768 \
  --enable-chunked-prefill \
  --cache-aware-scheduling \
  --pinned-memory \
  --transfer-stream \
  --port 18082
```

分别压测：

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18081/generate \
  --num-requests 64 \
  --concurrency 8 \
  --prompt "Summarize the LLM serving pipeline, including prefill, decode, KV cache, scheduling, profiling, and admission control." \
  --max-tokens 64 \
  --json

python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18082/generate \
  --num-requests 64 \
  --concurrency 8 \
  --prompt "Summarize the LLM serving pipeline, including prefill, decode, KV cache, scheduling, profiling, and admission control." \
  --max-tokens 64 \
  --json
```

## 交付物

- baseline vs optimized 的 requests/s、tokens/s、p50/p90/p99、TPOT。
- nsys 时间线摘要。
- admission control 触发样例。
- 对 pinned memory/stream 是否真正改善当前工作负载的判断。
