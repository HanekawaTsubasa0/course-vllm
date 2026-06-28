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

## 建议命令

```bash
python -m course_vllm.benchmarks.system_optimization --pinned-memory --transfer-stream
python -m course_vllm.benchmarks.grader week12
```

## 交付物

- baseline vs optimized 的 requests/s、tokens/s、p50/p90/p99、TPOT。
- nsys 时间线摘要。
- admission control 触发样例。
- 对 pinned memory/stream 是否真正改善当前工作负载的判断。
