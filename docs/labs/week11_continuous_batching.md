# Week 11 连续批处理

目标：实现 prefill/decode 调度、chunked prefill 和 preemption 的可观察教学版本。

说明：本周调度是 teaching approximation。它展示 waiting/running 队列、token budget 分块和基础 preemption；生产系统还需要结合 block allocator、SLO、优先级、公平性、prefix cache 和重算/换出策略。

## 当前入口

- `Scheduler(enable_chunked_prefill=True)`
- `Engine.generate_batch(..., enable_chunked_prefill=True)`
- HTTP 服务：`--enable-chunked-prefill`

## 验证

```bash
python -m pytest -q tests/test_scheduler.py tests/test_server_batching.py
python -m course_vllm.benchmarks.grader week11
```

## 说明

当前实现优先保证结构清晰：长 prompt 可以按 token budget 切块推进；`Scheduler.preempt(seq)` 会释放运行态并放回 waiting 队列。报告中需要明确区分课程 demo 和 vLLM/SGLang 级生产调度。

HTTP 服务开启：

```bash
python -m course_vllm.server.api \
  --backend paged \
  --stage week11 \
  --enable-chunked-prefill \
  --max-batch-size 8 \
  --max-batched-tokens 2048
```
