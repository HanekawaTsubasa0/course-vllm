# Week 01 课程导论与 Baseline Serving

目标：理解 prefill/decode、TTFT/TPOT/tokens/s/SLO，并启动一个可流式输出的最小 LLM serving baseline。

## 代码入口

- `course_vllm/server/api.py`
- `course_vllm/engine/engine.py`
- `examples/offline_generate.py`
- `examples/chat_client.py`

## 实验步骤

1. 启动 HTTP 服务并指定 `--stage week01`。
2. 调用 `/generate` 的 streaming 与 non-streaming 模式。
3. 记录首个 baseline：请求数、输出 tokens、平均延迟、最大延迟。
4. 阅读 `Engine.generate_stream`，标注 prefill 与 decode 分界。

## 验证

```bash
python -m pytest -q tests/test_engine.py tests/test_server_batching.py
python -m course_vllm.benchmarks.grader week01
```

## 交付物

- 一条可复现的启动命令。
- 一条 non-streaming 请求和一条 streaming 请求。
- baseline 指标表。
- prefill/decode 调用链说明。
