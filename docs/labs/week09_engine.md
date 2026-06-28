# Week 09 推理引擎

目标：实现请求生命周期、tokenizer 边界、prefill/decode 主循环、流式输出和单请求 profiler baseline。

## 代码入口

- `course_vllm/engine/engine.py`
- `course_vllm/engine/request.py`
- `course_vllm/server/protocol.py`
- `scripts/profile/torch_profiler.py`

## 实验任务

1. 从 prompt 创建 `Request` 和 `Sequence`。
2. 跟踪 `generate_stream` 中 prefill、sample、decode、stop 的顺序。
3. 验证 stop token、EOS、max_tokens 三种终止条件。
4. 对单请求运行 PyTorch profiler，作为 week10/11 优化前基线。

## TODO(lab09)

- Edit: `course_vllm/engine/request.py` 中 request/sequence 状态字段或状态转换。
- Edit: `course_vllm/engine/engine.py::generate_stream` 主循环。
- Keep boundary: 不改模型 forward 数值逻辑；本周关注请求生命周期和流式事件。

## 验证

```bash
python -m pytest -q tests/test_engine.py tests/test_chat_client.py
python scripts/profile/torch_profiler.py --backend paged --max-tokens 8
python -m course_vllm.benchmarks.grader week09
```

## 交付物

- 请求状态机说明。
- streaming 事件样例。
- 单请求 profiler 报告。
