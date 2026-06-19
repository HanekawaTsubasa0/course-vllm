# course-vllm

`course-vllm` is the final implementation workspace for the LLM serving course.
The first version is intentionally small: it uses a HuggingFace-backed Qwen3
runner as the stable reference path, while the course serving data structures
are introduced as separate modules with tests.

The development plan is tracked in:

```text
/home/wangqi/llm_serving/docs/course-vllm-development-plan.md
```

## Environment

```bash
cd /home/wangqi/llm_serving/course-vllm
source .venv/bin/activate
```

The uv environment pins the CUDA-12-compatible stack verified on this server:

- `torch==2.8.0`
- `triton==3.4.0`
- `transformers>=4.57,<4.58`

## Run Tests

```bash
pytest -q
```

## Offline Generate

```bash
python examples/offline_generate.py \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --chat \
  --prompt "用一句话介绍你自己。" \
  --max-tokens 32 \
  --temperature 0
```

## HTTP Server

```bash
python -m course_vllm.server.api \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --port 18080
```

Non-streaming generation:

```bash
curl -s -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":false,"sampling_params":{"temperature":0,"max_tokens":16}}'
```

Streaming chat:

```bash
curl -s -N -X POST http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话介绍你自己。"}],"stream":true,"sampling_params":{"temperature":0,"max_tokens":24}}'
```

Interactive client:

```bash
python examples/chat_client.py --url http://127.0.0.1:18080/v1/chat/completions
```

## Implemented So Far

- HuggingFace-backed model runner with explicit prefill/decode loop.
- Greedy and temperature sampler.
- Offline generate example.
- FastAPI server with `/health`, `/generate`, and `/v1/chat/completions`.
- SSE-style streaming responses.
- Separate HTTP chat client.
- Continuous KV cache skeleton and tests.
- Paged-KV-style block manager skeleton and tests.
- Single-process prefill/decode scheduler skeleton and tests.
- Reference notes for `nano-vllm`, `tiny-llm`, `llm.c`, `nanoGPT`, and `mini-sglang`.

## Next Work

- Replace the HuggingFace runner with a course-owned Qwen3 model path.
- Connect continuous KV cache to the model runner.
- Connect `BlockManager` to paged KV metadata and slot mapping.
- Add continuous batching to the HTTP serving path.
- Add CUDA kernel harness and first kernels.
