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

## Model Alignment Checks

```bash
python validation/compare_qwen3.py forward --model /home/wangqi/huggingface/Qwen3-0.6B
python validation/compare_qwen3.py decode --model /home/wangqi/huggingface/Qwen3-0.6B --backend paged
python validation/compare_qwen3.py batch-prefill --model /home/wangqi/huggingface/Qwen3-0.6B --backend paged
python validation/compare_qwen3.py batch-decode --model /home/wangqi/huggingface/Qwen3-0.6B --backend paged
```

The comparison scripts default to `float32` and Hugging Face eager attention so
the course implementation can be checked without fused-kernel numeric drift.
Hugging Face is the numeric reference for tokenizer behavior, full-sequence
logits, and KV-cache decode logits.

## Offline Generate

```bash
python examples/offline_generate.py \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --chat \
  --prompt "用一句话介绍你自己。" \
  --max-tokens 32 \
  --temperature 0
```

## Paged KV Debug

```bash
python examples/block_usage.py --num-blocks 8 --block-size 4 --prompt-lens 3,6,9 --decode-steps 2
```

## Offline Batch Generate

```bash
python examples/offline_generate.py \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --prompts "Hello|What is KV cache?" \
  --max-tokens 16 \
  --temperature 0
```

## HTTP Server

```bash
python -m course_vllm.server.api \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --max-batch-size 8 \
  --batch-wait-ms 2 \
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

Server benchmark:

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18080/generate \
  --num-requests 16 \
  --concurrency 4 \
  --max-tokens 16
```

## Implemented So Far

- HuggingFace-backed model runner with explicit prefill/decode loop.
- Course-owned Qwen3 PyTorch runner with explicit prefill/decode KV cache.
- Reusable `ContinuousKVCache` connected to the course Qwen3 backend.
- Paged KV physical-slot storage with block tables and dense readback tests.
- PyTorch `paged_attention_decode` reference that reads directly from paged physical slots and matches dense attention.
- `paged` backend decode path writes new KV to physical slots and reads prior context through block-table attention.
- Greedy and temperature sampler.
- Offline generate example.
- Offline `generate_batch` path driven by the prefill/decode scheduler.
- Mixed-length prefill requests share one padded model forward.
- Continuous-cache Qwen3 backend supports true same-length batched decode.
- Paged backend supports `decode_batch` through PyTorch paged attention, including mixed context lengths.
- FastAPI server with `/health`, `/generate`, and `/v1/chat/completions`.
- HTTP generation enters `BatchingEngine`; model work runs on a dedicated worker thread.
- Streaming responses also run through `BatchingEngine.stream`.
- `/health` reports batching counters such as total batches and observed batch size.
- SSE-style streaming responses.
- Separate HTTP chat client.
- Minimal CUDA kernel harness with a `vector_add` kernel exercise.
- Triton paged-attention decode kernel for CUDA tensors, with the PyTorch reference kept as the CPU/fallback oracle.
- Teaching Triton kernels for RMSNorm, RoPE, softmax, and matmul, each covered by CUDA-gated correctness tests.
- Continuous KV cache skeleton and tests.
- Paged-KV-style block manager skeleton and tests.
- Single-process prefill/decode scheduler skeleton and tests.
- Reference notes for `nano-vllm`, `tiny-llm`, `llm.c`, `nanoGPT`, and `mini-sglang`.

## Next Work

- Run the Triton kernel tests on a CUDA-visible machine and record benchmark numbers for paged attention and the teaching kernels.
- Add a short profiler walkthrough for HTTP batching latency and decode throughput.
