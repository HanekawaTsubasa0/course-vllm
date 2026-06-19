# Reference Notes

This project uses the existing repositories in `/home/wangqi/llm_serving` as code references.

## nano-vllm

Primary reference for the serving spine:

- `nanovllm/engine/llm_engine.py`: `add_request`, `step`, `generate`.
- `nanovllm/engine/scheduler.py`: prefill/decode scheduling.
- `nanovllm/engine/block_manager.py`: paged KV blocks and hash-based prefix reuse.
- `nanovllm/engine/model_runner.py`: batch preparation and model execution boundary.
- `nanovllm/models/qwen3.py`: compact Qwen3 model structure.

We do not make `flash-attn` mandatory in the base path.

## tiny-llm

Reference for course organization and readable cache abstractions:

- `src/tiny_llm/kv_cache.py`
- `src/tiny_llm_ref/paged_kv_cache.py`
- `src/tiny_llm_ref/batch.py`
- `book/src/week2-01-kv-cache.md`
- `book/src/week3-01-paged-attention-part1.md`
- `book/src/week3-02-paged-attention-part2.md`

The implementation is MLX-based, so the structure is useful but kernels and tensor APIs need PyTorch/CUDA equivalents.

## llm.c

Reference for CUDA kernel exercises:

- `dev/cuda/common.h`: `validate_result`, `benchmark_kernel`, reductions, error checks.
- `dev/cuda/softmax_forward.cu`: stable and online softmax variants.
- `dev/cuda/matmul_forward.cu`: naive, cuBLAS, and handwritten matmul variants.
- `dev/cuda/attention_forward.cu`: CPU reference and multiple attention kernels.
- `dev/cuda/layernorm_forward.cu`: normalization benchmark structure.

The course version should expose these ideas through PyTorch extensions, not standalone C binaries only.

## nanoGPT

Reference for readable transformer code:

- `model.py`: manual causal attention fallback and simple block structure.
- `model.py::generate`: minimal autoregressive generation loop.

## Course Qwen3 Path

- `course_vllm.model.qwen3_torch` is the first course-owned model implementation.
- It starts with an eager full-sequence PyTorch forward pass so RMSNorm, RoPE, grouped-query attention,
  MLP, residuals, and Hugging Face weight loading are explicit.
- `validation/compare_qwen3.py` checks full forward, KV decode, batch prefill, and batch decode against
  Hugging Face eager attention.
- `course_vllm.model.qwen3_backend.Qwen3TorchBackend` stores model KV tensors through
  `course_vllm.engine.kv_cache.ContinuousKVCache`, so the course cache abstraction is now on the
  serving path instead of being only a standalone exercise.
- `course_vllm.engine.paged_kv_cache.PagedKVCache` is the first paged KV data layer: `BlockManager`
  owns block tables and slot mapping, while the cache stores per-layer physical-slot tensors and can
  read a sequence back as dense `[batch, heads, tokens, dim]` KV for correctness tests.
- `course_vllm.model.attention.paged_attention_decode` is a slow PyTorch paged-attention reference:
  it gathers physical slots through each sequence block table, repeats grouped-query KV heads, and
  checks paged attention against dense attention in `tests/test_attention.py`.
- `course_vllm.model.qwen3_backend.Qwen3PagedBackend` puts that paged storage on the real prefill/decode
  path. Its decode path writes each new token KV into physical slots, then reads prior context through
  `paged_attention_decode`; dense readback remains available for debug and correctness checks.

## Scheduling Path

- `course_vllm.engine.scheduler.Scheduler` implements the first waiting/running queue policy with
  prefill priority and decode batches.
- `Engine.generate_batch` now drives multiple requests through that scheduler, and backend prefill is
  executed as one padded model forward. Decode still enters the backend `decode_batch` interface.
  The continuous-cache Qwen3 backend executes same-length decode batches as one model forward; the
  paged backend uses PyTorch paged attention and can decode a batch with mixed context lengths.
- `course_vllm.server.batching.BatchingEngine` connects HTTP requests to `Engine.generate_batch` and
  `Engine.generate_stream` through an async queue plus a dedicated model worker thread.
- `course_vllm.benchmarks.bench_server` is the first HTTP throughput/latency probe. It is intentionally
  simple so batching counters from `/health` can be compared with client-side request rates.

## mini-sglang

Industrial comparison, not a direct base:

- scheduler overlap loop
- radix prefix cache
- multi-process service separation

Those are useful for later readings and optional experiments, but the main project stays single-process and single-GPU initially.
