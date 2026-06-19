# course-vllm

`course-vllm` 是一个面向 LLM serving 课程的小型单卡推理服务工程。它的目标不是复刻工业级 vLLM/sglang，而是在一套可运行、可测试的代码里讲清楚：

- prefill / decode
- KV cache
- paged KV cache 和 block table
- batch prefill / batch decode
- continuous batching
- HTTP serving 和 streaming
- CUDA extension harness
- RMSNorm、RoPE、softmax、matmul、paged attention decode 等教学 CUDA kernel

主工程路径：

```text
/home/wangqi/llm_serving/course-vllm
```

开发计划文档：

```text
/home/wangqi/llm_serving/docs/course-vllm-development-plan.md
```

## 环境配置

进入项目并激活已有虚拟环境：

```bash
cd /home/wangqi/llm_serving/course-vllm
source .venv/bin/activate
```

当前环境主要依赖：

```text
torch==2.8.0
transformers>=4.57,<4.58
fastapi
uvicorn
httpx
pydantic
safetensors
pytest
ninja
```

模型默认使用本机路径：

```text
/home/wangqi/huggingface/Qwen3-0.6B
```

### CUDA 编译依赖

CUDA tests 会通过 PyTorch extension JIT 编译 `kernels/*.cu`。这台机器是 CUDA 12.8，但系统默认 `g++-15` 对 nvcc 的半精度/bfloat16 头文件太新，所以项目把 GCC 14 解包到本地 `dependence/`，不安装到系统：

```bash
mkdir -p dependence/debs dependence/gcc14-root
cd dependence/debs
apt download g++-14-x86-64-linux-gnu gcc-14-x86-64-linux-gnu cpp-14-x86-64-linux-gnu
apt download libgcc-14-dev libstdc++-14-dev gcc-14-base
cd ../..
for deb in dependence/debs/*.deb; do dpkg-deb -x "$deb" dependence/gcc14-root; done
```

`course_vllm.kernels.harness` 会自动优先使用：

```text
dependence/gcc14-root/usr/bin/x86_64-linux-gnu-g++-14
```

`dependence/` 已写入 `.gitignore`，不会进入 git。

## 快速运行

### 离线单请求生成

```bash
python examples/offline_generate.py \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --chat \
  --prompt "用一句话介绍你自己。" \
  --temperature 0
```

如果要限制生成长度：

```bash
python examples/offline_generate.py \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --chat \
  --prompt "用一句话介绍你自己。" \
  --max-tokens 128 \
  --temperature 0
```

默认 `max_tokens=None`，也就是不按 token 数截断，只在 EOS 或 stop token 时停止。如果模型一直不输出 EOS，请求会继续生成；演示或压测时建议显式设置 `--max-tokens`。

### 离线 batch 生成

```bash
python examples/offline_generate.py \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --prompts "Hello|What is KV cache?" \
  --max-tokens 64 \
  --temperature 0
```

### Paged KV 调试

```bash
python examples/block_usage.py \
  --num-blocks 8 \
  --block-size 4 \
  --prompt-lens 3,6,9 \
  --decode-steps 2
```

## 启动 HTTP 服务

在第一个终端启动服务：

```bash
cd /home/wangqi/llm_serving/course-vllm
source .venv/bin/activate

python -m course_vllm.server.api \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --dtype bfloat16 \
  --max-batch-size 8 \
  --batch-wait-ms 2 \
  --port 18080
```

健康检查：

```bash
curl -s http://127.0.0.1:18080/health
```

`/health` 会返回模型、backend、batching 统计信息，包括 total requests、total batches、平均 batch size 和 queue depth。

### `/generate`

非 streaming：

```bash
curl -s -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":false,"sampling_params":{"temperature":0,"max_tokens":64}}'
```

无显式 token 上限：

```bash
curl -s -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":false,"sampling_params":{"temperature":0,"max_tokens":null}}'
```

### `/v1/chat/completions`

Streaming chat：

```bash
curl -s -N -X POST http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话介绍你自己。"}],"stream":true,"sampling_params":{"temperature":0,"max_tokens":128}}'
```

## 命令行客户端

在另一个终端启动交互式 CLI：

```bash
cd /home/wangqi/llm_serving/course-vllm
source .venv/bin/activate

python examples/chat_client.py
```

一次性调用：

```bash
python examples/chat_client.py \
  --once "用一句话介绍你自己。" \
  --max-tokens 128 \
  --temperature 0
```

指定服务地址：

```bash
python examples/chat_client.py \
  --url http://127.0.0.1:18080/v1/chat/completions
```

CLI 支持的交互命令：

```text
/help
/health
/params
/set max_tokens 512
/set max_tokens none
/set temperature 0
/set top_k 40
/set top_k none
/stream on
/stream off
/system <text>
/history [n]
/clear
/save [path]
/load <path>
/exit
```

多个终端可以同时运行 `examples/chat_client.py` 连接同一个服务端：

- non-streaming 请求会进入 server batching queue，采样参数一致时可以合批。
- streaming 请求也可以从多个客户端同时进入服务，但当前教学实现只有一个 model worker，模型执行会串行化。
- 这是单进程单卡教学版，不是工业级多 worker / 多 GPU serving。

## Benchmark

服务启动后可以跑 HTTP 压测：

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18080/generate \
  --num-requests 16 \
  --concurrency 4 \
  --max-tokens 16
```

## 正确性验证

### 单元测试

普通环境：

```bash
pytest -q -rs
```

如果当前 shell 看不到 CUDA，CUDA 测试会 skip。之前在 CUDA 可见环境已验证过：

```text
pytest -q tests/test_kernels.py tests/test_attention.py -rs
  -> 12 passed
```

当前新增 CLI 和默认无上限改动后，普通沙箱下：

```text
44 passed, 9 skipped
```

### Qwen3 / HuggingFace 对齐

```bash
python validation/compare_qwen3.py forward \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --dtype float32

python validation/compare_qwen3.py decode \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --dtype float32

python validation/compare_qwen3.py batch-prefill \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --dtype float32

python validation/compare_qwen3.py batch-decode \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --dtype float32
```

已验证过的典型结果：

```text
forward: course vs HF max_abs_diff=0
float32 paged batch decode vs HF batch decode: max_abs_diff=0.000012
bfloat16 paged batch decode vs HF batch decode: max_abs_diff=0.500000
HF bfloat16 batch decode vs HF single decode 同量级: max_abs_diff=0.562500
```

因此 float32 路径已和 HF 对齐；bf16 的 batch/single 差异主要来自浮点执行路径差异，不是 paged attention 的逻辑错误。

## 代码结构

```text
course_vllm/
  engine/
    engine.py           推理主循环，generate / generate_batch / generate_stream
    request.py          Request、Sequence、生成状态和 max_tokens 停止条件
    scheduler.py        单进程 prefill/decode scheduler
    sampler.py          greedy、temperature、top-k sampling
    kv_cache.py         连续 KV cache
    paged_kv_cache.py   paged KV cache，物理 slot 写入和读取
    block_manager.py    block 分配、释放、block table
  model/
    hf_backend.py       HuggingFace 参考 backend
    qwen3_torch.py      课程自有 Qwen3 PyTorch 模型实现
    qwen3_backend.py    course / paged backend，prefill/decode/batch decode
    attention.py        dense attention、paged attention reference、CUDA dispatch
    types.py            ModelOutput / BatchModelOutput
  server/
    api.py              FastAPI app，/health、/generate、/v1/chat/completions
    batching.py         HTTP batching queue 和 model worker thread
    protocol.py         Pydantic request/response schema
  kernels/
    harness.py          PyTorch CUDA extension JIT 加载和 benchmark helper
    cuda_ops.py         Python wrapper，调用 CUDA kernels
  benchmarks/
    bench_server.py     HTTP 并发压测

kernels/
  vector_add.cu         最小 CUDA extension 示例
  course_ops.cpp        PyTorch binding 和 tensor 参数检查
  course_ops.cu         softmax、RMSNorm、RoPE、matmul、paged attention decode CUDA kernels

examples/
  offline_generate.py   离线单请求 / batch generate
  chat_client.py        交互式 HTTP chat CLI
  block_usage.py        paged KV block 使用情况演示

validation/
  compare_qwen3.py      和 HuggingFace Qwen3 做 logits 对齐

tests/
  test_*.py             KV、paged KV、scheduler、engine、server batching、CUDA kernel、CLI 测试

docs/
  reference_notes.md    参考 nano-vllm、mini-sglang、llm.c、nanoGPT、tiny-llm 的阅读笔记
```

## 功能说明

### Backend

工程有三个 backend：

```text
hf      使用 HuggingFace 模型作为稳定参考路径
course  使用课程自有 Qwen3 PyTorch 实现和连续 KV cache
paged   使用课程自有 Qwen3 PyTorch 实现和 paged KV cache
```

启动服务时通过 `--backend` 选择：

```bash
--backend hf
--backend course
--backend paged
```

### Prefill / Decode

`Engine.generate_stream` 会先对 prompt 做 prefill，得到第一步 logits 和 KV cache；之后每次 decode 只输入最新 token，并复用之前的 KV cache。这样代码能直接展示 LLM 推理中 prefill 和 decode 的区别。

### KV Cache

`course_vllm/engine/kv_cache.py` 实现连续 KV cache，用于教学中最直观的 KV 存储方式。`course` backend 使用这种缓存。

### Paged KV Cache

`course_vllm/engine/block_manager.py` 和 `course_vllm/engine/paged_kv_cache.py` 实现 paged KV：

- KV cache 被切成固定大小 block。
- 每个 sequence 持有 block table。
- block table 把逻辑 token 位置映射到物理 KV slot。
- 请求结束后释放 block。

`paged` backend 会把每层 K/V 写进物理 slot，并在 decode 阶段通过 block table 读取历史上下文。

### Continuous Batching

`course_vllm/engine/scheduler.py` 是离线 batch 的教学 scheduler。它维护 waiting/running 序列，区分 prefill batch 和 decode batch，并受 `max_num_seqs`、`max_num_batched_tokens` 限制。

HTTP 服务层还有 `course_vllm/server/batching.py`：

- non-streaming HTTP 请求先进队列；
- 在 `batch_wait_ms` 时间内收集相同 sampling 参数的请求；
- 调用 `Engine.generate_batch`；
- 模型执行放到单独 worker thread，避免阻塞 FastAPI event loop。

### Streaming

`/generate` 和 `/v1/chat/completions` 都支持 `stream=true`。服务端用 SSE 风格返回：

```text
data: {"event":"token","text":"...","token_id":...}
data: {"event":"finished","finish_reason":"..."}
data: [DONE]
```

### Sampling

`SamplingParams` 支持：

```text
temperature
top_k
seed
max_tokens
stop_token_ids
```

`temperature=0` 表示 greedy。默认 `max_tokens=None`，不按 token 数截断。

### CUDA Kernels

CUDA 代码在 `kernels/course_ops.cu`：

- row-wise softmax
- RMSNorm
- Qwen-style RoPE
- naive matmul
- paged attention decode

`kernels/course_ops.cpp` 做 PyTorch binding、tensor shape/dtype/device 检查和 kernel launcher 调用。

`course_vllm/kernels/cuda_ops.py` 提供 Python 包装函数：

```python
cuda_softmax
cuda_rms_norm
cuda_rope
cuda_matmul
cuda_paged_attention_decode
```

paged attention decode 在 CUDA 可用且满足限制时走手写 CUDA kernel；否则保留 PyTorch reference 作为正确性 oracle。

### 当前限制

- 单进程、单卡。
- 不是工业级多 worker / 多 GPU serving。
- streaming 请求当前通过单 model worker 串行执行。
- CUDA kernels 是教学实现，重点是正确性和可读性，不追求极限性能。
- profiler、capacity planning、course mapping、多卡讲解和 AscendC 文档还未整理完成。

## 参考项目

本项目参考这些开源工程的设计思想和课程材料：

- `nano-vllm`
- `mini-sglang`
- `llm.c`
- `nanoGPT`
- `tiny-llm`

本地参考笔记在：

```text
docs/reference_notes.md
```
