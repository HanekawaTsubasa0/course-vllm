# course-llm-serving

`course-llm-serving` 是一个面向 LLM serving 课程的小型推理服务工程。当前 Python 包名仍保留为 `course_vllm`，以减少既有脚本和测试的迁移成本；课程文档中统一把它称为 course LLM serving 工程。它的目标不是复刻工业级 vLLM/sglang，而是在一套可运行、可测试的代码里讲清楚：

- prefill / decode
- KV cache
- paged KV cache 和 block table
- batch prefill / batch decode
- continuous batching
- HTTP serving 和 streaming
- CUDA extension harness
- RMSNorm、RoPE、softmax、matmul、paged attention decode 等教学 CUDA kernel

主工程路径就是当前仓库根目录。

## 教学进度开关

工程显式暴露课程阶段和算子实现开关：

```bash
--backend reference|course
--kv-mode dense|paged
--stage week04
--kernel-impl torch|auto|cuda
```

- `backend=reference` 使用 HuggingFace/PyTorch 作为正确性 oracle，只用于对齐和评测参考。
- `backend=course` 使用课程主线 engine；连续 KV 与 paged KV 通过 `kv-mode` 选择。
- `kv-mode=dense` 使用连续 KV cache，便于先理解 append/fetch。
- `kv-mode=paged` 使用 block table 和 paged KV cache，是后半学期 serving 主线。
- `stage` 用来标记当前实验周次，`/health` 会返回对应周次、主题、代码状态和测试提示。
- `kernel-impl=torch` 保持纯 PyTorch/reference 路径。
- `kernel-impl=auto` 在 CUDA tensor 上优先尝试课程 CUDA kernel，失败时回退 PyTorch。
- `kernel-impl=cuda` 要求走课程 CUDA kernel，kernel 不可用时直接报错，适合检查是否真的接入。

周次说明在 `docs/labs/`，性能分析和系统报告模板在 `docs/reports/`。

完整课程使用文档：

- `docs/project_usage_guide.md`：项目怎么安装、运行、验证、压测、生成学生分支。
- `docs/code_walkthrough.md`：按模块讲清楚所有核心代码、调用链和设计取舍。
- `docs/course_code_teaching_guide.md`：教师按周讲代码、实验位置和运行方式时使用。
- `docs/teaching/README.md`：按周学习讲义，覆盖核心概念、机制、术语和本工程对应实现。
- `docs/course_teaching_runbook.md`：教师/助教逐周上课运行手册。
- `docs/runnable_validation_guide.md`：正确性、profiling、benchmark、报告产物的可复现验收指南。

## 环境配置

进入项目并创建虚拟环境：

```bash
cd course-vllm
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
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

脚本和服务默认使用公开模型 ID：

```text
Qwen/Qwen3-0.6B
```

如果本机已经有模型目录，直接用 `--model` 参数覆盖：

```bash
--model /path/to/Qwen3-0.6B
```

### CUDA 编译依赖

CUDA tests 会通过 PyTorch extension JIT 编译 `kernels/*.cu`。运行 CUDA kernel 验收需要 NVIDIA GPU、CUDA toolkit、`ninja`，以及与当前 `nvcc` 兼容的 host C++ compiler。

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
```

如果遇到 nvcc 和系统 G++ 版本不兼容、CUDA extension 编译失败或 profiler 权限问题，按 `docs/runnable_validation_guide.md` 的 troubleshooting 处理。

## 快速运行

### 离线单请求生成

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week04 \
  --kernel-impl auto \
  --chat \
  --prompt "用一句话介绍你自己。" \
  --temperature 0
```

如果要限制生成长度：

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week04 \
  --kernel-impl auto \
  --chat \
  --prompt "用一句话介绍你自己。" \
  --max-tokens 128 \
  --temperature 0
```

默认 `max_tokens=None`，也就是不按 token 数截断，只在 EOS 或 stop token 时停止。如果模型一直不输出 EOS，请求会继续生成；演示或压测时建议显式设置 `--max-tokens`。

### 离线 batch 生成

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week04 \
  --kernel-impl auto \
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
cd course-vllm
source .venv/bin/activate

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
  --max-prompt-chars 8192 \
  --enable-chunked-prefill \
  --cache-aware-scheduling \
  --pinned-memory \
  --transfer-stream \
  --port 18080
```

健康检查：

```bash
curl -s http://127.0.0.1:18080/health
```

`/health` 会返回模型、backend、课程 stage、kernel 实现、batching 统计信息，包括 total requests、total batches、平均 batch size、queue depth、chunked prefill 和 cache-aware scheduling 开关。

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
cd course-vllm
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
  --max-tokens 16 \
  --json
```

该 benchmark 输出 `requests/s`、`output_tokens/s`、`latency p50/p90/p99` 和估算 `TPOT`。

容量规划工具：

```bash
python -m course_vllm.benchmarks.capacity_planner \
  --gpu-memory-gb 24 \
  --weight-memory-gb 2 \
  --num-layers 28 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --block-size 16 \
  --max-model-len 2048
```

生成第十三节容量规划报告：

```bash
python -m course_vllm.benchmarks.capacity_planner \
  --gpu-memory-gb 24 \
  --weight-memory-gb 2 \
  --num-layers 28 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --hidden-size 1024 \
  --intermediate-size 2816 \
  --tp 2 \
  --pp 2 \
  --cp 2 \
  --target-concurrency 32 \
  --target-sequence-len 2048 \
  --report
```

报告会同时输出 KV 容量、TP all-reduce、PP bubble、EP all-to-all、CP pass-Q/pass-KV 等字段，用于判断容量瓶颈、计算瓶颈和通信瓶颈。

性能分析脚本：

```bash
bash scripts/profile/nsys_server.sh
KERNEL_SCENARIO=paged_attention bash scripts/profile/ncu_kernel.sh
KERNEL_SCENARIO=matmul KERNEL_NAME=matmul bash scripts/profile/ncu_kernel.sh
python scripts/profile/torch_profiler.py --backend course \
  --kv-mode paged --workload mixed --warmup 1 --repeat 3 --max-tokens 8
python -m course_vllm.benchmarks.system_optimization --pinned-memory --transfer-stream
```

按周自动检查：

```bash
python -m course_vllm.benchmarks.grader week05
python -m course_vllm.benchmarks.grader week07
python -m course_vllm.benchmarks.grader week11
python -m course_vllm.benchmarks.grader week12
python -m course_vllm.benchmarks.grader week13
python -m course_vllm.benchmarks.grader week15
```

第十五节 cache-aware serving 最小复现实验：

```bash
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "cache-aware serving" \
  --prompts "1,2,3,4|1,2,3,9|8,7|1,2,5"
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "prefill-decode disaggregation" \
  --requests "128:16|2048:8|256:64|1024:12"
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "tokendance-style scheduling" \
  --requests "128:32|2048:4|256:16"
```

## 正确性验证

### 单元测试

普通环境：

```bash
pytest -q -rs
```

CUDA kernel 接入验收单独运行：

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
```

普通 `pytest -q -rs` 适合基础回归；如果当前 shell 看不到 CUDA，CUDA 相关测试会 skip。

### Qwen3 / HuggingFace 对齐

```bash
python validation/compare_qwen3.py forward \
  --model Qwen/Qwen3-0.6B \
  --dtype float32

python validation/compare_qwen3.py decode \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --dtype float32

python validation/compare_qwen3.py batch-prefill \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --dtype float32

python validation/compare_qwen3.py batch-decode \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
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
    qwen3_continuous_backend.py  dense KV runner，连续 KV cache 的 prefill/decode/batch decode
    qwen3_paged_backend.py       paged KV runner，paged KV cache 的 prefill/decode/batch decode
    qwen3_backend.py             兼容导出 Qwen3TorchBackend / Qwen3PagedBackend
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
  file_guide.md         每个文件的职责说明
```

更细的逐文件说明见：

```text
docs/file_guide.md
```

## 功能说明

### Backend 与 KV Mode

工程只向学生暴露两个 backend 概念：

```text
reference  HuggingFace/PyTorch oracle，用于 correctness 对齐，不作为课程主线
course     课程主线 engine，调度、KV cache、CUDA kernel、服务化都在这里展开
```

KV cache 形态不是第三个 backend，而是 `course` backend 内部的实现模式：

```bash
--backend reference
--backend course --kv-mode dense
--backend course --kv-mode paged
```

为了兼容早期脚本，`--backend hf` 会映射到 `reference`，`--backend paged` 会映射到 `course --kv-mode paged`。新文档和实验统一使用 `reference|course` 加 `dense|paged` 的写法。

### Prefill / Decode

`Engine.generate_stream` 会先对 prompt 做 prefill，得到第一步 logits 和 KV cache；之后每次 decode 只输入最新 token，并复用之前的 KV cache。这样代码能直接展示 LLM 推理中 prefill 和 decode 的区别。

### KV Cache

`course_vllm/engine/kv_cache.py` 实现连续 KV cache，用于教学中最直观的 KV 存储方式。运行时通过 `--backend course --kv-mode dense` 使用这种缓存。

### Paged KV Cache

`course_vllm/engine/block_manager.py` 和 `course_vllm/engine/paged_kv_cache.py` 实现 paged KV：

- KV cache 被切成固定大小 block。
- 每个 sequence 持有 block table。
- block table 把逻辑 token 位置映射到物理 KV slot。
- 请求结束后释放 block。

`--backend course --kv-mode paged` 会把每层 K/V 写进物理 slot，并在 decode 阶段通过 block table 读取历史上下文。

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
- tiled matmul
- dense prefill attention
- dense decode attention
- paged attention decode

`kernels/course_ops.cpp` 做 PyTorch binding、tensor shape/dtype/device 检查和 kernel launcher 调用。

`course_vllm/kernels/cuda_ops.py` 提供 Python 包装函数：

```python
cuda_softmax
cuda_rms_norm
cuda_rope
cuda_matmul
cuda_matmul_tiled
cuda_dense_attention_prefill
cuda_dense_attention_decode
cuda_paged_attention_decode
```

paged attention decode 在 CUDA 可用且满足限制时走手写 CUDA kernel；否则保留 PyTorch reference 作为正确性 oracle。

### 当前限制

- 单进程、单卡。
- 不是工业级多 worker / 多 GPU serving。
- streaming 请求当前通过单 model worker 串行执行。
- CUDA kernels 是教学实现，重点是正确性和可读性，不追求极限性能。
- AscendC 暂缓，等待后续硬件/后端条件后再补。

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
