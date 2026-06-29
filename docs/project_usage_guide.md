# Course LLM Serving 项目使用手册

本文档面向教师、助教和项目维护者，说明这个工程如何安装、运行、验证、压测、生成学生分支，以及课堂上应该怎么用。完整代码说明见 `docs/code_walkthrough.md`。

## 1. 项目定位

`course-llm-serving` 是一套课程用 LLM serving 教学工程。它不是工业级 vLLM 或 SGLang 的复刻，而是把推理服务里的关键概念放进一个小型、可运行、可测试的代码库：

- prefill / decode
- dense KV cache
- paged KV cache、block table、slot mapping
- batch prefill / batch decode
- continuous batching
- HTTP serving、streaming、chat API
- CUDA extension harness
- RMSNorm、RoPE、softmax、matmul、dense attention、paged attention decode 等教学 kernel
- profiling、capacity planning、paper-to-system demo

当前 Python 包名是 `course_vllm`，仓库展示名是 course LLM serving。

## 2. 分支说明

```text
main     完整答案版，教师/助教维护和验收用
student  学生 starter 版，核心 lab 代码被替换成 TODO(labXX)
```

`main` 保留完整实现。`student` 从 `main` 生成，学生在 `student` 上补代码。

生成脚本：

```bash
python scripts/validation/generate_student_branch.py --branch student
```

注意：生成脚本要求当前工作区干净，并会创建新分支。已有 `student` 分支时不要直接覆盖，先确认远端和本地状态。

## 3. 环境安装

进入项目：

```bash
cd /home/wangqi/llm_serving/course-vllm
```

创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

主要依赖：

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

CUDA kernel 测试需要：

```text
NVIDIA GPU
CUDA toolkit
ninja
与 nvcc 兼容的 host C++ compiler
```

## 4. 模型配置

默认模型：

```text
Qwen/Qwen3-0.6B
```

如果机器已有本地模型目录，建议直接传本地路径：

```bash
--model /path/to/Qwen3-0.6B
```

如果要离线运行，设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 5. 核心运行参数

工程公开四类关键开关：

```bash
--backend reference|course
--kv-mode dense|paged
--stage week04
--kernel-impl torch|auto|cuda
```

含义：

```text
reference  HuggingFace/PyTorch oracle，用于正确性对齐
course     课程主线 engine
dense      course backend 使用连续 KV cache
paged      course backend 使用 paged KV cache
torch      只走 PyTorch/reference 路径
auto       CUDA tensor 上优先尝试课程 CUDA kernel，失败回退
cuda       强制课程 CUDA kernel，kernel 不可用时报错
```

兼容旧写法：

```text
--backend hf    等价于 --backend reference
--backend paged 等价于 --backend course --kv-mode paged
```

新文档和课堂命令统一使用 `reference|course` 加 `dense|paged`。

## 6. 离线生成

单 prompt：

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week04 \
  --kernel-impl auto \
  --prompt "用一句话介绍你自己。" \
  --max-tokens 128 \
  --temperature 0
```

chat template：

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --chat \
  --prompt "用一句话介绍你自己。" \
  --max-tokens 128 \
  --temperature 0
```

batch 生成：

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --prompts "Hello|What is KV cache?" \
  --max-tokens 64 \
  --temperature 0
```

学生分支早期没有完成 course 主线时，可先用 reference：

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend reference \
  --prompt "Hello" \
  --max-tokens 16
```

## 7. HTTP 服务

启动服务：

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

`/health` 返回内容包括：

```text
status
model
backend
kv_mode
stage
kernel_impl
model_backend
batching stats
system optimization flags
```

普通生成：

```bash
curl -s -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":false,"sampling_params":{"temperature":0,"max_tokens":64}}'
```

SSE streaming：

```bash
curl -s -N -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":true,"sampling_params":{"temperature":0,"max_tokens":64}}'
```

Chat API：

```bash
curl -s -N -X POST http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话介绍你自己。"}],"stream":true,"sampling_params":{"temperature":0,"max_tokens":128}}'
```

## 8. 命令行客户端

启动服务后，另开终端：

```bash
python examples/chat_client.py
```

一次性调用：

```bash
python examples/chat_client.py \
  --once "用一句话介绍你自己。" \
  --max-tokens 128 \
  --temperature 0
```

常用交互命令：

```text
/help
/health
/params
/set max_tokens 512
/set max_tokens none
/set temperature 0
/set top_k 40
/stream on
/stream off
/system <text>
/history [n]
/clear
/save [path]
/load <path>
/exit
```

## 9. Paged KV 演示

```bash
python examples/block_usage.py \
  --num-blocks 8 \
  --block-size 4 \
  --prompt-lens 3,6,9 \
  --decode-steps 2
```

这个脚本用于观察：

```text
block allocation
block table
decode append 后 block 增长
fragmentation
```

## 10. 测试与按周验收

基础测试：

```bash
pytest -q -rs
```

按周 grader：

```bash
python -m course_vllm.benchmarks.grader week01
python -m course_vllm.benchmarks.grader week02
python -m course_vllm.benchmarks.grader week11
python -m course_vllm.benchmarks.grader week13
python -m course_vllm.benchmarks.grader week15
```

CUDA smoke：

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
```

没有 CUDA 时，普通 CUDA 测试会 skip；`cuda_smoke` 会设置严格 CUDA 环境变量，适合助教验收。

## 11. HuggingFace 对齐

forward 对齐：

```bash
python validation/compare_qwen3.py forward \
  --model Qwen/Qwen3-0.6B \
  --dtype float32
```

decode 对齐：

```bash
python validation/compare_qwen3.py decode \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --dtype float32
```

batch prefill：

```bash
python validation/compare_qwen3.py batch-prefill \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --dtype float32
```

batch decode：

```bash
python validation/compare_qwen3.py batch-decode \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --dtype float32
```

## 12. Benchmark

服务启动后压测：

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18080/generate \
  --num-requests 16 \
  --concurrency 4 \
  --max-tokens 16 \
  --json
```

输出指标：

```text
requests_per_s
output_tokens_per_s
latency_avg_s
latency_p50_s
latency_p90_s
latency_p99_s
estimated_tpot_s
```

## 13. Profiling

Torch profiler：

```bash
python scripts/profile/torch_profiler.py \
  --backend course \
  --kv-mode paged \
  --workload mixed \
  --warmup 1 \
  --repeat 3 \
  --max-tokens 8
```

`--workload` 可选：

```text
prefill
decode
mixed
```

Nsight Systems：

```bash
bash scripts/profile/nsys_server.sh
```

Nsight Compute 指定 kernel 场景：

```bash
KERNEL_SCENARIO=paged_attention bash scripts/profile/ncu_kernel.sh
KERNEL_SCENARIO=matmul KERNEL_NAME=matmul bash scripts/profile/ncu_kernel.sh
```

`KERNEL_SCENARIO` 可选：

```text
rmsnorm
rope
softmax
matmul
dense_attention
paged_attention
all
```

## 14. Week 13 容量规划

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

报告字段包括：

```text
KV budget
KV blocks
token slots
full-length sequences
TP all-reduce bytes/token
PP bubble fraction
EP all-to-all bytes/token
CP pass-Q bytes/token
CP pass-KV bytes/prefill
communication time/token
```

## 15. Week 15 前沿专题 Demo

Cache-aware serving：

```bash
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "cache-aware serving" \
  --prompts "1,2,3,4|1,2,3,9|8,7|1,2,5"
```

Prefill-decode disaggregation：

```bash
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "prefill-decode disaggregation" \
  --requests "128:16|2048:8|256:64|1024:12"
```

TokenDance-style scheduling：

```bash
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "tokendance-style scheduling" \
  --requests "128:32|2048:4|256:16"
```

## 16. 学生分支怎么用

切到学生分支：

```bash
git switch student
```

学生先跑 oracle：

```bash
python -m course_vllm.server.api \
  --backend reference \
  --stage week01 \
  --port 18080
```

然后逐周补 TODO：

```bash
rg "TODO\\(lab" kernels course_vllm
python -m course_vllm.benchmarks.grader week03
python -m course_vllm.benchmarks.grader week04
```

`main` 是完整答案，不直接发给学生；`student` 才是 starter。

## 17. 常见问题

### CUDA 测试 skip

说明当前环境看不到 CUDA 或 extension 无法编译。普通开发可以先跑 CPU/文档/benchmark 测试；CUDA lab 验收必须在 GPU 环境跑。

### `--backend course` 在 student 分支报 TODO

这是正常的。学生没有补完对应周次时，course 主线会失败在 `TODO(labXX)`。早期演示用：

```bash
--backend reference
```

### 模型下载失败

优先使用本地模型路径：

```bash
--model /path/to/Qwen3-0.6B
```

或者设置离线环境变量，确保 HuggingFace cache 已有模型。

### push GitHub 失败

本机可用 key 是：

```text
/home/wangqi/.ssh/id_ed25519_github
```

推送时可用：

```bash
GIT_SSH_COMMAND='ssh -i /home/wangqi/.ssh/id_ed25519_github -o IdentitiesOnly=yes' git push origin main student
```
