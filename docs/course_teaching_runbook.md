# course-vllm 课程教学全过程运行手册

本文档面向教师和助教，说明如何把 `course-vllm` 当作贯穿全学期的课程项目使用。它按 16 周组织课堂目标、代码入口、演示命令、学生任务、自动评测和交付物。AscendC 第 14 周按当前课程决策暂缓，其余周次均已跑通过。

## 0. 上课前准备

### 0.1 文档边界

本文是教师/助教内部执行手册。学生发布版以 `docs/labs/README.md` 和 `docs/labs/week*.md` 为准；学生版只保留相对路径、通用环境要求、示例输出和交付物，不包含 TA 本机路径、硬件型号、profiles 文件名或一次性实测结论。

### 0.2 进入工程

```bash
cd course-vllm
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 0.3 确认 GPU

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

注意：某些沙箱默认看不到 GPU。若 `torch.cuda.is_available()` 为 False，但 `nvidia-smi` 在非沙箱终端可见，需要在 GPU 可见环境运行 CUDA 测试和 profiler。

### 0.4 全量健康检查

```bash
pytest -q -rs
```

### 0.5 逐周自动评测

```bash
for week in week01 week02 week03 week04 week05 week06 week07 week08 week09 week10 week11 week12 week13 week15; do
  echo "=== $week ==="
  python -m course_vllm.benchmarks.grader "$week"
done
```

第 14 周 AscendC 暂缓，不纳入当前自动评测。

### 0.6 强制 CUDA 验收

课堂演示可以使用 `--kernel-impl=auto` 展示自动回退行为；阶段验收至少跑一次强制 CUDA smoke，避免 CUDA kernel 未真正接入时被 fallback 掩盖。

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
```

### 0.7 Clean-clone smoke test

发布前由 TA 在新目录从零验证一次，证明结果不依赖当前工作树的历史缓存、旧 profiles 或已安装包。

```bash
bash scripts/validation/clean_clone_smoke.sh /tmp/course-vllm-clean-smoke
```

该脚本会 clone 当前仓库、创建新 venv、安装项目、运行非 CUDA 基础 pytest、`grader week01/week02/week11/week12`，并启动一次 HTTP demo。CUDA kernel 接入仍由 `grader cuda_smoke` 在 GPU 可见且 nvcc/G++ 兼容的环境单独验收。

推送远程后，可用公开仓库 fresh clone 复验：

```bash
REMOTE_URL=git@github.com:HanekawaTsubasa0/course-vllm.git \
  bash scripts/validation/clean_clone_smoke.sh /tmp/course-vllm-github-smoke
```

### 0.8 课程工程核心开关

服务和离线脚本都支持：

```text
--backend reference|course
--kv-mode dense|paged
--stage week01..week16
--kernel-impl torch|auto|cuda
```

- `reference`: HuggingFace/PyTorch oracle，只用于 correctness 对齐和评测参考。
- `course`: 课程主线 engine，调度、KV cache、CUDA kernel 和服务化都在这里展开。
- `kv-mode=dense`: 连续 KV cache，用于讲清最直观的 append/fetch。
- `kv-mode=paged`: paged KV cache 和 block table，是后半学期 serving 主线。
- `kernel-impl=torch`: 只走 PyTorch/reference。
- `kernel-impl=auto`: CUDA tensor 上优先课程 CUDA kernel，失败回退。
- `kernel-impl=cuda`: 强制课程 CUDA kernel，kernel 不可用时报错。

兼容说明：早期脚本中的 `--backend hf` 会映射到 `reference`，`--backend paged` 会映射到 `course --kv-mode paged`；新讲义和课堂命令统一使用 `reference|course` 加 `dense|paged`。

## 1. Week 01 课程导论与 Baseline Serving

### 教学目标

- 讲清楚 LLM serving 的 prefill 和 decode。
- 介绍 TTFT、TPOT、requests/s、tokens/s、SLO。
- 让学生第一次启动服务并看到流式输出。

### 代码入口

- `course_vllm/server/api.py`
- `course_vllm/engine/engine.py`
- `course_vllm/engine/request.py`
- `examples/chat_client.py`

### 课堂演示

启动服务：

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week01 \
  --kernel-impl auto \
  --dtype bfloat16 \
  --port 18080
```

健康检查：

```bash
curl -s http://127.0.0.1:18080/health
```

非流式请求：

```bash
curl -s -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":false,"sampling_params":{"temperature":0,"max_tokens":8}}'
```

流式请求：

```bash
curl -s -N -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":true,"sampling_params":{"temperature":0,"max_tokens":8}}'
```

### 学生任务

- 在 `Engine.generate_stream` 中标注 prefill、sample、decode、stop 的位置。
- 记录一次 baseline 请求延迟。
- 解释为什么 decode 每次只输入一个 token。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week01
```

### 交付物

- 服务启动命令。
- `/health` 输出。
- 一条 streaming 和一条 non-streaming 请求结果。
- prefill/decode 调用链说明。

## 2. Week 02 性能分析

### 教学目标

- 讲 FLOPs、访存量、计算访存比、roofline。
- 使用 torch profiler、nsys、ncu 定位瓶颈。
- 建立服务性能基线。

### 代码入口

- `course_vllm/benchmarks/bench_server.py`
- `scripts/profile/torch_profiler.py`
- `scripts/profile/nsys_server.sh`
- `scripts/profile/ncu_kernel.sh`
- `docs/reports/week02_profile_template.md`

### 课堂演示

Torch profiler：

```bash
python scripts/profile/torch_profiler.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --max-tokens 8 \
  --out profiles/torch_profiler \
  | tee profiles/reports/torch_profiler_summary.txt
```

Nsight Systems：

```bash
MODEL=Qwen/Qwen3-0.6B BACKEND=course KV_MODE=paged DTYPE=bfloat16 MAX_TOKENS=8 OUT=profiles/nsys_server_ready \
  bash scripts/profile/nsys_server.sh \
  | tee profiles/reports/nsys_server_ready_summary.txt
```

Nsight Compute：

```bash
KERNEL_SCENARIO=paged_attention OUT=profiles/ncu_paged_attention bash scripts/profile/ncu_kernel.sh \
  | tee profiles/reports/ncu_paged_attention_summary.txt
KERNEL_SCENARIO=matmul KERNEL_NAME=matmul OUT=profiles/ncu_matmul bash scripts/profile/ncu_kernel.sh \
  | tee profiles/reports/ncu_kernel_summary.txt
```

### 示例观察点

- nsys benchmark 至少记录 requests/s、output tokens/s、p50/p90/p99 和 estimated TPOT。
- torch profiler 至少列出 3 个主要 CUDA 或 PyTorch hotspot。
- ncu 若遇到 `ERR_NVGPUCTRPERM`，需要在报告中说明权限限制，并用 CUDA smoke 证明 kernel 可以编译和运行。

### 学生任务

- 截取 nsys timeline 或说明 CUDA kernel 排布。
- 从 profiler 表中找出 3 个主要耗时项。
- 写出一个瓶颈判断和下一步优化方向。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week02
```

### 交付物

- `docs/reports/week02_profile_template.md` 填写完成。
- profiler 输出文件路径。
- 瓶颈判断。

## 3. Week 03 CUDA 入门

### 教学目标

- 理解 CUDA kernel 基本写法。
- 学会 PyTorch extension JIT 编译。
- 搭建后续算子复用的 correctness/benchmark harness。

### 代码入口

- `kernels/vector_add.cu`
- `course_vllm/kernels/harness.py`
- `tests/test_kernels.py`

### 课堂演示

```bash
python -m pytest -q tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch -rs
```

### 学生任务

- 修改 block size，观察是否影响 correctness。
- 解释 `blockIdx`, `threadIdx`, `blockDim` 如何映射数组下标。
- 记录首次 JIT 编译时间和再次运行时间差异。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week03
```

### 交付物

- vector add 正确性结果。
- kernel index 计算说明。
- JIT 编译观察。

## 4. Week 04 RMSNorm 与 RoPE

### 教学目标

- 理解 RMSNorm、RoPE、浮点误差和混合精度。
- 把 CUDA RMSNorm/RoPE 接入 Qwen3 主路径。

### 代码入口

- `course_vllm/model/qwen3_torch.py`
- `course_vllm/kernels/cuda_ops.py`
- `kernels/course_ops.cu`

### 课堂演示

```bash
python -m pytest -q \
  tests/test_kernels.py::test_cuda_rms_norm_matches_torch \
  tests/test_kernels.py::test_cuda_rope_matches_qwen3_rotate_half \
  -rs
```

离线短生成：

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week04 \
  --kernel-impl auto \
  --prompt "Hello" \
  --max-tokens 4 \
  --temperature 0
```

本次输出示例：

```text
Answer! I'm
```

### 学生任务

- 比较 PyTorch 和 CUDA RMSNorm 输出误差。
- 比较 PyTorch 和 CUDA RoPE 输出误差。
- 用 `--kernel-impl=cuda` 确认 kernel 不可用时会失败。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week04
```

### 交付物

- RMSNorm/RoPE 误差表。
- kernel dispatch 路径说明。

## 5. Week 05 线性层与矩阵乘

### 教学目标

- 理解 QKV/MLP projection 本质上是矩阵乘。
- 对比 naive matmul、tiled matmul 和 PyTorch/cuBLAS。
- 把 tiled matmul 接入 `CourseLinear` 主路径。

### 代码入口

- `course_vllm/model/ops.py`
- `course_vllm/model/qwen3_torch.py`
- `kernels/course_ops.cu`
- `kernels/course_ops.cpp`

### 课堂演示

```bash
python -m pytest -q \
  tests/test_kernels.py::test_cuda_matmul_matches_torch \
  tests/test_kernels.py::test_cuda_matmul_tiled_matches_torch \
  tests/test_qwen3_torch.py::test_course_linear_matches_torch_linear \
  -rs
```

### 学生任务

- 画出 naive matmul 的 global memory 访问模式。
- 画出 tiled matmul 的 shared memory 复用模式。
- 计算一个 M x K 乘 K x N 的 FLOPs。
- 说明为什么教学 kernel 不要求超过 cuBLAS。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week05
```

### 交付物

- naive/tiled 正确性结果。
- naive/tiled/PyTorch 运行时间对比。
- `CourseLinear` 在 Qwen3 的接入位置。

## 6. Week 06 归约与 Softmax

### 教学目标

- 理解 parallel reduction。
- 理解稳定 softmax 为什么要减最大值。
- 把 softmax 接入 sampling 路径。

### 代码入口

- `course_vllm/engine/sampler.py`
- `course_vllm/kernels/cuda_ops.py`
- `kernels/course_ops.cu`

### 课堂演示

```bash
python -m pytest -q \
  tests/test_kernels.py::test_cuda_softmax_matches_torch \
  tests/test_sampler.py \
  -rs
```

### 学生任务

- 构造大 logits 说明朴素 softmax 溢出。
- 解释 row-wise max reduction 和 sum reduction。
- 比较 greedy、temperature、top-k 的采样路径。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week06
```

### 交付物

- softmax 误差。
- 溢出示例。
- sampling 参数说明。

## 7. Week 07 Attention

### 教学目标

- 区分 prefill attention 和 decode attention。
- 理解 online softmax/FlashAttention 风格核心思想。
- 验证 dense prefill、dense decode、paged decode CUDA 路径。

### 代码入口

- `course_vllm/model/ops.py`
- `course_vllm/model/attention.py`
- `course_vllm/model/qwen3_paged_backend.py`
- `kernels/course_ops.cu`

### 课堂演示

```bash
python -m pytest -q tests/test_attention.py -rs
```

### 学生任务

- 对比 dense prefill 和 paged decode 的输入 shape。
- 解释 online softmax 中 row max、denom、acc 如何更新。
- 说明为什么 prefill 可以按 query position 并行，decode 是单 token 查询历史 KV。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week07
```

### 交付物

- attention correctness 测试结果。
- prefill/decode 路径图。
- FlashAttention 风格内存节省说明。

## 8. Week 08 KV Cache

### 教学目标

- 理解 KV cache 的作用。
- 观察连续 KV cache append 和 decode 复用。
- 理解 batch decode 按历史长度分桶。

### 代码入口

- `course_vllm/engine/kv_cache.py`
- `course_vllm/model/qwen3_continuous_backend.py`

### 课堂演示

```bash
python -m pytest -q tests/test_kv_cache.py tests/test_qwen3_torch.py
```

### 学生任务

- 打印每层 key/value shape。
- 对比无 cache 全序列 forward 和 cache decode 数据量。
- 说明 cache handle 为什么要记录 `seq_id` 和 `seq_len`。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week08
```

### 交付物

- KV cache shape 表。
- 连续 decode 调用链。

## 9. Week 09 推理引擎

### 教学目标

- 实现请求生命周期。
- 理解 tokenizer 边界、stop 条件、streaming event。
- 对单请求推理做 profiler baseline。

### 代码入口

- `course_vllm/engine/engine.py`
- `course_vllm/engine/request.py`
- `scripts/profile/torch_profiler.py`

### 课堂演示

```bash
python -m pytest -q tests/test_engine.py tests/test_chat_client.py
python scripts/profile/torch_profiler.py --model Qwen/Qwen3-0.6B --backend course \
  --kv-mode paged --max-tokens 8
```

### 学生任务

- 追踪 `Request` 和 `Sequence` 状态变化。
- 解释 EOS、stop token、max_tokens 三种结束原因。
- 保存 profiler 输出。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week09
```

### 交付物

- 请求状态机。
- profiler baseline。

## 10. Week 10 分页 KV Cache

### 教学目标

- 理解 block manager、block table、slot mapping。
- 理解教学版 prefix cache 和碎片统计。

说明：本周 prefix cache 是 teaching approximation，用来展示完整 block 前缀复用、block table 和 fragmentation 的关系；它不等价于 vLLM/SGLang 生产实现中的哈希前缀匹配、引用计数、淘汰策略和跨请求生命周期管理。

### 代码入口

- `course_vllm/engine/block_manager.py`
- `course_vllm/engine/paged_kv_cache.py`
- `examples/block_usage.py`

### 课堂演示

```bash
python examples/block_usage.py \
  --num-blocks 8 \
  --block-size 4 \
  --prompt-lens 3,6,9 \
  --decode-steps 2
```

示例输出摘要：

- prefill: used=6, free=2, fragmentation_ratio=0.250
- decode step 1: fragmentation_ratio=0.125
- decode step 2: used=7, free=1, fragmentation_ratio=0.143

### 学生任务

- 手算一个 sequence 的 slot mapping。
- 解释 block table 如何把逻辑位置映射到物理 slot。
- 构造共享 prefix 的 prompt，观察 prefix cached blocks。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week10
```

### 交付物

- block table 示例。
- fragmentation 统计。
- prefix cache 复用解释。

## 11. Week 11 连续批处理

### 教学目标

- 理解 waiting/running 队列。
- 区分 prefill batch 和 decode batch。
- 支持教学版 chunked prefill、preemption 和 HTTP batching。

说明：本周调度机制是 teaching approximation。chunked prefill 用 token budget 展示长 prompt 分块推进；preemption 展示释放运行态并回到 waiting 队列。生产系统还会结合 block allocator、优先级、SLO、prefix cache、重算/换出策略和公平性。

### 代码入口

- `course_vllm/engine/scheduler.py`
- `course_vllm/server/batching.py`
- `course_vllm/engine/engine.py`

### 课堂演示

```bash
python -m pytest -q tests/test_scheduler.py tests/test_server_batching.py
```

启动服务压测：

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week11 \
  --kernel-impl auto \
  --dtype bfloat16 \
  --max-batch-size 4 \
  --batch-wait-ms 2 \
  --max-queue-size 64 \
  --max-prompt-chars 8192 \
  --port 18081
```

另一个终端：

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18081/generate \
  --num-requests 8 \
  --concurrency 2 \
  --max-tokens 8 \
  --json
```

示例 baseline 指标字段：

- requests/s
- output tokens/s
- p50/p90/p99
- estimated TPOT

TA 本机一次实测数值集中记录在本文末尾，学生报告应以自己环境复测结果为准。

### 学生任务

- 画出 scheduler 状态转换。
- 修改 `max_num_batched_tokens`，观察 chunked prefill。
- 比较不同 `batch_wait_ms` 对 batch size 和 latency 的影响。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week11
```

### 交付物

- throughput/latency 曲线。
- scheduler 策略说明。

## 12. Week 12 系统优化

### 教学目标

- 讲 pinned memory、CUDA stream、异步执行和请求准入。
- 将优化开关接入服务并对比性能。

### 代码入口

- `course_vllm/model/qwen3_continuous_backend.py`
- `course_vllm/server/batching.py`
- `course_vllm/benchmarks/system_optimization.py`
- `docs/reports/week12_system_optimization_template.md`

### 课堂演示

```bash
python -m course_vllm.benchmarks.system_optimization \
  --pinned-memory \
  --transfer-stream \
  --max-queue-size 64 \
  --max-prompt-chars 8192
```

优化配置服务：

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week12 \
  --kernel-impl auto \
  --dtype bfloat16 \
  --max-batch-size 4 \
  --batch-wait-ms 2 \
  --max-queue-size 64 \
  --max-prompt-chars 8192 \
  --enable-chunked-prefill \
  --cache-aware-scheduling \
  --pinned-memory \
  --transfer-stream \
  --port 18082
```

压测：

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18082/generate \
  --num-requests 8 \
  --concurrency 2 \
  --max-tokens 8 \
  --json
```

示例 optimized 指标字段：

- requests/s
- output tokens/s
- p50/p90/p99
- estimated TPOT

TA 本机短 prompt、短 decode、小并发实测中，optimized 的 requests/s 和 p50 没有优于 Week 11 baseline，仅 p99 略好。这里不能写成“优化显著提升”；应要求学生解释短负载下收益不明显的原因，并补充长 prompt 或更高并发实验来观察 pinned memory、transfer stream、chunked prefill 和 admission control 更可能生效的场景。

### 学生任务

- 判断短负载下优化收益为什么不明显。
- 解释 admission control 如何保护服务。
- 用 nsys 找 memcpy、kernel、同步点。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week12
python -m course_vllm.benchmarks.grader cuda_smoke
```

### 交付物

- `docs/reports/week12_system_optimization_template.md`。
- before/after benchmark 表。
- nsys evidence。

## 13. Week 13 多卡推理与容量规划

### 教学目标

- 讲 TP/PP/EP/CP、NCCL 通信和 placement。
- 在单卡上估算 KV cache 容量，并用理论模型估算多卡后的计算量、通信量、显存拆分和瓶颈。

### 代码入口

- `course_vllm/benchmarks/capacity_planner.py`
- `docs/reports/week13_capacity_planning_template.md`

### 课堂演示

```bash
python -m course_vllm.benchmarks.capacity_planner \
  --gpu-memory-gb 24 \
  --weight-memory-gb 2 \
  --num-layers 28 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --hidden-size 1024 \
  --intermediate-size 2816 \
  --block-size 16 \
  --max-model-len 2048 \
  --tp 2 \
  --pp 2 \
  --ep 1 \
  --cp 2 \
  --microbatch-size 4 \
  --network-bandwidth-gbps 200 \
  --target-concurrency 32 \
  --target-sequence-len 2048 \
  --report
```

示例容量规划字段：

- KV budget
- KV blocks
- token slots
- full-length sequences
- TP all-reduce bytes/token
- PP bubble fraction
- EP all-to-all bytes/token
- CP pass-Q/pass-KV bytes
- target concurrency 是否需要多卡扩容，以及 compute-bound / communication-bound 判断

### 学生任务

- 修改 target concurrency，找到需要多卡的阈值。
- 比较 float16/bfloat16/int8/fp8 对 KV 容量的影响。
- 比较 TP/PP/EP/CP 参数变化对计算量、通信量和显存的影响。
- 说明何时是容量瓶颈，何时是吞吐瓶颈，何时通信开销会抵消多卡收益。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week13
```

### 交付物

- `docs/reports/week13_capacity_planning_template.md`。
- 多卡触发条件判断。

## 14. Week 14 AscendC

### 当前状态

本周按课程决策暂缓。工程保留 `week14` stage，但不要求当前仓库实现 AscendC kernel、官方 Add 样例或 CUDA/Ascend 对照。课堂时间可用于前沿专题、期中补验收、Ascend 架构讲解或 CUDA/Ascend 编程模型对照；不计硬件实验分，不要求无 Ascend 环境的学生完成 AscendC kernel。

### 后续补齐条件

- 有 Ascend 硬件或 CI。
- 明确 CANN/AscendC 版本。
- 选定一个与 CUDA 教学 kernel 对应的算子，例如 vector add、RMSNorm 或 softmax。

### 当前交付

- `docs/labs/week14_ascend_deferred.md`
- `/health` 能说明 week14 deferred。

## 15. Week 15 前沿专题

### 教学目标

- 把论文机制映射到工程模块。
- 复现一个小型机制：cache-aware serving。
- 理解 prefill/decode disaggregation、TokenDance-style scheduling 等方向如何落到系统。

说明：本周 cache-aware serving 是 teaching approximation，用 shared-prefix score 展示请求重排的潜在价值；它还不是完整在线 scheduler，不包含真实到达时间、SLO、KV block 生命周期、饥饿控制或多租户公平性。

### 代码入口

- `course_vllm/engine/policies.py`
- `course_vllm/benchmarks/cache_aware_demo.py`
- `docs/reports/week15_paper_to_system_template.md`

### 课堂演示

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

示例结果字段：

- baseline shared-prefix score
- cache-aware shared-prefix score
- PD disaggregation estimated speedup
- TokenDance-style completion cost
- mapped modules: `engine/policies.py`, `engine/block_manager.py`, `engine/engine.py`

### 学生任务

- 选择一个机制写 paper-to-system map。
- 说明要改哪些数据结构。
- 说明影响 TTFT、TPOT、tokens/s、KV fragmentation 的路径。

### 自动评测

```bash
python -m course_vllm.benchmarks.grader week15
```

### 交付物

- `docs/reports/week15_paper_to_system_template.md`。
- 一项机制 demo 的 before/after 指标。

## 16. Week 16 总结与展示

### 教学目标

- 串联算子、缓存、调度、服务、profiling、容量规划和前沿机制。
- 展示完整简化版 LLM serving engine。
- 做一次工程故障诊断复盘。

### 课堂演示顺序

1. 全量测试：

```bash
pytest -q -rs
```

2. Qwen3/HF 对齐：

```bash
for mode in forward decode batch-prefill batch-decode; do
  echo "=== $mode ==="
  python validation/compare_qwen3.py "$mode" \
    --model Qwen/Qwen3-0.6B \
    --backend course \
    --kv-mode paged \
    --dtype float32
done
```

3. CUDA kernel/attention：

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
pytest -q tests/test_kernels.py tests/test_attention.py -rs
```

4. profiling：

```bash
python scripts/profile/torch_profiler.py --model Qwen/Qwen3-0.6B --backend course \
  --kv-mode paged --max-tokens 8
MODEL=Qwen/Qwen3-0.6B BACKEND=course KV_MODE=paged DTYPE=bfloat16 MAX_TOKENS=8 OUT=profiles/nsys_server_ready bash scripts/profile/nsys_server.sh
```

5. capacity planning：

```bash
python -m course_vllm.benchmarks.capacity_planner \
  --gpu-memory-gb 24 \
  --weight-memory-gb 2 \
  --num-layers 28 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --report
```

6. frontend topic demo：

```bash
python -m course_vllm.benchmarks.cache_aware_demo --mechanism "cache-aware serving"
python -m course_vllm.benchmarks.cache_aware_demo --mechanism "prefill-decode disaggregation"
python -m course_vllm.benchmarks.cache_aware_demo --mechanism "tokendance-style scheduling"
```

### 最终交付物

- `docs/reports/week16_final_report_template.md`
- `docs/runnable_validation_guide.md`
- `profiles/reports/*`
- `profiles/nsys_server_ready.nsys-rep`

## 17. 常见故障与处理

### GPU 不可见

现象：

```text
torch.cuda.is_available() == False
```

处理：

- 先运行 `nvidia-smi`。
- 如果 `nvidia-smi` 在普通终端可见但当前环境不可见，说明是沙箱/容器权限问题。
- 在 GPU 可见环境重跑 CUDA/profiler 命令。

### HF 模型联网超时

现象：

```text
HTTPSConnectionPool(host='huggingface.co'...)
```

处理：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

课程 backend 已支持把 `Qwen/Qwen3-0.6B` 自动解析到本地 HuggingFace snapshot。

### ncu 找不到

脚本会自动找：

- `/usr/local/cuda-12.8/bin/ncu`
- `/usr/local/cuda/bin/ncu`
- `/usr/local/NVIDIA-Nsight-Compute-2025.3/ncu`

也可以手动：

```bash
NCU_BIN=/usr/local/cuda-12.8/bin/ncu KERNEL_SCENARIO=paged_attention OUT=profiles/ncu_paged_attention bash scripts/profile/ncu_kernel.sh
```

### ncu 无 performance counter 权限

现象：

```text
ERR_NVGPUCTRPERM
```

处理：

- 让管理员放开 NVIDIA GPU Performance Counters。
- 或使用有权限的用户运行。
- 课堂上可先用 torch profiler/nsys 完成瓶颈定位，说明 ncu 权限限制。

### nsys 服务未启动就压测

已修复：`scripts/profile/nsys_server.sh` 会轮询 `/health`，ready 后才运行 benchmark。

## 18. 课程代码与文档对应关系

| 类别 | 文件 |
| --- | --- |
| 周次定义 | `course_vllm/stages.py` |
| 自动评测 | `course_vllm/benchmarks/grader.py` |
| CUDA wrapper | `course_vllm/kernels/cuda_ops.py` |
| CUDA source | `kernels/course_ops.cu`, `kernels/course_ops.cpp` |
| Qwen3 模型 | `course_vllm/model/qwen3_torch.py` |
| 连续 KV | `course_vllm/engine/kv_cache.py` |
| 分页 KV | `course_vllm/engine/paged_kv_cache.py`, `course_vllm/engine/block_manager.py` |
| 调度 | `course_vllm/engine/scheduler.py` |
| HTTP batching | `course_vllm/server/batching.py` |
| 服务 API | `course_vllm/server/api.py` |
| 实验文档 | `docs/labs/week*.md` |
| 报告模板 | `docs/reports/*.md` |
| 学生分支计划 | `docs/student_branch_plan.md` |
| TA 验证记录 | `docs/validation/ta_validation_log.md` |
| 运行验收 | `docs/runnable_validation_guide.md` |

## 19. TA validation log

以下记录仅用于 TA 内部发布检查，不进入学生 handout。学生文档应使用相对路径和“示例输出”，不要要求学生复现 TA 机器的绝对数值。

### 19.1 本次验证环境

- Workspace: `/home/wangqi/llm_serving/course-vllm`
- GPU: 2 x NVIDIA GeForce RTX 4090
- Driver: 570.86.15
- CUDA driver capability: 12.8
- PyTorch: 2.8.0+cu128
- Model: Qwen/Qwen3-0.6B, local HuggingFace cache

### 19.2 本次健康检查摘要

- Full pytest: 88 passed in 4.56s
- Stage grader: week01/week02/week03/week04/week05/week06/week07/week08/week09/week10/week11/week12/week13/week15 passed
- CUDA smoke: required before publication; run `python -m course_vllm.benchmarks.grader cuda_smoke` in a GPU-visible environment
- Clean-clone smoke: required before publication; run `bash scripts/validation/clean_clone_smoke.sh /tmp/course-vllm-clean-smoke`
- Strict CUDA smoke: required on a GPU-visible environment with a CUDA-compatible host compiler; run `python -m course_vllm.benchmarks.grader cuda_smoke`

### 19.3 本次性能观察

Week 11 baseline short workload:

- requests/s=4.038430
- output tokens/s=32.307436
- p50=0.231115
- p99=1.248274

Week 12 optimized short workload:

- requests/s=3.981867
- output tokens/s=31.854933
- p50=0.251096
- p99=1.219466

结论：短 prompt、短 decode、小并发下，optimized 的吞吐和 p50 没有优于 Week 11 baseline，只有 p99 略好。发布材料应把它作为“优化收益不明显，需要解释 workload 原因”的案例，而不是性能提升证据。

### 19.4 本次证据文件

```text
profiles/reports/all_stage_grader_summary.txt
profiles/reports/course_demo_smoke.txt
profiles/reports/qwen3_alignment_float32.txt
profiles/reports/torch_profiler_summary.txt
profiles/reports/ncu_kernel_summary.txt
profiles/reports/nsys_server_ready_summary.txt
profiles/reports/bench_baseline_week11.json
profiles/reports/bench_optimized_week12.json
profiles/reports/week12_system_optimization_plan.json
profiles/reports/week13_capacity_report.md
profiles/reports/week15_cache_aware_demo.json
profiles/nsys_server_ready.nsys-rep
profiles/torch_profiler/*.pt.trace.json
```

这些文件可用于 TA 课程展示、阶段检查和最终报告，不作为学生版固定路径要求。

## 20. Review closure

| 评论项 | 当前处理 | 后续要求 |
| --- | --- | --- |
| 区分 TA runbook 和学生 handout | 本手册明确为 TA 内部；学生入口为 `docs/labs/README.md` 与逐周 lab | 学生版只保留相对路径、示例输出和交付物 |
| Week 12 优化数据谨慎表述 | 已改为短负载收益不明显，要求解释原因 | 可补长 prompt/高并发实验作为附加证据 |
| `auto` kernel 掩盖问题 | 新增 `grader cuda_smoke`，发布前强制执行 | 课堂 demo 可继续使用 `auto` |
| Week 10/11/15 机制过度承诺 | 已标注 teaching approximation，并说明生产差距 | 报告中必须区分 demo 和生产实现 |
| Week 14 暂缓出现空洞 | 已加入课堂替代项和不计硬件实验说明 | 有 Ascend 硬件/CI 后再补真实实验 |
| clean-clone 验收 | 新增 `scripts/validation/clean_clone_smoke.sh` 和 runbook 命令 | 发布前必须保存一次输出记录 |
| 旧 review 的课程闭环 | 周次、profiler、CUDA、grader、报告模板已经串联 | AscendC 仍为明确 deferred |
