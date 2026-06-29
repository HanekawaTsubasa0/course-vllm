# course LLM serving 按周代码讲解与任务手册

本文面向授课教师，用来把整个 `course-vllm` 工程按 16 周课程顺序讲清楚。它不是完整 API 文档，也不是逐文件索引：

- 想知道项目怎么安装、启动、压测、生成学生分支，看 `docs/project_usage_guide.md`。
- 想按模块理解所有核心代码，看 `docs/code_walkthrough.md`。
- 想按周安排课堂、对着代码讲任务、给学生运行命令，用本文。

授课时建议始终让学生区分两个版本：

- `main`：教师版和完整答案，课堂讲解、对照正确实现、发布前验收都用它。
- `student`：学生 starter 版，核心实验代码被 `TODO(labXX)` 挖空，学生从这里补代码。

服务和脚本里最重要的四个开关是：

```bash
--backend reference|course
--kv-mode dense|paged
--stage week01
--kernel-impl torch|auto|cuda
```

- `reference` 是 HuggingFace/PyTorch oracle，用来先跑通服务和做正确性对照。
- `course` 是课程主线 engine，后续所有系统设计都落在这里。
- `dense` 用连续 KV cache，适合先讲清 append/fetch。
- `paged` 用 block table 和 paged KV cache，是后半学期的 serving 主线。
- `torch` 只走 PyTorch/reference path。
- `auto` 在 CUDA tensor 上优先走课程 CUDA kernel，失败时回退。
- `cuda` 强制走课程 CUDA kernel，适合验收是否真的接入。

## 建议课堂节奏

每周课堂按这个固定顺序组织，学生最容易建立工程地图：

1. 先运行本周 demo，让学生看到现象。
2. 打开 2 到 4 个核心文件，从入口函数沿调用链读到关键逻辑。
3. 切到 `student` 分支，展示本周 `TODO(labXX)` 在哪里。
4. 讲函数签名、输入输出 shape、推荐实现步骤。
5. 跑本周 pytest 或 grader，告诉学生交付标准。

学生在早期没有补完课程代码时，可以先用 `--backend reference` 跑服务；补到对应周次后，再切到 `--backend course` 和对应 `kv-mode` 验证自己的实现。

## Week 01：课程导论与 baseline serving

本周让学生第一次理解 serving 请求从 HTTP 入口到 token 流式返回的完整路径。重点不是优化，而是建立 prefill、decode、sample、stop 的基本时序。

课堂打开这些文件：

- `course_vllm/server/api.py`
- `course_vllm/server/batching.py`
- `course_vllm/engine/engine.py`
- `course_vllm/engine/request.py`
- `examples/chat_client.py`

代码讲解顺序：

1. 从 `course_vllm.server.api:create_app` 看 `/health` 和 `/generate` 路由。
2. 在 `/generate` 里说明请求体如何转成 prompt、sampling params 和 stream/non-stream 模式。
3. 追到 `BatchingEngine.generate`，说明 HTTP 层和推理 engine 之间的边界。
4. 进入 `Engine.generate_stream`，指出 prefill、采样、decode loop、finish reason 分别在哪里。
5. 回到 `examples/chat_client.py`，解释客户端如何消费 streaming response。

学生任务：

- 能启动服务并请求 `/health`、`/generate`。
- 能在 `Engine.generate_stream` 的调用链里标出 prefill 和 decode。
- 能解释为什么 prefill 一次处理 prompt，而 decode 每步只处理一个新 token。

运行方式：

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend reference \
  --stage week01 \
  --port 18080
```

```bash
curl -s http://127.0.0.1:18080/health
curl -s -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":false,"sampling_params":{"temperature":0,"max_tokens":8}}'
curl -s -N -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":true,"sampling_params":{"temperature":0,"max_tokens":8}}'
```

验收：

```bash
python -m course_vllm.benchmarks.grader week01
```

## Week 02：性能分析与指标

本周把“能跑”推进到“知道慢在哪里”。课堂重点是 TTFT、TPOT、tokens/s、requests/s、p50/p90/p99，以及 profiler 产物如何对应到代码。

课堂打开这些文件：

- `course_vllm/benchmarks/bench_server.py`
- `scripts/profile/torch_profiler.py`
- `scripts/profile/nsys_server.sh`
- `scripts/profile/ncu_kernel.sh`
- `docs/reports/week02_profile_template.md`

代码讲解顺序：

1. 从 `bench_server.py` 看压测请求如何并发发出、如何统计延迟和吞吐。
2. 看 `torch_profiler.py`，解释 profiler 的 schedule、trace 输出和 summary。
3. 看 `nsys_server.sh`，说明系统级 timeline 适合观察 CPU/GPU 空洞和服务端等待。
4. 看 `ncu_kernel.sh`，说明 kernel 级指标适合观察访存、occupancy 和 kernel launch。
5. 打开报告模板，让学生把 profiler 观察写成“指标 -> 现象 -> 可能瓶颈”。

学生任务：

- 跑一次服务端压测，记录吞吐和延迟。
- 跑一次 torch profiler，列出主要热点。
- 如果机器有 Nsight，补充 nsys 或 ncu 截图/摘要；没有权限时要写明原因。

运行方式：

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week02 \
  --kernel-impl auto \
  --port 18080
```

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18080/generate \
  --requests 8 \
  --concurrency 2 \
  --prompt "Explain KV cache in one sentence." \
  --max-tokens 16
```

```bash
python scripts/profile/torch_profiler.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --workload mixed \
  --max-tokens 8 \
  --out profiles/torch_profiler
```

```bash
KERNEL_SCENARIO=paged_attention bash scripts/profile/ncu_kernel.sh
```

验收：

```bash
python -m course_vllm.benchmarks.grader week02
```

## Week 03：CUDA 入门与 vector add

本周让学生第一次写 CUDA extension。代码量很小，但要讲清楚 Python 调用 C++/CUDA kernel 的路径、grid/block、边界检查和测试方式。

课堂打开这些文件：

- `kernels/vector_add.cu`
- `course_vllm/kernels/cuda_ops.py`
- `course_vllm/kernels/build.py`
- `tests/test_kernels.py`
- `docs/labs/week03_cuda_intro.md`

代码讲解顺序：

1. 从测试 `test_vector_add_cuda_kernel_matches_torch` 看输入输出 shape：`a,b,out` 都是一维 CUDA tensor。
2. 打开 `cuda_ops.py`，说明 Python 包装函数做参数检查和 extension 调用。
3. 打开 `build.py`，说明 PyTorch JIT extension 如何编译 `kernels/*.cu`。
4. 打开 `vector_add.cu`，讲 `idx = blockIdx.x * blockDim.x + threadIdx.x`。
5. 在 `student` 分支指出 `TODO(lab03)`：只需要实现越界检查和 `out[idx] = a[idx] + b[idx]`。

学生任务：

- 实现 `vector_add_kernel`。
- 输入 shape：`a: [n]`，`b: [n]`。
- 输出 shape：`out: [n]`。
- 推荐步骤：算全局线程 id，判断 `idx < n`，写入加法结果。

运行方式：

```bash
python -m pytest -q tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch -rs
python -m course_vllm.benchmarks.grader week03
```

## Week 04：RMSNorm 与 RoPE

本周把 CUDA kernel 接进真实模型层。重点是让学生看到算子不是孤立存在，而是从 Qwen3 模型 forward 调用到课程 kernel。

课堂打开这些文件：

- `course_vllm/model/qwen3_torch.py`
- `course_vllm/kernels/cuda_ops.py`
- `kernels/course_ops.cu`
- `tests/test_kernels.py`
- `tests/test_qwen3_torch.py`
- `docs/labs/week04_rms_rope.md`

代码讲解顺序：

1. 从 `Qwen3RMSNorm.forward` 看 RMSNorm 的数学形式：按 hidden 维求均方、rsqrt、乘 weight。
2. 从 `apply_rotary_pos_emb` 看 RoPE 如何把 cos/sin 应用到 Q/K。
3. 进入 `cuda_ops.py`，说明 `kernel_impl=torch|auto|cuda` 三种行为。
4. 打开 `course_ops.cu` 的 RMSNorm 和 RoPE kernel，讲每行/每 token 的并行粒度。
5. 回到模型 forward，指出 attention 前必须先做 norm 和 rotary。

学生任务：

- `TODO(lab04)` 在 `Qwen3RMSNorm.forward`、`apply_rotary_pos_emb` 和 `kernels/course_ops.cu`。
- RMSNorm 输入 shape 通常是 `x: [tokens, hidden]`，`weight: [hidden]`，输出同 `x`。
- RoPE 输入 shape 通常是 `q,k: [batch, heads, seq, head_dim]` 或模型内部等价展平视图，输出 shape 不变。
- 推荐先写 PyTorch path 对齐，再接 CUDA path。

运行方式：

```bash
python -m pytest -q \
  tests/test_kernels.py::test_cuda_rms_norm_matches_torch \
  tests/test_kernels.py::test_cuda_rope_matches_qwen3_rotate_half -rs
```

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode dense \
  --stage week04 \
  --kernel-impl auto \
  --prompt "用一句话介绍 RMSNorm。" \
  --max-tokens 16 \
  --temperature 0
```

验收：

```bash
python -m course_vllm.benchmarks.grader week04
```

## Week 05：线性层与矩阵乘

本周从单个 kernel 进入模型中最重要的计算模式：linear projection 和 MLP matmul。课堂要把 naive matmul、tiled matmul、`CourseLinear` 的 dispatch 关系讲清楚。

课堂打开这些文件：

- `kernels/course_ops.cu`
- `course_vllm/model/ops.py`
- `course_vllm/model/qwen3_torch.py`
- `tests/test_kernels.py`
- `tests/test_qwen3_torch.py`
- `docs/labs/week05_matmul_linear.md`

代码讲解顺序：

1. 从 `test_cuda_matmul_matches_torch` 看矩阵乘输入输出：`A: [M,K]`，`B: [K,N]`，`C: [M,N]`。
2. 讲 `course_ops.cu` 里的 naive matmul：每个线程负责一个输出元素。
3. 讲 tiled matmul：shared memory tile 如何减少 global memory 访问。
4. 打开 `CourseLinear.forward`，说明它如何在 CUDA kernel 和 `torch.nn.functional.linear` 之间切换。
5. 回到 `Qwen3Attention` 和 `Qwen3MLP`，指出 Q/K/V/O projection、gate/up/down projection 都依赖 linear。

学生任务：

- `TODO(lab05)` 在 `course_ops.cu` 的 matmul kernel 和 `CourseLinear.forward`。
- 输入 shape：`x: [tokens, in_features]`，`weight: [out_features, in_features]`。
- 输出 shape：`[tokens, out_features]`。
- 推荐步骤：先实现 naive matmul，再实现 tiled matmul，最后在 `CourseLinear.forward` 接入 dispatch 和 fallback。

运行方式：

```bash
python -m pytest -q \
  tests/test_kernels.py::test_cuda_matmul_matches_torch \
  tests/test_kernels.py::test_cuda_matmul_tiled_matches_torch -rs
```

```bash
python -m pytest -q tests/test_qwen3_torch.py::test_course_linear_matches_torch_linear
python -m course_vllm.benchmarks.grader week05
```

## Week 06：Softmax 与 sampling

本周把归约类 CUDA kernel 接到采样逻辑。重点是稳定 softmax、temperature、top-k/top-p 和 deterministic sampling 的测试方法。

课堂打开这些文件：

- `course_vllm/engine/sampler.py`
- `kernels/course_ops.cu`
- `course_vllm/kernels/cuda_ops.py`
- `tests/test_sampler.py`
- `tests/test_kernels.py`
- `docs/labs/week06_softmax_sampling.md`

代码讲解顺序：

1. 从 `Sampler.sample` 看 logits 如何变成 next token。
2. 进入 `Sampler._softmax`，说明为什么要先减去 row max。
3. 打开 `course_ops.cu` 的 softmax kernel，讲 row-wise reduction 的输入输出。
4. 解释 temperature、top-k、top-p 在概率分布上的效果。
5. 用测试说明随机逻辑如何固定 seed 和断言概率性质。

学生任务：

- `TODO(lab06)` 在 `Sampler._softmax` 和 `course_ops.cu` 的 softmax kernel。
- 输入 shape：`logits: [batch, vocab]`。
- 输出 shape：`probs: [batch, vocab]`，每行和约为 1。
- 推荐步骤：实现稳定 PyTorch softmax，再接 row-wise CUDA softmax，最后确认采样测试通过。

运行方式：

```bash
python -m pytest -q \
  tests/test_kernels.py::test_cuda_softmax_matches_torch \
  tests/test_sampler.py -rs
```

验收：

```bash
python -m course_vllm.benchmarks.grader week06
```

## Week 07：Attention、causal prefill 与 paged decode

本周是前半学期的核心。学生要理解 prefill attention 和 decode attention 的 shape 差异，并第一次接触 paged KV decode。

课堂打开这些文件：

- `course_vllm/model/attention.py`
- `course_vllm/model/ops.py`
- `course_vllm/model/qwen3_torch.py`
- `kernels/course_ops.cu`
- `tests/test_attention.py`
- `docs/labs/week07_attention.md`

代码讲解顺序：

1. 从 `dense_attention_prefill_reference` 讲 Q/K/V 的常规 attention：score、mask、softmax、weighted sum。
2. 讲 causal mask 为什么只在 prefill 多 token 时显式出现。
3. 进入 `dense_attention_prefill`，说明课程实现如何在 reference 和 CUDA 之间 dispatch。
4. 打开 `paged_attention_decode_reference`，说明 decode 时 query 是当前 token，K/V 来自历史 block table。
5. 进入 `course_ops.cu` 的 paged decode attention kernel，讲 block table 到物理 KV slot 的映射。

学生任务：

- `TODO(lab07)` 在 `dense_attention_prefill`、`paged_attention_decode` 和 `course_ops.cu` attention kernel。
- Dense prefill 常见输入 shape：`q,k,v: [batch, heads, seq, head_dim]`，输出同 `q`。
- Paged decode 常见输入 shape：`q: [batch, heads, head_dim]`，`k_cache/v_cache` 是按物理 block 存储的 cache，输出 `[batch, heads, head_dim]`。
- 推荐步骤：先写 dense causal prefill，再写 paged decode reference dispatch，最后实现 CUDA smoke。

运行方式：

```bash
python -m pytest -q tests/test_attention.py -rs
python -m course_vllm.benchmarks.grader week07
python -m course_vllm.benchmarks.grader cuda_smoke
```

## Week 08：连续 KV cache

本周从 attention 算法进入 serving 状态管理。连续 KV cache 是最直观的实现：每个 sequence 保存每层 K/V，并沿 sequence 维 append。

课堂打开这些文件：

- `course_vllm/engine/kv_cache.py`
- `course_vllm/model/qwen3_torch.py`
- `course_vllm/engine/engine.py`
- `tests/test_kv_cache.py`
- `tests/test_qwen3_torch.py`
- `docs/labs/week08_kv_cache.md`

代码讲解顺序：

1. 打开 `ContinuousKVCache`，说明内部字典如何按 `sequence_id` 和 `layer_idx` 保存 K/V。
2. 讲 `append` 的输入输出：新 K/V 沿 seq 维拼到历史 K/V 后面。
3. 讲 `release` 为什么必须释放结束请求的 cache。
4. 回到 `Qwen3TorchBackend._store_cache` 和 `_load_cache`，说明模型层如何读写 cache。
5. 在 `Engine.generate_stream` 中指出 decode loop 如何复用历史 KV。

学生任务：

- `TODO(lab08)` 在 `ContinuousKVCache.append` 和 `ContinuousKVCache.release`。
- 单层 K/V shape 与模型实现一致，关键要求是历史长度维递增。
- 推荐步骤：先处理首次写入，再处理后续 concat，最后实现按 sequence 释放。

运行方式：

```bash
python -m pytest -q tests/test_kv_cache.py tests/test_qwen3_torch.py
python -m course_vllm.benchmarks.grader week08
```

## Week 09：推理引擎、请求状态与停止条件

本周把模型 forward、KV cache、sampling 串成一个完整 engine。重点是 request/sequence 状态机、流式输出和 finish reason。

课堂打开这些文件：

- `course_vllm/engine/request.py`
- `course_vllm/engine/engine.py`
- `course_vllm/engine/sampler.py`
- `tests/test_engine.py`
- `tests/test_chat_client.py`
- `docs/labs/week09_engine.md`

代码讲解顺序：

1. 从 `Request` 和 `Sequence` 看 prompt token、generated token、状态和 finish reason。
2. 讲 `Sequence.append_token`：一个 decode step 产生一个 token，必须更新 token 列表和文本增量。
3. 讲 `Sequence.finish`：停止 sequence，同时让 request 可被上层收尾。
4. 进入 `Engine.generate_stream`，把 prefill、decode、sample、append、finish 串成一张时序图。
5. 打开 HTTP streaming 测试，说明 engine yield 的内容如何变成服务端事件。

学生任务：

- `TODO(lab09)` 在 `Sequence.append_token` 和 `Sequence.finish`。
- 输入：一个新 token id、可选文本增量、finish reason。
- 输出：更新后的 sequence/request 状态。
- 推荐步骤：先维护 token 列表和长度，再处理 finished 状态，最后跑 streaming 测试。

运行方式：

```bash
python -m pytest -q tests/test_engine.py tests/test_chat_client.py
python scripts/profile/torch_profiler.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --workload decode \
  --max-tokens 8
python -m course_vllm.benchmarks.grader week09
```

## Week 10：Paged KV cache 与 block manager

本周进入后半学期主线：用固定大小物理 block 管理 KV cache，避免连续 cache 带来的搬移和碎片问题。

课堂打开这些文件：

- `course_vllm/engine/block_manager.py`
- `course_vllm/engine/paged_kv_cache.py`
- `course_vllm/model/qwen3_paged_backend.py`
- `examples/block_usage.py`
- `tests/test_block_manager.py`
- `tests/test_paged_kv_cache.py`
- `docs/labs/week10_paged_kv.md`

代码讲解顺序：

1. 先运行 `examples/block_usage.py`，让学生看到 logical token 到 physical slot 的映射。
2. 打开 `BlockManager.ensure_capacity`，讲一个 sequence 需要多少 block。
3. 打开 `slot_mapping`，讲 logical position 如何映射到 `(block_id, offset)`。
4. 打开 `_allocate_with_prefix_cache`，讲前缀缓存如何复用完整 prompt block。
5. 进入 `PagedKVCache.reserve/write`，说明 block table 最终如何服务 attention decode。

学生任务：

- `TODO(lab10)` 在 `BlockManager.ensure_capacity`、`slot_mapping`、`_allocate_with_prefix_cache`、`PagedKVCache.reserve`、`PagedKVCache.write`。
- 输入：sequence id、token position/new length、每层 K/V。
- 输出：可用于 attention 的 block table 和写入完成的物理 KV cache。
- 推荐步骤：先不考虑 prefix cache 跑通分配和写入，再补引用计数和 prefix reuse。

运行方式：

```bash
python examples/block_usage.py \
  --num-blocks 8 \
  --block-size 4 \
  --prompt-lens 3,6,9 \
  --decode-steps 2
```

```bash
python -m pytest -q tests/test_block_manager.py tests/test_paged_kv_cache.py
python -m course_vllm.benchmarks.grader week10
```

## Week 11：连续批处理与调度

本周讲 serving 系统中最关键的吞吐优化：prefill 和 decode 请求不断进入，scheduler 每轮决定谁进 batch、谁等待、谁继续 decode。

课堂打开这些文件：

- `course_vllm/engine/scheduler.py`
- `course_vllm/server/batching.py`
- `course_vllm/engine/engine.py`
- `tests/test_scheduler.py`
- `tests/test_server_batching.py`
- `docs/labs/week11_continuous_batching.md`

代码讲解顺序：

1. 从 `Scheduler.add` 看新 sequence 如何进入 waiting queue。
2. 讲 `_schedule_prefill`：在 max sequences 和 max tokens 预算下选 prompt。
3. 讲 `_schedule_decode`：running sequence 每轮通常只 decode 一个 token。
4. 打开 `BatchingEngine`，说明 HTTP 请求不是直接立即 forward，而是进入服务端队列。
5. 讲 chunked prefill 和 cache-aware scheduling 是如何通过参数传入 engine 的。

学生任务：

- `TODO(lab11)` 在 `Scheduler.add`、`_schedule_prefill`、`_schedule_decode`。
- 输入：等待队列、运行队列、batch budget。
- 输出：本轮 prefill/decode batch。
- 推荐步骤：先实现 FIFO prefill，再实现 running decode，最后加入 token budget 限制。

运行方式：

```bash
python -m pytest -q tests/test_scheduler.py tests/test_server_batching.py
```

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week11 \
  --kernel-impl auto \
  --max-batch-size 8 \
  --batch-wait-ms 2 \
  --enable-chunked-prefill \
  --port 18080
```

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18080/generate \
  --requests 16 \
  --concurrency 4 \
  --prompt "Explain continuous batching." \
  --max-tokens 16
python -m course_vllm.benchmarks.grader week11
```

## Week 12：系统优化与服务 admission

本周讲工程化优化：CPU 到 GPU 拷贝、pinned memory、transfer stream、队列上限和 prompt 长度限制。重点是系统边界，不再只看单个模型算子。

课堂打开这些文件：

- `course_vllm/model/qwen3_continuous_backend.py`
- `course_vllm/server/batching.py`
- `course_vllm/benchmarks/system_optimization.py`
- `tests/test_server_batching.py`
- `docs/labs/week12_system_optimization.md`
- `docs/reports/week12_system_optimization_template.md`

代码讲解顺序：

1. 打开 `_to_device`，讲普通 `.to(device)`、pinned memory 和 non-blocking copy 的关系。
2. 讲 transfer stream 为什么可以和 compute stream 做一定重叠。
3. 打开 `BatchingEngine._admit`，讲服务端 admission control：prompt 长度、队列深度。
4. 运行 `system_optimization.py`，比较开关前后的延迟或吞吐。
5. 打开报告模板，要求学生说明优化是否真的有效，而不是只贴命令。

学生任务：

- `TODO(lab12)` 在 `_to_device` 和 `BatchingEngine._admit`。
- 输入：CPU tensor 或请求对象、服务端限制参数。
- 输出：目标设备 tensor，或 admission accept/reject 结果。
- 推荐步骤：先保证功能正确，再打开 pinned memory，再加入 transfer stream，最后补 admission 测试。

运行方式：

```bash
python -m pytest -q tests/test_server_batching.py
python -m course_vllm.benchmarks.system_optimization \
  --pinned-memory \
  --transfer-stream \
  --max-queue-size 128 \
  --max-prompt-chars 8192 \
  --batch-wait-ms 2 \
  --max-batch-size 8
python -m course_vllm.benchmarks.grader week12
```

## Week 13：多卡推理容量规划

本周主要是系统估算和报告，不要求学生改核心 engine。目标是让学生用模型规模、KV cache、并行策略和通信量估算一台或多台机器能服务多大 workload。

课堂打开这些文件：

- `course_vllm/benchmarks/capacity_planner.py`
- `tests/test_benchmarks.py`
- `docs/labs/week13_capacity_planning.md`
- `docs/reports/week13_capacity_planning_template.md`

代码讲解顺序：

1. 从 `capacity_planner.py` 的参数看模型层数、hidden size、head 数、dtype bytes 如何进入估算。
2. 讲权重显存、KV cache 显存、activation/overhead 的区别。
3. 讲 tensor parallel 的 all-reduce、pipeline parallel 的 bubble、context parallel 的 pass-Q/pass-KV。
4. 打开测试，说明报告里的关键字段必须稳定输出。
5. 让学生用同一个 workload 改 TP/PP/CP 参数，比较瓶颈变化。

学生任务：

- 不以补代码为主，重点是改参数、跑估算、写容量规划报告。
- 需要说明最大 batch、最大上下文长度、显存余量和通信代价。

运行方式：

```bash
python -m course_vllm.benchmarks.capacity_planner \
  --gpu-memory-gb 24 \
  --weight-memory-gb 1.4 \
  --num-layers 28 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --dtype bfloat16 \
  --block-size 16 \
  --max-model-len 2048 \
  --hidden-size 1024 \
  --tp 2 \
  --pp 1 \
  --cp 1 \
  --target-concurrency 8 \
  --target-sequence-len 2048 \
  --report
```

验收：

```bash
python -m pytest -q tests/test_benchmarks.py
python -m course_vllm.benchmarks.grader week13
```

## Week 14：AscendC 暂缓周

当前工程没有放 AscendC 本地后端和样例，因为课程决策是等硬件/后端条件确认后再补真实实现。本周可以作为 CUDA 到异构后端的迁移讨论课，或者留给前面实验补交。

课堂打开这些文件：

- `docs/labs/week14_ascend_deferred.md`
- `docs/labs/README.md`

建议讲解内容：

- 为什么不能在没有真实后端和可验证硬件时放“假 AscendC”代码。
- 如果后续补 AscendC，应该遵守同样结构：reference path、kernel path、pytest、grader、报告。
- 让学生回顾 Week03 到 Week07 的 CUDA kernel 接入路径，思考迁移到另一套后端需要改哪些层。

运行方式：

```bash
python -m course_vllm.benchmarks.grader week13
python -m course_vllm.benchmarks.grader week15
```

第 14 周本身不设独立自动 grader。

## Week 15：前沿专题与 cache-aware policy

本周把已有系统能力抽象成策略讨论：prefix cache、chunked prefill、cache-aware scheduling 等机制对真实 workload 的影响。

课堂打开这些文件：

- `course_vllm/engine/policies.py`
- `course_vllm/benchmarks/cache_aware_demo.py`
- `course_vllm/engine/block_manager.py`
- `course_vllm/engine/scheduler.py`
- `docs/labs/week15_frontier.md`
- `docs/reports/week15_paper_to_system_template.md`

代码讲解顺序：

1. 从 `policies.py` 看策略函数如何把请求属性转成排序或优先级。
2. 回到 `block_manager.py`，说明 prefix cache 是策略能发挥作用的底层条件。
3. 回到 `scheduler.py`，说明策略最终要影响 batch 选择。
4. 运行 `cache_aware_demo.py`，用同一组请求比较不同 policy 的命中率和等待。
5. 让学生选一篇 serving 论文，把论文机制映射到本工程可落地的位置。

学生任务：

- 不强制补核心代码，重点是实现或解释一个 policy，并写“论文机制 -> 本工程模块”的映射。
- 如果要扩展代码，优先放在 `policies.py` 和对应 demo，不直接大改 engine。

运行方式：

```bash
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "cache-aware serving" \
  --prompts "1,2,3,4|1,2,3,9|8,7|1,2,5"
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "prefill-decode disaggregation" \
  --requests "128:16|2048:8|256:64|1024:12"
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "tokendance-style scheduling" \
  --requests "128:16|2048:8|256:64|1024:12"
python -m course_vllm.benchmarks.grader week15
```

## Week 16：总复盘与最终展示

本周不再引入新模块，而是把学生补过的算子、KV cache、engine、paged KV、batching、system optimization 串成一个可展示的 serving project。

课堂打开这些文件：

- `README.md`
- `docs/project_usage_guide.md`
- `docs/code_walkthrough.md`
- `docs/runnable_validation_guide.md`
- `docs/reports/week16_final_report_template.md`
- 学生自己改过的 `TODO(labXX)` 文件

代码讲解顺序：

1. 让学生从 README 的四个开关复述整个项目如何切换路径。
2. 从 HTTP 请求入口重新追一遍：`api.py -> batching.py -> engine.py -> model backend -> kernels/cache`。
3. 对照 `code_walkthrough.md`，让学生说明自己实现过的模块在完整调用链中的位置。
4. 运行最终正确性和 smoke test。
5. 最终展示时要求给出一个可复现命令，而不是只展示截图。

学生任务：

- 整理所有实验代码。
- 跑通最终测试。
- 写最终报告：工程结构、实现内容、性能观察、问题和改进方向。

运行方式：

```bash
python -m pytest -q -rs
python -m course_vllm.benchmarks.grader cuda_smoke
```

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week16 \
  --kernel-impl auto \
  --prompt "Summarize LLM serving in one sentence." \
  --max-tokens 32 \
  --temperature 0
```

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week16 \
  --kernel-impl auto \
  --max-batch-size 8 \
  --batch-wait-ms 2 \
  --enable-chunked-prefill \
  --cache-aware-scheduling \
  --pinned-memory \
  --transfer-stream \
  --port 18080
```

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18080/generate \
  --requests 16 \
  --concurrency 4 \
  --prompt "Explain paged KV cache." \
  --max-tokens 16
```

当前版本没有单独注册 `week16` grader，最终周用下面的组合验收：

```bash
python -m pytest -q -rs
python -m course_vllm.benchmarks.grader week11
python -m course_vllm.benchmarks.grader week12
python -m course_vllm.benchmarks.grader week15
python -m course_vllm.benchmarks.grader cuda_smoke
```

## 学生补代码量的大致分布

学生真正需要补的核心代码集中在 Week03 到 Week12：

| 周次 | 主要 TODO | 代码量级 |
| --- | --- | --- |
| Week03 | vector add CUDA kernel | 5 到 10 行 |
| Week04 | RMSNorm、RoPE、dispatch | 40 到 80 行 |
| Week05 | matmul、tiled matmul、CourseLinear | 60 到 120 行 |
| Week06 | stable softmax、Sampler dispatch | 30 到 70 行 |
| Week07 | dense/paged attention | 80 到 160 行 |
| Week08 | ContinuousKVCache append/release | 20 到 50 行 |
| Week09 | Sequence 状态更新和 finish | 20 到 40 行 |
| Week10 | block manager、paged KV reserve/write | 80 到 160 行 |
| Week11 | scheduler prefill/decode | 50 到 100 行 |
| Week12 | device copy 优化、admission control | 30 到 70 行 |

总量大约是 400 到 800 行，取决于学生是否把 CUDA kernel、fallback、边界检查和注释写完整。Week01、Week02、Week13、Week15、Week16 更偏运行、分析和报告，Week14 当前暂缓。

## 发布学生版前的检查

发布前建议教师在 `main` 上重新生成并检查 `student` 分支：

```bash
python scripts/validation/generate_student_branch.py --branch student
git switch student
rg -n "TODO\\(lab" course_vllm kernels docs README.md
python -m course_vllm.benchmarks.grader week01
git switch main
```

如果要把学生版推到远端：

```bash
git push -u origin student
```

发布说明里要明确：

- 学生从 `student` 分支开始。
- 完整答案在 `main`，不要提前开放给学生。
- 每周先读 `docs/labs/weekXX_*.md`，再补对应 `TODO(labXX)`。
- 每周交付以对应 pytest/grader 和报告模板为准。
