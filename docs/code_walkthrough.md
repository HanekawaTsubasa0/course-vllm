# Course LLM Serving 代码全景讲解

本文档按代码结构讲清楚 course LLM serving 工程。它面向维护者、助教和需要改代码的学生。使用命令见 `docs/project_usage_guide.md`。

## 1. 总体架构

工程的主链路是：

```text
HTTP API / offline script
  -> Engine
  -> Scheduler
  -> Model backend
  -> Qwen3 model forward
  -> KV cache
  -> Sampler
  -> response / stream event
```

核心分层：

```text
course_vllm/server      HTTP API、请求协议、HTTP batching
course_vllm/engine      请求生命周期、scheduler、sampler、KV cache
course_vllm/model       Qwen3、HF oracle、dense/paged backend、attention
course_vllm/kernels     Python 侧 CUDA extension wrapper
kernels                 C++/CUDA kernel 源码
benchmarks              压测、容量规划、前沿 demo、grader
examples                离线生成、chat client、block usage demo
validation              和 HuggingFace 对齐
tests                   单元测试和集成测试
docs                    课程文档、报告模板、运行手册
```

## 2. 运行时概念

### backend

```text
reference  HuggingFace/PyTorch oracle
course     课程自有推理 engine
```

`reference` 用于 correctness 对齐，不是课程主线。`course` 才是学生学习的服务系统。

### kv-mode

```text
dense  连续 KV cache
paged  paged KV cache
```

KV 形态不是 backend，而是 course backend 内部的实现模式。

### kernel-impl

```text
torch  只走 PyTorch/reference
auto   CUDA tensor 上优先走课程 CUDA kernel，失败回退
cuda   强制走课程 CUDA kernel，失败直接报错
```

### stage

`stage` 是课程周次标记。它不会强制禁用功能，但会出现在 `/health`，用于课堂演示和报告记录。

## 3. 顶层文件

### `README.md`

项目入口文档。说明项目定位、安装、快速运行、HTTP API、benchmark、profiling、正确性验证、功能说明。

### `pyproject.toml`

Python 包配置。主要内容：

```text
project name: course-vllm
dependencies: torch, transformers, fastapi, uvicorn, httpx, pydantic, safetensors, ninja
pytest 配置
setuptools package include
```

### `docs/`

课程和维护文档。最重要的是：

```text
docs/project_usage_guide.md        项目怎么用
docs/code_walkthrough.md           代码全景讲解
docs/labs/                         每周 lab handout
docs/course_teaching_runbook.md    教师/助教逐周运行手册
docs/runnable_validation_guide.md  可复现验收指南
docs/student_branch_plan.md        学生分支生成计划
```

## 4. 包入口

### `course_vllm/__init__.py`

对外懒加载：

```python
from course_vllm import Engine, SamplingParams
```

这样 `examples/offline_generate.py` 可以直接从包入口拿到主要类。

## 5. Server 层

Server 层负责把 HTTP 请求变成 engine 调用。

### `course_vllm/server/protocol.py`

定义 Pydantic 请求/响应模型：

```text
SamplingParamsRequest
GenerateRequest
GenerateResponse
ChatMessage
ChatCompletionRequest
```

`GenerateRequest` 包含：

```text
prompt: str
stream: bool
sampling_params: SamplingParamsRequest
```

`ChatCompletionRequest` 包含 OpenAI 风格的 `messages`。

### `course_vllm/server/api.py`

FastAPI 入口。主要函数：

```python
create_app(...)
main()
```

`create_app` 做这些事：

1. 创建 `Engine`
2. 创建 `BatchingEngine`
3. 注册 lifespan，启动/停止 batching worker
4. 注册 `/health`
5. 注册 `/generate`
6. 注册 `/v1/chat/completions`

`/health` 返回：

```text
status
model
backend
kv_mode
engine.info()
max_batch_size
max_batched_tokens
batching stats
```

`/generate`：

```text
stream=false -> await batching_engine.generate(...)
stream=true  -> StreamingResponse(_sse(batching_engine.stream(...)))
```

`/v1/chat/completions`：

1. 把 messages 交给 tokenizer chat template
2. 得到 prompt
3. 复用 `/generate` 的 batching/streaming 路径

### `course_vllm/server/batching.py`

HTTP batching 层。关键类：

```python
BatchingEngine
BatchingStats
```

`BatchingEngine` 的目的：

- non-streaming 请求进入 async queue
- 等待 `batch_wait_ms`
- 采样参数一致时合批
- 把模型执行放进单独 model worker thread
- 避免模型执行阻塞 FastAPI event loop

关键字段：

```text
_queue          async request queue
_pending        被采样参数不一致等原因退回的请求
_model_queue    交给 model worker thread 的同步 queue
_model_thread   单独模型执行线程
stats           total_requests / total_batches / max_observed_batch_size
```

关键方法：

```python
start()
stop()
generate()
stream()
_run()
_collect_batch()
_process_batch()
_admit()
_run_model()
_run_model_worker()
```

`_collect_batch` 的合批条件：

```text
batch size 未达 max_batch_size
队列里还有请求
sampling_params 与当前 batch 一致
```

`_admit` 做请求准入：

```text
max_prompt_chars
max_queue_size
```

## 6. Engine 层

Engine 层负责请求生命周期、调度、prefill/decode、采样、cache 释放。

### `course_vllm/engine/request.py`

定义：

```python
RequestStatus
Request
Sequence
```

`Request` 是用户请求元数据：

```text
prompt
sampling_params
request_id
created_at
status
```

`Sequence` 是模型生成状态：

```text
prompt_token_ids
generated_token_ids
past_key_values
next_token_id
finish_reason
prefill_offset
scheduled_start
scheduled_end
```

常用方法：

```python
token_ids
append_token()
scheduled_prompt_tokens()
prefill_complete()
reached_max_tokens()
finish()
```

### `course_vllm/engine/engine.py`

主入口类：

```python
Engine
```

#### backend 选择

`normalize_backend_config` 处理新旧参数：

```text
hf -> reference,dense
paged -> course,paged
reference -> reference,dense
course + dense/paged -> course,dense/paged
```

`Engine.__init__` 里按配置创建：

```text
HFModelBackend          reference
Qwen3TorchBackend       course + dense
Qwen3PagedBackend       course + paged
```

#### 单请求生成

`generate(prompt)` 调用 `generate_stream(prompt)`，把 token event 拼成最终文本。

`generate_stream` 流程：

```text
1. prompt -> token ids
2. 创建 Request / Sequence
3. backend.prefill(prompt_token_ids)
4. sampler.sample(logits) 得到 next_token_id
5. 循环：
   - seq.append_token(next_token_id)
   - yield token event
   - 检查 eos / stop / max_tokens
   - backend.decode_step(next_token_id, past_key_values)
   - sample 下一个 token
6. finally release cache
```

#### Batch 生成

`generate_batch(prompts)` 使用 `Scheduler`：

```text
1. encode 所有 prompt
2. 可选 cache-aware scheduling 重排
3. 每个 prompt 创建 Sequence
4. scheduler.add(seq)
5. while scheduler.has_unfinished():
   - scheduler.schedule()
   - PREFILL: _prefill_batch()
   - DECODE: 接受 next token，_decode_batch()
6. 释放 cache
7. 按原始请求顺序返回结果
```

`_prefill_batch` 优先用 backend 的 `prefill_batch`，没有则逐个 prefill。

`_decode_batch` 优先用 backend 的 `decode_batch`，没有则逐个 decode_step。

### `course_vllm/engine/scheduler.py`

教学版 scheduler。核心类型：

```python
BatchKind.PREFILL
BatchKind.DECODE
ScheduledBatch
Scheduler
```

字段：

```text
waiting
running
max_num_seqs
max_num_batched_tokens
enable_chunked_prefill
```

调度策略：

```text
优先调 prefill
没有 prefill 再调 decode
prefill 受 token budget 限制
decode 每条 running sequence 取一个 token
```

`_schedule_prefill` 做：

1. 从 waiting 队列取 sequence
2. 计算 prompt 剩余 token
3. 检查 token budget
4. 如果开启 chunked prefill，可以只调一段 prompt
5. prefill 完整后放入 running

`_schedule_decode` 做：

```text
从 running 中取最多 max_num_seqs 条 RUNNING sequence
每条 decode 1 个 token
```

### `course_vllm/engine/sampler.py`

采样逻辑。

`SamplingParams`：

```text
temperature
max_tokens
top_k
seed
stop_token_ids
```

`Sampler.sample(logits)`：

```text
temperature == 0 -> argmax
temperature > 0  -> softmax + multinomial
top_k != None    -> 先 topk，再 softmax，再采样
```

`_softmax` 在 CUDA tensor 上尝试 `cuda_softmax`，不可用时回退 PyTorch。

## 7. KV Cache

### `course_vllm/engine/kv_cache.py`

连续 KV cache。

关键类型：

```python
KVCacheHandle(seq_id, seq_len)
LayerKV(key, value)
ContinuousKVCache
```

存储结构：

```python
dict[(seq_id, layer_id)] -> LayerKV
```

`append`：

```text
第一次写入：clone key/value
后续写入：在 token sequence 维度拼接
```

KV shape：

```text
[batch=1, kv_heads, tokens, head_dim]
```

拼接维度是 `-2`，也就是 tokens 维。

### `course_vllm/engine/block_manager.py`

paged KV 的 block 管理器。

核心类型：

```python
Block
BlockTable
BlockManager
```

`Block`：

```text
block_id
ref_count
token_hash
token_ids
```

`BlockTable`：

```text
block_size
block_ids
owned_block_ids
length
```

`BlockManager` 维护：

```text
blocks
free_block_ids
tables: seq_id -> BlockTable
hash_to_block_id
```

关键方法：

```python
allocate()
ensure_capacity()
append_tokens()
block_table()
slot_mapping()
release()
usage_stats()
```

`slot_mapping(seq_id, positions)` 把逻辑 token 位置映射成物理 slot：

```text
block_index = position // block_size
block_offset = position % block_size
slot = block_id * block_size + block_offset
```

prefix cache 是 teaching approximation：

```text
只复用完整 block
用 token hash 找可复用 block
用 ref_count 维护共享
不实现生产级 eviction
```

### `course_vllm/engine/paged_kv_cache.py`

物理 slot KV cache。

`PagedKVConfig`：

```text
num_layers
num_blocks
block_size
num_kv_heads
head_dim
dtype
device
```

物理 cache shape：

```text
key_cache/value_cache:
[num_layers, num_blocks * block_size, num_kv_heads, head_dim]
```

关键方法：

```python
allocate()
append()
reserve()
write()
get_dense()
block_table()
slot_mapping()
release()
usage_stats()
```

`write` 接收模型输出的 K/V：

```text
input: [1, kv_heads, tokens, head_dim]
内部转成: [tokens, kv_heads, head_dim]
按 slot 写入物理 cache
```

`get_dense` 用 block table 把物理 slot 读回连续 K/V，主要用于 correctness 和 debug。

## 8. Model 层

### `course_vllm/model/types.py`

定义模型输出结构：

```python
ModelOutput(logits, past_key_values)
BatchModelOutput(logits, past_key_values)
```

`parse_dtype` 把字符串转成 torch dtype：

```text
auto
float32
float16
bfloat16
```

### `course_vllm/model/hf_backend.py`

HuggingFace oracle backend。

用途：

```text
reference backend
正确性对齐
学生 starter 早期可运行路径
```

提供和 course backend 类似的接口：

```python
encode()
decode()
apply_chat_template()
prefill()
decode_step()
prefill_batch()
decode_batch()
release_cache()
```

### `course_vllm/model/qwen3_torch.py`

课程自有 Qwen3 PyTorch 实现。

主要类型：

```python
Qwen3Config
Qwen3KVCache
Qwen3RMSNorm
Qwen3RotaryEmbedding
Qwen3MLP
Qwen3Attention
Qwen3DecoderLayer
Qwen3Model
Qwen3ForCausalLM
```

#### `Qwen3Config`

从 `config.json` 读取：

```text
vocab_size
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
max_position_embeddings
rms_norm_eps
rope_theta
attention_bias
hidden_act
```

#### `Qwen3RMSNorm`

数学：

```text
y = x / sqrt(mean(x^2) + eps) * weight
```

CUDA 可用时通过 `cuda_rms_norm` dispatch。

#### RoPE

`rotate_half(x)`：

```text
[x1, x2] -> [-x2, x1]
```

`apply_rotary_pos_emb(q, k, cos, sin)`：

```text
q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
```

CUDA 可用时走 `cuda_rope`。

#### `Qwen3Attention`

流程：

```text
x
 -> q_proj/k_proj/v_proj
 -> reshape heads
 -> q_norm/k_norm
 -> RoPE
 -> concat past KV if present
 -> repeat_kv for GQA
 -> attention
 -> o_proj
```

prefill：

```text
past_key_value is None
走 dense_attention_prefill
```

decode：

```text
past_key_value exists
seq_len == 1 且 mask 不需要时走 dense_attention_decode
否则走 PyTorch masked attention
```

#### `Qwen3ForCausalLM`

包装：

```text
Qwen3Model
lm_head
from_pretrained
forward
forward_with_cache
```

`from_pretrained` 负责：

1. 解析本地模型路径
2. 读取 `config.json`
3. 创建模型
4. 加载 safetensors 权重
5. 做必要的权重名映射

### `course_vllm/model/ops.py`

放模型层可复用算子。

#### `CourseLinear`

替代普通 `nn.Linear`，增加 CUDA matmul dispatch：

```text
kernel_impl=torch -> F.linear
kernel_impl=auto/cuda 且 x.is_cuda -> cuda_matmul_tiled
```

输入 shape：

```text
x: [..., in_features]
weight: [out_features, in_features]
output: [..., out_features]
```

#### dense attention reference

`dense_attention_prefill_reference`：

```text
Q/K/V shape: [batch, heads, seq, dim]
causal attention
online softmax 风格逐 tile 累积
```

`dense_attention_decode_reference`：

```text
query: [batch, heads, dim]
key/value: [batch, heads, seq, dim]
```

### `course_vllm/model/attention.py`

paged attention decode。

主函数：

```python
paged_attention_decode()
paged_attention_decode_reference()
```

输入：

```text
query: [batch, heads, dim]
key_cache/value_cache: [slots, kv_heads, dim]
block_tables: [batch, max_blocks]
context_lens: [batch]
block_size: int
```

流程：

1. 校验 shape/device/head 数量
2. CUDA tensor 上尝试 `cuda_paged_attention_decode`
3. 不可用时走 reference

reference 路径：

```text
每条 sequence:
  positions -> slots
  index_select 读 K/V
  repeat_kv 处理 GQA
  scores = Q @ K^T
  softmax
  output = weights @ V
```

### `course_vllm/model/qwen3_continuous_backend.py`

dense KV runner。

职责：

```text
加载 tokenizer
加载 Qwen3ForCausalLM
维护 ContinuousKVCache
实现 prefill/decode/batch prefill/batch decode
处理 pinned memory 和 transfer stream
```

关键方法：

```python
prefill()
prefill_chunk()
prefill_batch()
decode_step()
decode_batch()
release_cache()
_store_cache()
_load_cache()
_load_batch_cache()
_to_device()
```

`_store_cache` 把模型返回的 `Qwen3KVCache` 写入 `ContinuousKVCache`，并返回 `KVCacheHandle`。

`decode_batch` 会按历史长度分桶，因为 dense KV 需要同长度才能拼成 batch tensor。

### `course_vllm/model/qwen3_paged_backend.py`

paged KV runner，继承 dense runner。

差异：

```text
kv_cache 换成 PagedKVCache
decode_step 通过 decode_batch 实现
decode_batch 支持不同 context length
_decode_paged_batch 手写逐层 decode forward
_store_cache 写物理 slot
_load_cache 从 paged KV 读回 dense KV
```

`_decode_paged_batch` 的核心流程：

```text
input token -> embedding
position_ids = handle.seq_len
计算 RoPE cos/sin
为每条 seq reserve 1 个新 token slot
每层：
  RMSNorm
  q/k/v projection
  q/k norm
  RoPE
  写新 K/V 到 PagedKVCache
  paged_attention_decode 读取完整上下文
  o_proj + residual
  MLP + residual
final norm + lm_head
```

## 9. CUDA Extension

### `course_vllm/kernels/harness.py`

负责 JIT 编译 extension。

关键函数：

```python
load_cuda_extension(name, sources)
benchmark_cuda(fn, warmup, repeat)
assert_close(name, actual, expected, atol, rtol)
```

它会尝试使用本地可用的 GCC/CUDA 编译配置，编译 `kernels/*.cu`。

### `course_vllm/kernels/cuda_ops.py`

Python wrapper。对上层暴露：

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

职责：

1. lazy load CUDA extension
2. 做 Python 侧 shape/dtype/device 准备
3. 把 Python tensor 传给 C++ binding
4. 将 kernel unavailable 转成 `KernelUnavailable`

### `course_vllm/kernels/errors.py`

定义：

```python
KernelUnavailable
```

用于区分“CUDA kernel 不可用”和普通逻辑错误。`auto` 模式会捕获它并 fallback。

### `kernels/course_ops.cpp`

C++ binding。

职责：

```text
检查 tensor 是否 CUDA/contiguous/floating point
检查 shape/dtype
调用 extern "C" launcher
把函数注册到 PyTorch extension module
```

每个 Python wrapper 最终会进这里。

### `kernels/course_ops.cu`

CUDA kernel 实现。

包含：

```text
softmax_kernel
rms_norm_kernel
rope_kernel
matmul_kernel
matmul_tiled_kernel
dense_attention_prefill_kernel
dense_attention_decode_kernel
paged_attention_decode_kernel
各 launcher
```

约定：

```text
dtype code 0 = float32
dtype code 1 = float16
dtype code 2 = bfloat16
```

`to_float` / `from_float` 负责不同 dtype 和 float 之间转换。

## 10. Benchmarks

### `course_vllm/benchmarks/bench_server.py`

HTTP 压测工具。

流程：

```text
创建 httpx.AsyncClient
按 concurrency 发请求
记录每个请求 latency 和 output token 数
summarize 统计吞吐和延迟分位
```

输出：

```text
requests_per_s
output_tokens_per_s
latency_avg_s
latency_p50_s
latency_p90_s
latency_p99_s
latency_max_s
estimated_tpot_s
```

### `course_vllm/benchmarks/capacity_planner.py`

Week13 容量规划工具。

单卡 KV 估算：

```text
usable_bytes = gpu_memory * utilization
reserved = weight_memory + safety
kv_budget = usable - reserved
block_bytes = 2 * layers * block_size * kv_heads * head_dim * dtype_bytes
num_blocks = kv_budget / block_bytes
token_slots = num_blocks * block_size
```

多卡理论模型：

```text
TP all-reduce bytes/token
PP layers/stage
PP bubble fraction
EP all-to-all bytes/token
CP pass-Q bytes/token
CP pass-KV bytes/prefill
communication time/token
```

### `course_vllm/benchmarks/system_optimization.py`

Week12 系统优化说明工具。

包含：

```text
SystemOptimizationConfig
estimate_overlap_plan
admission_decision
```

它不直接运行服务，而是输出优化配置和解释。

### `course_vllm/benchmarks/cache_aware_demo.py`

Week15 前沿专题 demo。

支持：

```text
cache-aware serving
prefill-decode disaggregation
tokendance-style scheduling
```

输出 paper-to-system 映射和最小定量指标。

### `course_vllm/benchmarks/grader.py`

按周测试入口。

`STAGE_TESTS` 映射：

```text
week01 -> baseline / protocol / server smoke
week03 -> vector add
week04 -> RMSNorm / RoPE
week05 -> matmul / CourseLinear
week06 -> softmax / sampler
week07 -> attention
week08 -> KV cache
week09 -> engine / chat client
week10 -> block manager / paged KV
week11 -> scheduler / server batching
week12 -> system optimization / admission
week13 -> capacity planner
week15 -> frontier demos
cuda_smoke -> strict CUDA kernel checks
```

## 11. Examples

### `examples/offline_generate.py`

离线生成入口。

流程：

```text
parse args
Engine(...)
SamplingParams(...)
single prompt -> engine.generate/chat
multiple prompts -> engine.generate_batch
print text
```

### `examples/chat_client.py`

交互式 HTTP client。

功能：

```text
发送 /v1/chat/completions
stream/non-stream 切换
sampling 参数调整
system prompt
history
save/load
health check
```

### `examples/block_usage.py`

演示 BlockManager 如何分配 block、追加 decode token、统计碎片。

## 12. Validation

### `validation/compare_qwen3.py`

和 HuggingFace 做 logits 对齐。

模式：

```text
forward
decode
batch-prefill
batch-decode
```

输出：

```text
max_abs_diff
mean_abs_diff
```

用途：

```text
确认自有 Qwen3 权重加载正确
确认 prefill/decode 和 HF 行为一致
确认 paged KV decode 没有逻辑错误
```

## 13. Tests

测试是课程工程的重要部分。学生每周通过测试定位 TODO。

主要测试文件：

```text
tests/test_kernels.py          CUDA kernel correctness
tests/test_attention.py        dense/paged attention
tests/test_kv_cache.py         continuous KV
tests/test_block_manager.py    block allocation / prefix cache
tests/test_paged_kv_cache.py   physical slot KV
tests/test_scheduler.py        prefill/decode scheduler
tests/test_engine.py           generation lifecycle
tests/test_server_batching.py  HTTP batching
tests/test_server_api.py       FastAPI health/config
tests/test_benchmarks.py       benchmark/capacity/frontier helpers
tests/test_qwen3_torch.py      Qwen3 modules and backends
tests/test_sampler.py          sampling
tests/test_protocol.py         Pydantic protocol
tests/test_chat_client.py      CLI client
```

## 14. Student 分支

`student` 分支是从 `main` 自动生成的 starter 版本。

生成脚本：

```text
scripts/validation/generate_student_branch.py
```

挖空范围：

```text
lab03 vector add
lab04 RMSNorm / RoPE
lab05 matmul / CourseLinear
lab06 softmax / sampler
lab07 attention
lab08 continuous KV
lab09 request lifecycle
lab10 paged KV / block table
lab11 scheduler
lab12 pinned memory / admission
```

`student` 分支保留：

```text
reference backend
docs/labs
grader
benchmark
capacity planner
frontier demo
report templates
```

这样学生可以先运行 oracle，再逐周补 course 主线。

## 15. 典型调用链

### 离线单请求 course+paged

```text
examples/offline_generate.py
  -> Engine(...)
  -> Qwen3PagedBackend(...)
  -> Engine.generate()
  -> Engine.generate_stream()
  -> backend.prefill()
  -> Qwen3ForCausalLM.forward_with_cache()
  -> Qwen3Attention dense prefill
  -> Qwen3PagedBackend._store_cache()
  -> PagedKVCache.write()
  -> Sampler.sample()
  -> backend.decode_step()
  -> Qwen3PagedBackend.decode_batch()
  -> Qwen3PagedBackend._decode_paged_batch()
  -> paged_attention_decode()
  -> cuda_paged_attention_decode or reference
```

### HTTP non-streaming batch

```text
client POST /generate
  -> FastAPI route generate()
  -> BatchingEngine.generate()
  -> async queue
  -> BatchingEngine._collect_batch()
  -> model worker thread
  -> Engine.generate_batch()
  -> Scheduler.schedule()
  -> backend.prefill_batch/decode_batch
  -> result future
  -> GenerateResponse
```

### HTTP streaming

```text
client POST /generate stream=true
  -> FastAPI StreamingResponse
  -> BatchingEngine.stream()
  -> model worker runs Engine.generate_stream()
  -> event queue
  -> _sse()
  -> data: {...}
  -> data: [DONE]
```

## 16. 设计取舍

### 为什么不直接做工业级多进程

课程目标是让学生看懂 serving 主链路。多进程、ZMQ、分布式调度会显著增加理解成本，所以工程保留单进程主线和单 model worker。

### 为什么 reference 和 course 分开

`reference` 保证正确性 oracle 稳定；`course` 暴露学生要学习的系统结构。这样学生可以先跑通服务，再逐步替换 course 主线。

### 为什么 paged 是 kv-mode

paged KV 是 cache 实现机制，不应该是第三个 backend。现在公开概念是：

```text
reference: oracle
course: learning engine
kv-mode: dense or paged
```

### 为什么 CUDA kernel 只做教学版

课程 kernel 追求可读、可测、能接入主线，不追求超过 cuBLAS、FlashAttention 或 vLLM kernel。

### 为什么 Week13/15 多为工具和报告

TP/PP/EP/CP 和前沿 serving 机制完整实现成本过高。课程里先用理论模型和最小 demo 建立系统判断能力，再让学生说明如何映射到工程模块。

## 17. 修改代码时的建议顺序

如果要继续开发，建议按这个顺序：

1. 先跑对应测试，确认 baseline。
2. 找 `docs/labs/weekXX_*.md`。
3. 找 `course_vllm/benchmarks/grader.py` 中对应测试。
4. 只改该周列出的文件。
5. 跑单测。
6. 跑 week grader。
7. 必要时跑 integration 或 profiling。

不要一开始就改 API、模型权重加载、测试 expected 或 reference oracle。
