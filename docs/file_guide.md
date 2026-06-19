# 文件职责说明

这个文档说明 `course-vllm` 里每个源码文件的职责。运行方式没有因为 backend 拆分而变化：`course_vllm/model/qwen3_backend.py` 仍然保留兼容导出，所以原来的服务启动命令、离线生成命令和 CLI 客户端命令都继续可用。

## 运行入口

主要入口仍然是这些文件：

```text
python examples/offline_generate.py              离线单请求 / batch 生成
python -m course_vllm.server.api                 启动 HTTP 推理服务
python examples/chat_client.py                   连接服务端的交互式 CLI
python validation/compare_qwen3.py               和 HuggingFace 做正确性对齐
pytest -q                                        跑单元测试
```

服务端推荐 backend 仍然是：

```bash
python -m course_vllm.server.api \
  --model /home/wangqi/huggingface/Qwen3-0.6B \
  --backend paged \
  --dtype bfloat16 \
  --max-batch-size 8 \
  --batch-wait-ms 2 \
  --port 18080
```

## 顶层文件

| 文件 | 职责 |
| --- | --- |
| `README.md` | 项目主文档，包含环境配置、运行命令、接口说明、验证命令和功能总览。 |
| `pyproject.toml` | Python 包元数据、依赖范围、pytest 配置、setuptools 打包配置。 |
| `.gitignore` | 忽略虚拟环境、依赖缓存、编译产物、pytest cache、Python cache 等本地生成文件。 |

## `course_vllm/`

运行服务时真正加载的 Python 包。这里的代码是项目主体，也是“服务代码行数”统计的主要范围。

| 文件 | 职责 |
| --- | --- |
| `course_vllm/__init__.py` | 包入口，定义项目包。 |

## `course_vllm/engine/`

推理编排层，负责把请求、调度、采样、backend 调用和 cache 生命周期串起来。

| 文件 | 职责 |
| --- | --- |
| `course_vllm/engine/__init__.py` | engine 子包入口。 |
| `course_vllm/engine/engine.py` | 推理主入口。选择 `hf` / `course` / `paged` backend，实现 `generate`、`generate_stream`、`generate_batch`、`chat`、`chat_stream`，并负责 prefill/decode 循环、采样、停止条件和 KV cache 释放。 |
| `course_vllm/engine/request.py` | 请求和序列状态定义。包含 `Request`、`Sequence`，维护 prompt token、生成 token、finish reason、`max_tokens` 停止判断等。 |
| `course_vllm/engine/scheduler.py` | 教学版 batch scheduler。维护 waiting/running 序列，区分 prefill batch 和 decode batch，受 `max_num_seqs`、`max_num_batched_tokens` 限制。 |
| `course_vllm/engine/sampler.py` | 采样逻辑。定义 `SamplingParams`，实现 greedy、temperature、top-k、seed 控制。 |
| `course_vllm/engine/kv_cache.py` | 连续 KV cache。用于 `course` backend，直接按层保存 dense K/V 张量，便于理解 KV cache 基本机制。 |
| `course_vllm/engine/block_manager.py` | paged KV 的 block 管理器。负责分配、追加、释放固定大小 block，并维护 sequence 的 block table。 |
| `course_vllm/engine/paged_kv_cache.py` | paged KV cache 存储。把每层 K/V 写入物理 slot，并按 block table 读取逻辑上下文。 |

## `course_vllm/model/`

模型实现层，负责 tokenizer、模型 forward、attention、backend 适配和 HuggingFace 对齐参考。

| 文件 | 职责 |
| --- | --- |
| `course_vllm/model/__init__.py` | model 子包入口。 |
| `course_vllm/model/types.py` | 模型输出类型和 dtype 解析。定义 `ModelOutput`、`BatchModelOutput`、`parse_dtype`。 |
| `course_vllm/model/hf_backend.py` | HuggingFace 参考 backend。用于正确性 oracle，也可作为最稳定的推理路径。 |
| `course_vllm/model/qwen3_torch.py` | 课程自有 Qwen3 PyTorch 实现。包含权重加载、RMSNorm、RoPE、attention、MLP、decoder layer、causal LM forward，以及 HuggingFace safetensors 权重映射。 |
| `course_vllm/model/qwen3_continuous_backend.py` | 连续 KV cache backend。封装 tokenizer、chat template、prefill、decode step、batch prefill、batch decode，把 `Qwen3ForCausalLM` 接到 `ContinuousKVCache`。 |
| `course_vllm/model/qwen3_paged_backend.py` | paged KV cache backend。继承连续 backend 的公共能力，改用 `BlockManager` 和 `PagedKVCache`，在 decode 阶段通过 block table 走 paged attention。 |
| `course_vllm/model/qwen3_backend.py` | 兼容导出层。继续导出 `Qwen3TorchBackend` 和 `Qwen3PagedBackend`，保证旧 import 不需要修改。 |
| `course_vllm/model/attention.py` | attention 工具函数。包含 dense attention、paged attention PyTorch reference，以及满足条件时的 CUDA paged attention dispatch。 |

## `course_vllm/server/`

HTTP serving 层，负责 FastAPI 接口、请求协议、HTTP batching 和模型 worker 线程。

| 文件 | 职责 |
| --- | --- |
| `course_vllm/server/__init__.py` | server 子包入口。 |
| `course_vllm/server/api.py` | FastAPI 应用和命令行启动入口。提供 `/health`、`/generate`、`/v1/chat/completions`，支持 streaming SSE 和 non-streaming 响应。 |
| `course_vllm/server/batching.py` | HTTP batching engine。non-streaming 请求进入 async queue，按 `batch_wait_ms` 和 sampling 参数合批，再交给单独 model worker thread 执行，避免模型执行阻塞 FastAPI event loop。streaming 请求也通过同一个 model worker 串行执行。 |
| `course_vllm/server/protocol.py` | Pydantic 协议定义。描述 generate/chat 请求、message、sampling 参数和响应结构。 |

## `course_vllm/kernels/`

Python 侧 CUDA extension 封装。这里不写 kernel 本体，只负责 JIT 编译和 Python 调用。

| 文件 | 职责 |
| --- | --- |
| `course_vllm/kernels/__init__.py` | kernels 子包入口，导出 CUDA wrapper。 |
| `course_vllm/kernels/harness.py` | PyTorch CUDA extension JIT 加载器。自动寻找本地 GCC 14，编译 `kernels/course_ops.cpp` 和 `kernels/course_ops.cu`，并提供简单 benchmark helper。 |
| `course_vllm/kernels/cuda_ops.py` | Python wrapper。向上层暴露 `cuda_softmax`、`cuda_rms_norm`、`cuda_rope`、`cuda_matmul`、`cuda_paged_attention_decode`。 |

## `course_vllm/benchmarks/`

压测脚本，不属于服务主路径。

| 文件 | 职责 |
| --- | --- |
| `course_vllm/benchmarks/__init__.py` | benchmark 子包入口。 |
| `course_vllm/benchmarks/bench_server.py` | HTTP 并发压测脚本。向 `/generate` 发并发请求，统计延迟、吞吐和 token 数。 |

## `kernels/`

CUDA/C++ 源码目录。这里是手写教学 CUDA kernel 的实现。

| 文件 | 职责 |
| --- | --- |
| `kernels/course_ops.cpp` | PyTorch extension C++ binding。做 tensor device/dtype/shape 检查，声明 launcher，并把 CUDA kernel 暴露给 Python。 |
| `kernels/course_ops.cu` | CUDA kernel 实现。包含 row-wise softmax、RMSNorm、Qwen RoPE、naive matmul、paged attention decode。 |
| `kernels/vector_add.cu` | 最小 CUDA extension 示例，用于教学展示和基础编译验证。 |

## `examples/`

用户可直接运行的示例脚本，不计入服务主代码行数。

| 文件 | 职责 |
| --- | --- |
| `examples/offline_generate.py` | 离线生成脚本。支持单 prompt、多个 prompt、chat template、backend/dtype/max_tokens/sampling 参数。 |
| `examples/chat_client.py` | 交互式 HTTP chat CLI。支持多轮对话、stream/non-stream 切换、参数调整、健康检查、历史保存和加载。 |
| `examples/block_usage.py` | paged KV block 使用演示。展示 block 分配、block table、decode 追加 token 后的 block 使用情况。 |

## `validation/`

正确性验证脚本，不计入服务主代码行数。

| 文件 | 职责 |
| --- | --- |
| `validation/compare_qwen3.py` | 对齐 HuggingFace Qwen3。覆盖 forward、single decode、batch prefill、batch decode，可选择 `course` 或 `paged` backend，输出 logits 最大误差和平均误差。 |

## `tests/`

pytest 单元测试和集成测试，不计入服务主代码行数。

| 文件 | 职责 |
| --- | --- |
| `tests/test_attention.py` | attention 正确性测试，包括 paged attention reference 和 CUDA dispatch。 |
| `tests/test_block_manager.py` | paged KV block 分配、追加、释放和容量边界测试。 |
| `tests/test_chat_client.py` | CLI 客户端参数解析、命令行为、历史保存/加载等测试。 |
| `tests/test_engine.py` | Engine 生成、batch 生成、backend 调用和停止条件测试。 |
| `tests/test_kernels.py` | CUDA kernels 正确性测试，对比 PyTorch reference。 |
| `tests/test_kv_cache.py` | 连续 KV cache 写入、读取、释放测试。 |
| `tests/test_paged_kv_cache.py` | paged KV cache 物理 slot 写入和逻辑读取测试。 |
| `tests/test_protocol.py` | HTTP API Pydantic schema 默认值、序列化和参数校验测试。 |
| `tests/test_qwen3_torch.py` | 自有 Qwen3 PyTorch 模型组件和 backend 行为测试。 |
| `tests/test_sampler.py` | greedy、temperature、top-k、seed 等采样测试。 |
| `tests/test_scheduler.py` | scheduler 的 prefill/decode 调度、容量限制和结束状态测试。 |
| `tests/test_server_batching.py` | HTTP batching engine 合批、统计、异常传播和 worker thread 行为测试。 |

## `docs/`

设计说明和参考资料。

| 文件 | 职责 |
| --- | --- |
| `docs/reference_notes.md` | 参考项目阅读笔记，包括 nano-vllm、mini-sglang、llm.c、nanoGPT、tiny-llm 的可借鉴点。 |
| `docs/file_guide.md` | 当前文件职责说明，也就是本文档。 |

## 本地生成目录

这些目录是本机环境或运行测试后生成的，不属于源码职责说明，也不会进入 git：

```text
.venv/
.pytest_cache/
__pycache__/
course_vllm.egg-info/
dependence/
```

`dependence/` 里放的是本地解包的 GCC 14 deb，用于 CUDA extension 编译兼容，不是项目源码。
