# Week 02 性能分析报告

## 环境

- GPU: 2 x NVIDIA GeForce RTX 4090
- Driver: 570.86.15
- CUDA: 12.8
- PyTorch: 2.8.0+cu128
- 模型: Qwen/Qwen3-0.6B
- backend: paged
- dtype: bfloat16
- stage: week02/week09 profiling path

## 可复现命令

```bash
cd /home/wangqi/llm_serving/course-vllm
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/profile/torch_profiler.py \
  --model Qwen/Qwen3-0.6B \
  --backend paged \
  --max-tokens 8 \
  --out profiles/torch_profiler \
  | tee profiles/reports/torch_profiler_summary.txt

OUT=profiles/ncu_kernels bash scripts/profile/ncu_kernel.sh \
  | tee profiles/reports/ncu_kernel_summary.txt

MODEL=Qwen/Qwen3-0.6B BACKEND=paged DTYPE=bfloat16 MAX_TOKENS=8 OUT=profiles/nsys_server_ready \
  bash scripts/profile/nsys_server.sh \
  | tee profiles/reports/nsys_server_ready_summary.txt
```

## Baseline 指标

来自 `profiles/reports/nsys_server_ready_summary.txt`。

| 指标 | 数值 |
| --- | ---: |
| requests/s | 1.430951 |
| output tokens/s | 11.447604 |
| latency p50 | 0.350910 s |
| latency p90 | 1.710348 s |
| latency p99 | 1.710348 s |
| estimated TPOT | 0.086120 s |

## Torch Profiler 观察

来自 `profiles/reports/torch_profiler_summary.txt`。

| Kernel/Op | Self CUDA | 占比 |
| --- | ---: | ---: |
| `aten::copy_` | 70.501 ms | 53.59% |
| PyTorch elementwise kernels | 69.934 ms | 53.16% |
| `matmul_tiled_kernel<bf16>` | 54.790 ms | 41.65% |
| `rms_norm_kernel<bf16>` | 2.088 ms | 1.59% |
| `paged_attention_decode_kernel` | 1.145 ms | 0.87% |
| `rope_kernel<bf16>` | 471.293 us | 0.36% |

## nsys 观察

- 产物：`profiles/nsys_server_ready.nsys-rep`
- 脚本已修正为轮询 `/health`，服务 ready 后再压测。
- 当前环境提示 CPU IP/backtrace sampling 和 context switch tracing 不支持，CUDA timeline 仍生成。
- HTTP benchmark 在 nsys profile 期间完成 4 个请求。

## ncu 观察

- 命令能启动 Nsight Compute 并连接 Python 进程。
- 当前用户没有访问 GPU performance counters 的权限，输出 `ERR_NVGPUCTRPERM`。
- 即使 performance counters 不可用，CUDA correctness 测试在 ncu 启动进程中完成：`16 passed in 4.89s`。
- 若要完整 NCU 指标，需要管理员放开 NVIDIA performance counter 权限，或使用有权限的用户运行。

## 瓶颈判断

结论：当前短 decode profiling 中，主要成本集中在张量拷贝/elementwise 和课程 tiled matmul；paged attention decode 已接入主路径但不是主要耗时。

证据：

- `aten::copy_` 和 elementwise kernels 占比较高。
- `matmul_tiled_kernel` 是主要课程 CUDA kernel 热点。
- `paged_attention_decode_kernel` 占比低，说明短上下文下 attention decode 不是主要瓶颈。

下一步优化：

- 对 Qwen3 MLP/投影层使用更接近 cuBLAS 的 matmul 或直接引入 cuBLAS 对照。
- 减少小 tensor 拷贝和 Python/extension launch 次数。
- 用更长 prompt、更高并发复测 paged attention 和 scheduler 的影响。
