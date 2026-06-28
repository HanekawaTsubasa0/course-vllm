# course-vllm 可运行验证与交付指南

本文档记录 TA 发布前验收流程和本仓库当前可复现的正确性、CUDA、profiling、serving benchmark 和课程报告生成流程。命令默认在仓库根目录运行；本机路径、硬件型号和 profiles 文件名不进入学生 handout。

## 环境

```bash
cd /home/wangqi/llm_serving/course-vllm
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

本次验证环境：

- GPU: 2 x NVIDIA GeForce RTX 4090
- Driver: 570.86.15
- CUDA driver capability: 12.8
- PyTorch: 2.8.0+cu128
- Model: Qwen/Qwen3-0.6B, local HuggingFace cache

默认 Codex 沙箱可能看不到 GPU。需要在 GPU 可见环境运行时，确认：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 1. 全量测试

```bash
pytest -q -rs
```

本次结果：

```text
85 passed in 5.33s
```

单独验证 CUDA kernel 和 attention：

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
pytest -q tests/test_kernels.py tests/test_attention.py -rs
```

本次结果：

```text
16 passed in 17.41s
```

## 2. Qwen3/HuggingFace 对齐

```bash
mkdir -p profiles/reports
for mode in forward decode batch-prefill batch-decode; do
  echo "=== $mode ==="
  python validation/compare_qwen3.py "$mode" \
    --model Qwen/Qwen3-0.6B \
    --backend paged \
    --dtype float32
done | tee profiles/reports/qwen3_alignment_float32.txt
```

本次结果摘要：

| 模式 | max_abs_diff | mean_abs_diff |
| --- | ---: | ---: |
| forward course vs HF | 0.000000 | 0.000000 |
| decode step 1 | 0.000026 | 0.000005 |
| decode step 2 | 0.000021 | 0.000003 |
| decode step 3 | 0.000014 | 0.000002 |
| batch prefill course batch vs single | 0.000043 | 0.000005 |
| batch prefill course single vs HF | 0.000029 | 0.000003 |
| HF batch vs HF single | 0.000064 | 0.000008 |
| batch decode course vs HF batch | 0.000020 | 0.000003 |
| HF batch decode vs HF single decode | 0.000021 | 0.000003 |

完整输出见 `profiles/reports/qwen3_alignment_float32.txt`。

## 3. Torch Profiler

```bash
python scripts/profile/torch_profiler.py \
  --model Qwen/Qwen3-0.6B \
  --backend paged \
  --max-tokens 8 \
  --out profiles/torch_profiler \
  | tee profiles/reports/torch_profiler_summary.txt
```

本次主要 CUDA 热点：

- `aten::copy_`: 70.501 ms self CUDA, 53.59%
- PyTorch elementwise kernels: 69.934 ms self CUDA, 53.16%
- `matmul_tiled_kernel<__nv_bfloat16>`: 54.790 ms self CUDA, 41.65%
- `rms_norm_kernel<__nv_bfloat16>`: 2.088 ms self CUDA, 1.59%
- `paged_attention_decode_kernel`: 1.145 ms self CUDA, 0.87%
- `rope_kernel<__nv_bfloat16>`: 471.293 us self CUDA, 0.36%

完整输出见 `profiles/reports/torch_profiler_summary.txt`。

## 4. Nsight Compute

```bash
OUT=profiles/ncu_kernels bash scripts/profile/ncu_kernel.sh \
  | tee profiles/reports/ncu_kernel_summary.txt
```

脚本会自动查找以下路径：

- `ncu` in `PATH`
- `/usr/local/cuda-12.8/bin/ncu`
- `/usr/local/cuda/bin/ncu`
- `/usr/local/NVIDIA-Nsight-Compute-2025.3/ncu`

本次环境能启动 Nsight Compute 并连接进程，但当前用户没有 GPU performance counters 权限：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
16 passed in 4.89s
```

如果需要完整 NCU 指标，需要管理员按 NVIDIA 文档放开 performance counter 权限，或用具备权限的用户运行。

## 5. Nsight Systems

```bash
MODEL=Qwen/Qwen3-0.6B \
BACKEND=paged \
DTYPE=bfloat16 \
MAX_TOKENS=8 \
OUT=profiles/nsys_server_ready \
bash scripts/profile/nsys_server.sh \
  | tee profiles/reports/nsys_server_ready_summary.txt
```

脚本会启动服务、轮询 `/health`，ready 后运行 HTTP benchmark，再生成 `.nsys-rep`。

本次输出：

```text
requests=4
completed=4
concurrency=1
requests_per_s=1.430951
output_tokens_per_s=11.447604
latency_p50_s=0.350910
latency_p90_s=1.710348
latency_p99_s=1.710348
estimated_tpot_s=0.086120
Generated: profiles/nsys_server_ready.nsys-rep
```

注意：当前环境提示 CPU IP/backtrace sampling 和 context switch tracing 不支持，但 CUDA timeline 仍会生成。

## 6. Serving Benchmark

Baseline:

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend paged \
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
  --json | tee profiles/reports/bench_baseline_week11.json
```

Optimized:

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend paged \
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

另一个终端：

```bash
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18082/generate \
  --num-requests 8 \
  --concurrency 2 \
  --max-tokens 8 \
  --json | tee profiles/reports/bench_optimized_week12.json
```

本次结果：

| 配置 | requests/s | output tokens/s | p50 | p90 | p99 | estimated TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline week11 | 4.038430 | 32.307436 | 0.231115 | 1.245282 | 1.248274 | 0.060612 |
| optimized week12 | 3.981867 | 31.854933 | 0.251096 | 1.216504 | 1.219466 | 0.061504 |

结论：这个短 prompt、短 decode 的小负载下，pinned memory/transfer stream/chunked prefill/cache-aware scheduling 没有显著提升吞吐；p90/p99 略好，p50 略差，整体差异在小样本波动范围内。更长 prompt、更高并发时这些机制才更可能显现收益。

## 7. Capacity Planning

```bash
python -m course_vllm.benchmarks.capacity_planner \
  --gpu-memory-gb 24 \
  --weight-memory-gb 2 \
  --num-layers 28 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --block-size 16 \
  --max-model-len 2048 \
  --target-concurrency 32 \
  --target-sequence-len 2048 \
  --report | tee profiles/reports/week13_capacity_report.md
```

本次判断：

- KV budget: 17.400 GiB
- KV blocks: 10181
- Token slots: 162896
- Full-length sequences: 79
- Target concurrency: 32
- Need more KV capacity: False
- Need tensor/pipeline parallelism: False

## 8. Paper-to-System Demo

```bash
python -m course_vllm.benchmarks.cache_aware_demo \
  --mechanism "cache-aware serving" \
  --prompts "1,2,3,4|1,2,3,9|8,7|1,2,5" \
  | tee profiles/reports/week15_cache_aware_demo.json
```

本次结果：

- baseline shared-prefix score: 3
- cache-aware shared-prefix score: 5
- mapped modules: `engine/policies.py`, `engine/block_manager.py`, `engine/engine.py`
- target metrics: `prefix_cached_blocks`, `TTFT`, `requests/s`, `KV fragmentation`

## 9. 当前产物清单

```text
profiles/nsys_server_ready.nsys-rep
profiles/reports/qwen3_alignment_float32.txt
profiles/reports/torch_profiler_summary.txt
profiles/reports/ncu_kernel_summary.txt
profiles/reports/nsys_server_ready_summary.txt
profiles/reports/bench_baseline_week11.json
profiles/reports/bench_optimized_week12.json
profiles/reports/week12_system_optimization_plan.json
profiles/reports/week13_capacity_report.md
profiles/reports/week15_cache_aware_demo.json
```

这些文件是本次课程验收可直接引用的实测证据。

## 10. Clean-clone Smoke

发布前在新目录从零验证一次，证明课程闭环不依赖当前工作树的历史 profiles、旧 venv 或手工缓存。

```bash
bash scripts/validation/clean_clone_smoke.sh /tmp/course-vllm-clean-smoke
```

脚本会 clone 当前仓库、创建新 venv、安装项目、运行非 CUDA 基础 pytest、`grader week01/week02/week11/week12`，并启动一次 HTTP demo。CUDA kernel 编译与接入另用 `python -m course_vllm.benchmarks.grader cuda_smoke` 在 GPU 可见且 nvcc/G++ 兼容的环境验收。

推送远程后，可以用 GitHub fresh clone 验证公开仓库从零可复现：

```bash
REMOTE_URL=git@github.com:HanekawaTsubasa0/course-vllm.git \
  bash scripts/validation/clean_clone_smoke.sh /tmp/course-vllm-github-smoke
```

## 11. CUDA Toolchain Troubleshooting

CUDA tests 通过 PyTorch extension JIT 编译 `kernels/*.cu`。如果 GPU 可见但 `cuda_smoke` 在编译阶段失败，优先检查 host compiler 是否被当前 `nvcc` 支持。

本次 TA 机器的经验是：CUDA driver capability 为 12.8，系统默认 G++ 15 会触发 nvcc/标准库兼容问题；可用兼容 G++ 解包到本地 `dependence/`，不安装到系统：

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

`dependence/` 已写入 `.gitignore`，不会进入 git。不同机器不必照抄该版本号；原则是使用与本机 CUDA toolkit 兼容的 host C++ compiler。
