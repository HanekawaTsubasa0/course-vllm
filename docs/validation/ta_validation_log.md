# TA Validation Log

本文档保留 TA 在课程工程 v1 验收时的一次性实测记录，用于发布检查、课堂展示和故障复盘。不要把本文件作为学生最终报告模板；学生模板见 `docs/reports/week16_final_report_template.md`。

## 系统概览

- backend: paged
- model: Qwen/Qwen3-0.6B
- dtype: bfloat16 for serving, float32 for strict HF alignment
- kernel_impl: auto
- supported endpoints: `/health`, `/generate`, `/v1/chat/completions`
- AscendC: deferred by course decision

## 正确性证据

```bash
pytest -q -rs
```

本次结果：

```text
85 passed in 5.33s
```

CUDA kernel 和 attention：

```bash
pytest -q tests/test_kernels.py tests/test_attention.py -rs
```

本次结果：

```text
16 passed in 17.41s
```

## Qwen3/HF Alignment

```bash
for mode in forward decode batch-prefill batch-decode; do
  echo "=== $mode ==="
  python validation/compare_qwen3.py "$mode" \
    --model Qwen/Qwen3-0.6B \
    --backend paged \
    --dtype float32
done | tee profiles/reports/qwen3_alignment_float32.txt
```

最大误差摘要：

- forward: 0.000000
- decode: 0.000026
- batch-prefill course batch vs single: 0.000043
- batch-prefill course single vs HF: 0.000029
- batch-decode course vs HF batch: 0.000020

结论：float32 逻辑路径与 HF eager 实现对齐。

## 性能证据

Torch profiler:

- `matmul_tiled_kernel<bf16>`: 54.790 ms self CUDA
- `paged_attention_decode_kernel`: 1.145 ms self CUDA
- `rms_norm_kernel<bf16>`: 2.088 ms self CUDA
- `rope_kernel<bf16>`: 471.293 us self CUDA

Nsight Systems:

- 产物：`profiles/nsys_server_ready.nsys-rep`
- requests/s: 1.430951
- output tokens/s: 11.447604
- p99: 1.710348 s

Nsight Compute:

- 命令可启动，但当前用户缺少 performance counter 权限。
- 输出：`ERR_NVGPUCTRPERM`
- CUDA correctness in NCU process: `16 passed`

## 优化对比

| 阶段 | 改动 | requests/s | tokens/s | p50 | p90 | p99 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | paged backend, week11 | 4.038430 | 32.307436 | 0.231115 | 1.245282 | 1.248274 | 可运行 baseline |
| system optimization | pinned memory, transfer stream, chunked prefill, cache-aware scheduling | 3.981867 | 31.854933 | 0.251096 | 1.216504 | 1.219466 | 短负载收益不明显，p99 略好 |

## 容量规划

- GPU memory: 24 GiB
- KV budget: 17.400 GiB
- KV blocks: 10,181
- token slots: 162,896
- target concurrency: 32
- target sequence length: 2048
- 判断：不需要为了该目标引入 tensor/pipeline parallelism。

## 前沿机制复现

cache-aware serving demo:

- baseline shared-prefix score: 3
- cache-aware shared-prefix score: 5
- mapped modules: `engine/policies.py`, `engine/block_manager.py`, `engine/engine.py`

## 故障诊断复盘

现象：

- 默认沙箱下 `torch.cuda.is_available()` 为 False，CUDA 测试跳过。
- 直接用 `Qwen/Qwen3-0.6B` 跑课程 backend 时，早期版本只在当前目录找 `config.json`。
- `nsys_server.sh` 固定 sleep 5 秒，模型加载未完成时 benchmark 连接失败。
- `ncu` 不在 PATH。

定位证据：

- 提升权限下 `nvidia-smi` 显示 2 x RTX 4090。
- PyTorch CUDA build 为 `2.8.0+cu128`。
- HF cache 有 Qwen3 snapshot。
- `find /usr/local -path '*ncu'` 找到 `/usr/local/cuda-12.8/bin/ncu`。

修复：

- 在 GPU 可见环境重跑测试。
- 增加 `resolve_local_model_path`，支持 repo id 到本地 HF snapshot。
- `nsys_server.sh` 改为轮询 `/health`。
- profile 脚本自动查找 `ncu/nsys` 常见路径。

验证：

- 全量测试 `85 passed`。
- CUDA/attention 测试 `16 passed`。
- Qwen3/HF float32 对齐通过。
- nsys 生成 `profiles/nsys_server_ready.nsys-rep`。

剩余风险：

- NCU performance counters 需要管理员权限。
- 当前 benchmark 是小样本短负载，不能代表工业吞吐极限。
- 教学 CUDA kernel 重 correctness/readability，不追求超过 cuBLAS/FlashAttention。
