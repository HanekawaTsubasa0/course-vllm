# course-llm-serving

`course-llm-serving` 是 LLM serving 课程的学生起始工程。当前 Python 包名仍是 `course_vllm`。你会在一套可运行、可测试的代码里逐步实现：

- prefill / decode
- KV cache
- paged KV cache 和 block table
- batch prefill / batch decode
- continuous batching
- HTTP serving 和 streaming
- CUDA extension harness
- RMSNorm、RoPE、softmax、matmul、paged attention decode 等教学 CUDA kernel

本分支包含课程 starter code，核心实验位置用 `TODO(labXX)` 标出。每周先读 `docs/labs/weekXX_*.md`，再补对应代码并运行本周测试。

## 课程入口

建议先看这些文档：

- `docs/labs/README.md`：16 周实验目录。
- `docs/labs/weekXX_*.md`：每周背景、任务、TODO、测试和交付要求。
- `docs/reports/`：性能分析、容量规划、最终报告模板。
- `docs/file_guide.md`：工程文件职责说明。
- `docs/reference_notes.md`：参考项目和拓展阅读。

常用开关：

```bash
--backend reference|course
--kv-mode dense|paged
--stage week04
--kernel-impl torch|auto|cuda
```

- `backend=reference` 使用 HuggingFace/PyTorch 参考路径，适合在课程代码未补完时先跑通服务。
- `backend=course` 使用课程主线 engine。
- `kv-mode=dense` 使用连续 KV cache。
- `kv-mode=paged` 使用 paged KV cache 和 block table。
- `stage` 标记当前实验周次，`/health` 会返回对应信息。
- `kernel-impl=torch` 只走 PyTorch/reference path。
- `kernel-impl=auto` 在 CUDA tensor 上优先尝试课程 CUDA kernel，不可用时回退。
- `kernel-impl=cuda` 强制走课程 CUDA kernel，适合验收 CUDA 是否真正接入。

周次说明在 `docs/labs/`，按周学习讲义在 `docs/teaching/README.md`。

## 环境配置

进入项目并创建虚拟环境：

```bash
cd course-vllm
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

默认模型：

```text
Qwen/Qwen3-0.6B
```

如果本机已经有模型目录，可以用 `--model` 指定本地路径：

```bash
--model /path/to/Qwen3-0.6B
```

CUDA 实验需要 NVIDIA GPU、CUDA toolkit、`ninja`，以及与当前 `nvcc` 兼容的 host C++ compiler。没有 CUDA 的机器可以先完成非 CUDA 任务，CUDA 相关测试会按 pytest 标记跳过或失败提示。

## 每周工作流

1. 读本周讲义：

   ```bash
   ls docs/labs
   ```

2. 找到本周 TODO：

   ```bash
   rg "TODO\\(lab04\\)" course_vllm kernels docs/labs
   ```

3. 先跑本周测试，确认当前失败点。

4. 补代码，保持函数签名、输入输出 shape 和错误处理不变。

5. 重跑本周 pytest 或 grader。

## 快速运行

课程代码未补完时，优先用 `reference` 路径启动服务：

```bash
python -m course_vllm.server.api \
  --model Qwen/Qwen3-0.6B \
  --backend reference \
  --stage week01 \
  --port 18080
```

健康检查：

```bash
curl -s http://127.0.0.1:18080/health
```

非 streaming 请求：

```bash
curl -s -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":false,"sampling_params":{"temperature":0,"max_tokens":16}}'
```

streaming 请求：

```bash
curl -s -N -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","stream":true,"sampling_params":{"temperature":0,"max_tokens":16}}'
```

当你补完对应课程模块后，可以切到课程路径：

```bash
python examples/offline_generate.py \
  --model Qwen/Qwen3-0.6B \
  --backend course \
  --kv-mode paged \
  --stage week11 \
  --kernel-impl auto \
  --prompt "Explain KV cache in one sentence." \
  --max-tokens 32 \
  --temperature 0
```

Paged KV 调试：

```bash
python examples/block_usage.py \
  --num-blocks 8 \
  --block-size 4 \
  --prompt-lens 3,6,9 \
  --decode-steps 2
```

## 测试与验收

按周运行 grader：

```bash
python -m course_vllm.benchmarks.grader week03
python -m course_vllm.benchmarks.grader week04
python -m course_vllm.benchmarks.grader week11
```

CUDA smoke：

```bash
python -m course_vllm.benchmarks.grader cuda_smoke
```

也可以直接运行 pytest：

```bash
pytest -q -rs
pytest -q tests/test_kernels.py -rs
pytest -q tests/test_scheduler.py tests/test_server_batching.py
```

不同周次对应的测试命令以 `docs/labs/weekXX_*.md` 为准。

## 性能分析

Week02 和后续系统实验会用到这些脚本：

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
python -m course_vllm.benchmarks.bench_server \
  --url http://127.0.0.1:18080/generate \
  --requests 8 \
  --concurrency 2 \
  --prompt "Explain continuous batching." \
  --max-tokens 16
```

报告模板在 `docs/reports/`。

## 提交要求

每周提交时建议包含：

- 本周补充的源码。
- 本周 pytest 或 grader 输出。
- 讲义要求的报告或截图。
- 简短说明：实现了哪些 TODO、还有哪些限制。

不要修改测试来绕过失败；如果测试失败，先根据错误信息定位对应 TODO。
