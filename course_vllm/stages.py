from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageSpec:
    key: str
    week: int
    title: str
    topic: str
    code_status: str
    required_outputs: tuple[str, ...]
    test_hints: tuple[str, ...]


_STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        key="week01",
        week=1,
        title="课程导论",
        topic="prefill/decode, TTFT/TPOT/tokens/s, baseline serving",
        code_status="implemented",
        required_outputs=("能启动 HTTP 服务", "能发出非流式和流式请求", "记录首个 baseline 指标"),
        test_hints=("tests/test_engine.py", "tests/test_server_batching.py"),
    ),
    StageSpec(
        key="week02",
        week=2,
        title="性能分析",
        topic="roofline, nsys, ncu, PyTorch profiler, serving metrics",
        code_status="implemented",
        required_outputs=("nsys timeline", "ncu kernel report", "性能分析报告"),
        test_hints=("course_vllm/benchmarks/bench_server.py", "scripts/profile/"),
    ),
    StageSpec(
        key="week03",
        week=3,
        title="CUDA 入门",
        topic="CUDA extension, vector add, correctness and timing harness",
        code_status="implemented",
        required_outputs=("vector_add CUDA kernel", "correctness test", "micro benchmark"),
        test_hints=("tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch",),
    ),
    StageSpec(
        key="week04",
        week=4,
        title="RMSNorm 与 RoPE",
        topic="RMSNorm, RoPE, mixed precision, CUDA main-path dispatch",
        code_status="implemented",
        required_outputs=("RMSNorm CUDA", "RoPE CUDA", "通过 --kernel-impl auto/cuda 接入 Qwen3 路径"),
        test_hints=("tests/test_kernels.py", "tests/test_qwen3_torch.py"),
    ),
    StageSpec(
        key="week05",
        week=5,
        title="线性层与矩阵乘",
        topic="naive/tiled matmul, cuBLAS comparison, projection dispatch",
        code_status="implemented",
        required_outputs=("naive matmul", "tiled matmul", "CourseLinear 主路径接入", "与 torch/cublas 对照"),
        test_hints=("tests/test_kernels.py::test_cuda_matmul_matches_torch", "tests/test_kernels.py::test_cuda_matmul_tiled_matches_torch"),
    ),
    StageSpec(
        key="week06",
        week=6,
        title="归约与 Softmax",
        topic="parallel reduction, stable softmax, sampling path",
        code_status="implemented",
        required_outputs=("stable softmax CUDA", "采样 softmax 接入", "精度对比"),
        test_hints=("tests/test_kernels.py::test_cuda_softmax_matches_torch", "tests/test_sampler.py"),
    ),
    StageSpec(
        key="week07",
        week=7,
        title="Attention",
        topic="dense prefill attention, decode attention, tiled online softmax",
        code_status="implemented",
        required_outputs=("dense prefill CUDA", "dense decode CUDA", "paged decode CUDA", "FlashAttention 风格报告"),
        test_hints=("tests/test_attention.py",),
    ),
    StageSpec(
        key="week08",
        week=8,
        title="KV Cache",
        topic="continuous KV cache, append, variable sequence lengths",
        code_status="implemented",
        required_outputs=("dense KV cache", "连续生成多个 token", "cache correctness tests"),
        test_hints=("tests/test_kv_cache.py", "tests/test_qwen3_torch.py"),
    ),
    StageSpec(
        key="week09",
        week=9,
        title="推理引擎",
        topic="request lifecycle, tokenizer boundary, scheduler, streaming",
        code_status="implemented",
        required_outputs=("Engine 主循环", "streaming output", "单请求 profiler 报告"),
        test_hints=("tests/test_engine.py", "tests/test_chat_client.py"),
    ),
    StageSpec(
        key="week10",
        week=10,
        title="分页 KV Cache",
        topic="block manager, block table, slot mapping, fragmentation",
        code_status="implemented",
        required_outputs=("paged KV cache", "block usage demo", "碎片统计报告"),
        test_hints=("tests/test_block_manager.py", "tests/test_paged_kv_cache.py"),
    ),
    StageSpec(
        key="week11",
        week=11,
        title="连续批处理",
        topic="batch prefill/decode, queue policy, chunked prefill, preemption",
        code_status="implemented",
        required_outputs=("batch scheduler", "HTTP batching", "吞吐/时延曲线"),
        test_hints=("tests/test_scheduler.py", "tests/test_server_batching.py"),
    ),
    StageSpec(
        key="week12",
        week=12,
        title="系统优化",
        topic="pinned memory, stream, async overlap, admission control",
        code_status="implemented",
        required_outputs=("请求准入开关", "overlap 实验脚本", "优化前后报告"),
        test_hints=("tests/test_server_batching.py", "course_vllm/benchmarks/system_optimization.py", "docs/reports/week12_system_optimization_template.md"),
    ),
    StageSpec(
        key="week13",
        week=13,
        title="多卡推理",
        topic="TP/PP/EP, NCCL, placement and capacity planning",
        code_status="implemented",
        required_outputs=("capacity planning", "显存上限估算", "多卡触发条件判断", "容量规划报告"),
        test_hints=("course_vllm/benchmarks/capacity_planner.py", "docs/reports/week13_capacity_planning_template.md"),
    ),
    StageSpec(
        key="week14",
        week=14,
        title="AscendC",
        topic="AscendC programming model and CUDA comparison",
        code_status="scaffold",
        required_outputs=("暂缓：等待 Ascend 后端/硬件条件后补充",),
        test_hints=("deferred",),
    ),
    StageSpec(
        key="week15",
        week=15,
        title="前沿专题",
        topic="prefill/decode disaggregation, cache-aware serving, paper-to-system mapping",
        code_status="implemented",
        required_outputs=("paper-to-system 映射", "小型机制复现", "指标对比"),
        test_hints=("docs/reports/week15_paper_to_system_template.md",),
    ),
    StageSpec(
        key="week16",
        week=16,
        title="总结与展示",
        topic="operators, cache, scheduler, serving, diagnosis review",
        code_status="implemented",
        required_outputs=("最终展示", "性能证据", "故障诊断复盘"),
        test_hints=("docs/labs/week16_final_review.md", "docs/reports/week16_final_report_template.md"),
    ),
)

STAGES: dict[str, StageSpec] = {stage.key: stage for stage in _STAGES}
LATEST_STAGE = _STAGES[-1].key


def normalize_stage(stage: str | int | None) -> str:
    if stage is None:
        return LATEST_STAGE
    if isinstance(stage, int):
        key = f"week{stage:02d}"
    else:
        text = stage.strip().lower()
        key = f"week{int(text):02d}" if text.isdigit() else text
    if key not in STAGES:
        valid = ", ".join(STAGES)
        raise ValueError(f"unsupported course stage {stage!r}; valid stages: {valid}")
    return key


def stage_spec(stage: str | int | None = None) -> StageSpec:
    return STAGES[normalize_stage(stage)]


def stage_overview(stage: str | int | None = None) -> dict:
    spec = stage_spec(stage)
    return {
        "key": spec.key,
        "week": spec.week,
        "title": spec.title,
        "topic": spec.topic,
        "code_status": spec.code_status,
        "required_outputs": list(spec.required_outputs),
        "test_hints": list(spec.test_hints),
    }


def all_stage_overviews() -> list[dict]:
    return [stage_overview(stage.key) for stage in _STAGES]
