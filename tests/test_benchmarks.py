from course_vllm.benchmarks.bench_server import summarize
from course_vllm.benchmarks.cache_aware_demo import (
    cache_aware_order,
    run_pd_disaggregation,
    run_tokendance,
    score_order,
)
from course_vllm.benchmarks.capacity_planner import estimate_capacity, render_capacity_report
from course_vllm.benchmarks.grader import STAGE_TESTS
from course_vllm.benchmarks.system_optimization import (
    SystemOptimizationConfig,
    admission_decision,
    estimate_overlap_plan,
)
from course_vllm.engine.policies import paper_to_system_map


class Args:
    num_requests = 2
    concurrency = 1


def test_bench_server_summarize_reports_serving_metrics():
    metrics = summarize(
        [
            {"latency_s": 1.0, "num_output_tokens": 2, "finish_reason": "length"},
            {"latency_s": 3.0, "num_output_tokens": 4, "finish_reason": "length"},
        ],
        elapsed_s=4.0,
        args=Args(),
    )

    assert metrics["requests_per_s"] == 0.5
    assert metrics["output_tokens_per_s"] == 1.5
    assert metrics["latency_p50_s"] == 1.0
    assert metrics["latency_p99_s"] == 3.0
    assert metrics["estimated_tpot_s"] == 4.0 / 6.0


def test_capacity_planner_estimates_kv_blocks():
    result = estimate_capacity(
        gpu_memory_gb=4,
        utilization=0.5,
        weight_memory_gb=1,
        safety_gb=0,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype="float16",
        block_size=8,
        max_model_len=16,
        hidden_size=8,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        target_batch_size=2,
    )

    assert result["kv_budget_gb"] == 1
    assert result["num_kv_blocks"] > 0
    assert result["total_token_slots"] == result["num_kv_blocks"] * 8
    assert result["parallelism"]["world_size"] == 4
    assert result["parallelism"]["tensor_parallel"]["all_reduce_bytes_per_token"] == 64


def test_capacity_planner_renders_report_with_parallelism_decision():
    result = estimate_capacity(
        gpu_memory_gb=4,
        utilization=0.5,
        weight_memory_gb=1,
        safety_gb=0,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        dtype="float16",
        block_size=8,
        max_model_len=16,
        hidden_size=8,
        context_parallel_size=2,
        num_experts=4,
        expert_parallel_size=2,
        active_experts=2,
    )
    report = render_capacity_report(result, target_concurrency=2, target_sequence_len=16)
    assert "Capacity Planning Report" in report
    assert "Need tensor/pipeline parallelism" in report
    assert "TP all-reduce bytes/token" in report
    assert "EP all-to-all bytes/token" in report
    assert "CP pass-KV bytes/prefill" in report


def test_cache_aware_order_improves_shared_prefix_score():
    prompts = [[1, 2, 3, 4], [8, 9], [1, 2, 3, 5], [1, 2, 7]]
    baseline = list(range(len(prompts)))
    optimized = cache_aware_order(prompts)
    assert score_order(prompts, optimized) >= score_order(prompts, baseline)


def test_system_optimization_helpers_explain_overlap_and_admission():
    plan = estimate_overlap_plan(SystemOptimizationConfig(pinned_memory=True, transfer_stream=True))
    assert plan["overlap_enabled"] is True
    decision = admission_decision(queue_depth=3, prompt_chars=10, max_queue_size=3, max_prompt_chars=100)
    assert decision["accepted"] is False
    assert decision["reason"] == "queue_full"


def test_paper_to_system_map_lists_engine_modules():
    mapping = paper_to_system_map("prefill-decode disaggregation")
    assert "engine/scheduler.py" in mapping["engine_modules"]
    assert "TTFT" in mapping["metrics"]


def test_frontier_demos_produce_comparable_metrics():
    pd = run_pd_disaggregation("128:16|2048:8|256:64")
    assert pd["estimated_speedup"] >= 1.0
    tokendance = run_tokendance("128:32|2048:4|256:16")
    assert tokendance["tokendance_completion_cost_tokens"] <= tokendance["baseline_completion_cost_tokens"]


def test_grader_has_stage_mappings_for_course_tail():
    assert "cuda_smoke" in STAGE_TESTS
    assert "week11" in STAGE_TESTS
    assert "week12" in STAGE_TESTS
    assert "week13" in STAGE_TESTS
    assert "week15" in STAGE_TESTS
    assert "tests/test_kernels.py::test_cuda_rms_norm_matches_torch" in STAGE_TESTS["cuda_smoke"]
    assert "tests/test_attention.py::test_cuda_paged_attention_decode_matches_dense_attention" in STAGE_TESTS["cuda_smoke"]
