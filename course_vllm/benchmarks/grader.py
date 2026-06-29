from __future__ import annotations

import argparse
import os
import subprocess
import sys


STAGE_TESTS = {
    "cuda_smoke": [
        "tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch",
        "tests/test_kernels.py::test_cuda_rms_norm_matches_torch",
        "tests/test_kernels.py::test_cuda_rope_matches_qwen3_rotate_half",
        "tests/test_kernels.py::test_cuda_matmul_tiled_matches_torch",
        "tests/test_kernels.py::test_cuda_softmax_matches_torch",
        "tests/test_attention.py::test_cuda_dense_attention_decode_matches_reference",
        "tests/test_attention.py::test_cuda_paged_attention_decode_matches_dense_attention",
    ],
    "week01": ["tests/test_protocol.py", "tests/test_server_api.py::test_create_app_health_reports_stage_and_kernel_impl"],
    "week02": ["tests/test_benchmarks.py::test_bench_server_summarize_reports_serving_metrics"],
    "week03": ["tests/test_kernels.py::test_vector_add_cuda_kernel_matches_torch"],
    "week04": ["tests/test_kernels.py::test_cuda_rms_norm_matches_torch", "tests/test_kernels.py::test_cuda_rope_matches_qwen3_rotate_half"],
    "week05": [
        "tests/test_kernels.py::test_cuda_matmul_matches_torch",
        "tests/test_kernels.py::test_cuda_matmul_tiled_matches_torch",
        "tests/test_qwen3_torch.py::test_course_linear_matches_torch_linear",
    ],
    "week06": ["tests/test_kernels.py::test_cuda_softmax_matches_torch", "tests/test_sampler.py"],
    "week07": [
        "tests/test_attention.py",
        "tests/test_qwen3_torch.py::test_dense_attention_prefill_reference_matches_torch_attention",
    ],
    "week08": ["tests/test_kv_cache.py"],
    "week09": ["tests/test_engine.py", "tests/test_chat_client.py"],
    "week10": ["tests/test_block_manager.py", "tests/test_paged_kv_cache.py"],
    "week11": ["tests/test_scheduler.py", "tests/test_server_batching.py"],
    "week12": [
        "tests/test_server_api.py",
        "tests/test_server_batching.py::test_batching_engine_reports_admission_limits",
        "tests/test_benchmarks.py::test_system_optimization_helpers_explain_overlap_and_admission",
    ],
    "week13": [
        "tests/test_benchmarks.py::test_capacity_planner_estimates_kv_blocks",
        "tests/test_benchmarks.py::test_capacity_planner_renders_report_with_parallelism_decision",
    ],
    "week15": [
        "tests/test_benchmarks.py::test_cache_aware_order_improves_shared_prefix_score",
        "tests/test_benchmarks.py::test_paper_to_system_map_lists_engine_modules",
        "tests/test_benchmarks.py::test_frontier_demos_produce_comparable_metrics",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run course-vllm stage checks.")
    parser.add_argument(
        "stage",
        choices=sorted(STAGE_TESTS),
        help="course week such as week11, or cuda_smoke for forced CUDA kernel checks",
    )
    parser.add_argument("--pytest-arg", action="append", default=[])
    args = parser.parse_args()
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", *STAGE_TESTS[args.stage], *args.pytest_arg]
    env = os.environ.copy()
    if args.stage == "cuda_smoke":
        env["COURSE_VLLM_STRICT_CUDA"] = "1"
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
