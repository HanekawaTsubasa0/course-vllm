from __future__ import annotations

import argparse
import json
from math import floor


DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "fp8": 1,
}


def estimate_capacity(
    *,
    gpu_memory_gb: float,
    utilization: float,
    weight_memory_gb: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str,
    block_size: int,
    max_model_len: int,
    safety_gb: float,
) -> dict:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype {dtype!r}; choose one of {', '.join(DTYPE_BYTES)}")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    byte = DTYPE_BYTES[dtype]
    usable_bytes = gpu_memory_gb * utilization * 1024**3
    reserved_bytes = (weight_memory_gb + safety_gb) * 1024**3
    kv_budget_bytes = max(0, usable_bytes - reserved_bytes)
    block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * byte
    num_blocks = floor(kv_budget_bytes / block_bytes) if block_bytes else 0
    total_token_slots = num_blocks * block_size
    max_full_length_sequences = total_token_slots // max_model_len if max_model_len > 0 else 0
    return {
        "gpu_memory_gb": gpu_memory_gb,
        "utilization": utilization,
        "weight_memory_gb": weight_memory_gb,
        "safety_gb": safety_gb,
        "dtype": dtype,
        "kv_budget_gb": kv_budget_bytes / 1024**3,
        "kv_block_bytes": block_bytes,
        "num_kv_blocks": num_blocks,
        "block_size": block_size,
        "total_token_slots": total_token_slots,
        "max_full_length_sequences": max_full_length_sequences,
        "note": "This is a capacity upper bound; scheduler policy and kernel throughput can reduce usable concurrency.",
    }


def render_capacity_report(result: dict, *, target_concurrency: int, target_sequence_len: int) -> str:
    need_tensor_parallel = target_concurrency > result["max_full_length_sequences"]
    token_slots_needed = target_concurrency * target_sequence_len
    need_more_kv_capacity = token_slots_needed > result["total_token_slots"]
    lines = [
        "# Week 13 Capacity Planning Report",
        "",
        "## Inputs",
        f"- GPU memory: {result['gpu_memory_gb']} GiB",
        f"- Utilization target: {result['utilization']:.2f}",
        f"- Weight memory: {result['weight_memory_gb']} GiB",
        f"- Safety reserve: {result['safety_gb']} GiB",
        f"- KV dtype: {result['dtype']}",
        f"- Block size: {result['block_size']}",
        "",
        "## KV Capacity",
        f"- KV budget: {result['kv_budget_gb']:.3f} GiB",
        f"- KV block bytes: {result['kv_block_bytes']}",
        f"- KV blocks: {result['num_kv_blocks']}",
        f"- Token slots: {result['total_token_slots']}",
        f"- Full-length sequences: {result['max_full_length_sequences']}",
        "",
        "## Placement Decision",
        f"- Target concurrency: {target_concurrency}",
        f"- Target sequence length: {target_sequence_len}",
        f"- Needed token slots: {token_slots_needed}",
        f"- Need more KV capacity: {need_more_kv_capacity}",
        f"- Need tensor/pipeline parallelism: {need_tensor_parallel or need_more_kv_capacity}",
        "",
        "## Bottleneck Judgment",
        "- If token slots are insufficient, reduce max_model_len/batch size or add GPUs for KV capacity.",
        "- If token slots are sufficient but latency misses SLO, profile kernels and consider tensor parallelism.",
        "- If queueing dominates, tune admission control and continuous batching policy before adding GPUs.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate paged KV cache capacity for the course-vllm engine.")
    parser.add_argument("--gpu-memory-gb", type=float, required=True)
    parser.add_argument("--utilization", type=float, default=0.85)
    parser.add_argument("--weight-memory-gb", type=float, required=True)
    parser.add_argument("--safety-gb", type=float, default=1.0)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPE_BYTES))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--target-concurrency", type=int, default=8)
    parser.add_argument("--target-sequence-len", type=int, default=2048)
    parser.add_argument("--report", action="store_true", help="print a Markdown capacity planning report")
    args = parser.parse_args()
    values = vars(args).copy()
    target_concurrency = values.pop("target_concurrency")
    target_sequence_len = values.pop("target_sequence_len")
    report = values.pop("report")
    result = estimate_capacity(**values)
    if report:
        print(render_capacity_report(result, target_concurrency=target_concurrency, target_sequence_len=target_sequence_len))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
