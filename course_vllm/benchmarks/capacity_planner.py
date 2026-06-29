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
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    vocab_size: int | None = None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    context_parallel_size: int = 1,
    num_experts: int = 0,
    active_experts: int = 0,
    microbatch_size: int = 1,
    network_bandwidth_gbps: float = 200.0,
    target_batch_size: int = 1,
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
    parallelism = estimate_parallelism(
        num_layers=num_layers,
        hidden_size=hidden_size or num_kv_heads * head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        dtype_bytes=byte,
        weight_memory_gb=weight_memory_gb,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_model_len=max_model_len,
        target_batch_size=target_batch_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        context_parallel_size=context_parallel_size,
        num_experts=num_experts,
        active_experts=active_experts,
        microbatch_size=microbatch_size,
        network_bandwidth_gbps=network_bandwidth_gbps,
    )
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
        "parallelism": parallelism,
        "note": "This is a capacity upper bound; scheduler policy and kernel throughput can reduce usable concurrency.",
    }


def estimate_parallelism(
    *,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int | None,
    vocab_size: int | None,
    dtype_bytes: int,
    weight_memory_gb: float,
    num_kv_heads: int,
    head_dim: int,
    max_model_len: int,
    target_batch_size: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    context_parallel_size: int,
    num_experts: int,
    active_experts: int,
    microbatch_size: int,
    network_bandwidth_gbps: float,
) -> dict:
    for name, value in {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "context_parallel_size": context_parallel_size,
        "microbatch_size": microbatch_size,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    intermediate_size = intermediate_size or hidden_size * 4
    vocab_size = vocab_size or hidden_size * 4
    active_experts = active_experts or min(2, num_experts) if num_experts else 0
    world_size = tensor_parallel_size * pipeline_parallel_size * max(1, expert_parallel_size)
    weight_per_gpu_gb = weight_memory_gb / max(1, tensor_parallel_size * pipeline_parallel_size)
    layers_per_stage = (num_layers + pipeline_parallel_size - 1) // pipeline_parallel_size

    attention_flops_per_token = 4 * hidden_size * hidden_size + 2 * max_model_len * hidden_size
    mlp_flops_per_token = 3 * hidden_size * intermediate_size
    dense_flops_per_token = num_layers * (attention_flops_per_token + mlp_flops_per_token)
    lm_head_flops_per_token = 2 * hidden_size * vocab_size
    flops_per_token = dense_flops_per_token + lm_head_flops_per_token
    flops_per_gpu_per_token = flops_per_token / max(1, tensor_parallel_size * pipeline_parallel_size)

    activation_bytes = target_batch_size * hidden_size * dtype_bytes
    tp_all_reduce_bytes_per_token = 2 * num_layers * activation_bytes * (tensor_parallel_size - 1) / tensor_parallel_size
    pp_activation_bytes_per_microbatch = microbatch_size * max_model_len * hidden_size * dtype_bytes
    pp_bubble_fraction = (pipeline_parallel_size - 1) / max(1, microbatch_size + pipeline_parallel_size - 1)

    expert_weight_per_gpu_gb = 0.0
    ep_all_to_all_bytes_per_token = 0.0
    if num_experts > 0 and expert_parallel_size > 1:
        expert_params = num_layers * num_experts * 3 * hidden_size * intermediate_size
        expert_weight_per_gpu_gb = expert_params * dtype_bytes / expert_parallel_size / 1024**3
        ep_all_to_all_bytes_per_token = (
            2
            * num_layers
            * target_batch_size
            * active_experts
            * hidden_size
            * dtype_bytes
            * (expert_parallel_size - 1)
            / expert_parallel_size
        )

    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    cp_local_context = (max_model_len + context_parallel_size - 1) // context_parallel_size
    cp_pass_q_bytes = target_batch_size * hidden_size * dtype_bytes * max(0, context_parallel_size - 1)
    cp_pass_kv_bytes = (
        target_batch_size
        * cp_local_context
        * 2
        * num_layers
        * num_kv_heads
        * head_dim
        * dtype_bytes
        * max(0, context_parallel_size - 1)
        / context_parallel_size
    )

    comm_bytes_per_token = tp_all_reduce_bytes_per_token + ep_all_to_all_bytes_per_token + cp_pass_q_bytes
    network_bytes_per_s = network_bandwidth_gbps * 1e9 / 8
    comm_time_ms_per_token = comm_bytes_per_token / network_bytes_per_s * 1000 if network_bytes_per_s else 0
    return {
        "world_size": world_size,
        "weight_per_gpu_gb": weight_per_gpu_gb,
        "layers_per_pipeline_stage": layers_per_stage,
        "flops_per_token": flops_per_token,
        "flops_per_gpu_per_token": flops_per_gpu_per_token,
        "tensor_parallel": {
            "size": tensor_parallel_size,
            "all_reduce_bytes_per_token": tp_all_reduce_bytes_per_token,
        },
        "pipeline_parallel": {
            "size": pipeline_parallel_size,
            "activation_bytes_per_microbatch": pp_activation_bytes_per_microbatch,
            "bubble_fraction": pp_bubble_fraction,
        },
        "expert_parallel": {
            "size": expert_parallel_size,
            "num_experts": num_experts,
            "active_experts": active_experts,
            "expert_weight_per_gpu_gb": expert_weight_per_gpu_gb,
            "all_to_all_bytes_per_token": ep_all_to_all_bytes_per_token,
        },
        "context_parallel": {
            "size": context_parallel_size,
            "local_context_len": cp_local_context,
            "kv_bytes_per_token": kv_bytes_per_token,
            "pass_q_bytes_per_token": cp_pass_q_bytes,
            "pass_kv_bytes_per_prefill": cp_pass_kv_bytes,
        },
        "communication": {
            "network_bandwidth_gbps": network_bandwidth_gbps,
            "bytes_per_token": comm_bytes_per_token,
            "time_ms_per_token_at_peak": comm_time_ms_per_token,
        },
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
        "## Multi-GPU Resource Model",
        f"- World size: {result['parallelism']['world_size']}",
        f"- Weight per GPU: {result['parallelism']['weight_per_gpu_gb']:.3f} GiB",
        f"- FLOPs/token/GPU: {result['parallelism']['flops_per_gpu_per_token']:.3e}",
        f"- TP all-reduce bytes/token: {result['parallelism']['tensor_parallel']['all_reduce_bytes_per_token']:.0f}",
        f"- PP layers/stage: {result['parallelism']['layers_per_pipeline_stage']}",
        f"- PP bubble fraction: {result['parallelism']['pipeline_parallel']['bubble_fraction']:.3f}",
        f"- EP all-to-all bytes/token: {result['parallelism']['expert_parallel']['all_to_all_bytes_per_token']:.0f}",
        f"- CP local context length: {result['parallelism']['context_parallel']['local_context_len']}",
        f"- CP pass-Q bytes/token: {result['parallelism']['context_parallel']['pass_q_bytes_per_token']:.0f}",
        f"- CP pass-KV bytes/prefill: {result['parallelism']['context_parallel']['pass_kv_bytes_per_prefill']:.0f}",
        f"- Communication time/token at peak bandwidth: {result['parallelism']['communication']['time_ms_per_token_at_peak']:.6f} ms",
        "",
        "## Bottleneck Judgment",
        "- If token slots are insufficient, reduce max_model_len/batch size or add GPUs for KV capacity.",
        "- If TP all-reduce or EP all-to-all time is comparable to per-token compute time, the plan is communication-bound.",
        "- If PP bubble is high, increase microbatch count or avoid pipeline parallelism for latency-sensitive decode.",
        "- If CP pass-KV is large, use it only when long-context memory savings dominate the extra communication.",
        "- If token slots are sufficient but latency misses SLO, profile kernels before increasing parallelism.",
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
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--tp", "--tensor-parallel-size", dest="tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pp", "--pipeline-parallel-size", dest="pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--ep", "--expert-parallel-size", dest="expert_parallel_size", type=int, default=1)
    parser.add_argument("--cp", "--context-parallel-size", dest="context_parallel_size", type=int, default=1)
    parser.add_argument("--num-experts", type=int, default=0)
    parser.add_argument("--active-experts", type=int, default=0)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--network-bandwidth-gbps", type=float, default=200.0)
    parser.add_argument("--target-concurrency", type=int, default=8)
    parser.add_argument("--target-sequence-len", type=int, default=2048)
    parser.add_argument("--report", action="store_true", help="print a Markdown capacity planning report")
    args = parser.parse_args()
    values = vars(args).copy()
    target_concurrency = values.pop("target_concurrency")
    target_sequence_len = values.pop("target_sequence_len")
    report = values.pop("report")
    values["target_batch_size"] = target_concurrency
    result = estimate_capacity(**values)
    if report:
        print(render_capacity_report(result, target_concurrency=target_concurrency, target_sequence_len=target_sequence_len))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
