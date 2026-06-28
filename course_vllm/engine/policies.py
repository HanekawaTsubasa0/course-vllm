from __future__ import annotations


def shared_prefix_len(a: list[int], b: list[int]) -> int:
    total = 0
    for left, right in zip(a, b):
        if left != right:
            break
        total += 1
    return total


def cache_aware_order(prompts: list[list[int]]) -> list[int]:
    """Greedy request ordering that keeps similar prefixes adjacent."""

    if not prompts:
        return []
    remaining = set(range(len(prompts)))
    order = [min(remaining)]
    remaining.remove(order[0])
    while remaining:
        previous = prompts[order[-1]]
        next_index = max(
            remaining,
            key=lambda index: (shared_prefix_len(previous, prompts[index]), -index),
        )
        order.append(next_index)
        remaining.remove(next_index)
    return order


def score_order(prompts: list[list[int]], order: list[int]) -> int:
    score = 0
    for left, right in zip(order, order[1:]):
        score += shared_prefix_len(prompts[left], prompts[right])
    return score


def paper_to_system_map(mechanism: str) -> dict:
    mappings = {
        "cache-aware serving": {
            "paper_idea": "schedule requests with shared prefixes close together to increase KV reuse",
            "engine_modules": ["engine/policies.py", "engine/block_manager.py", "engine/engine.py"],
            "data_structures": ["prompt token ids", "block hash table", "scheduler waiting queue"],
            "metrics": ["prefix_cached_blocks", "TTFT", "requests/s", "KV fragmentation"],
            "minimal_experiment": "compare baseline order and cache_aware_order shared-prefix score",
        },
        "prefill-decode disaggregation": {
            "paper_idea": "separate prefill-heavy and decode-heavy work so each can use a tailored batch policy",
            "engine_modules": ["engine/scheduler.py", "server/batching.py", "engine/engine.py"],
            "data_structures": ["waiting queue", "running sequences", "KV cache handles"],
            "metrics": ["TTFT", "TPOT", "queue depth", "batch size"],
            "minimal_experiment": "toggle chunked prefill and compare latency distribution under mixed prompt lengths",
        },
        "tokendance-style scheduling": {
            "paper_idea": "adapt decode scheduling to token-level progress instead of static request order",
            "engine_modules": ["engine/scheduler.py", "engine/request.py"],
            "data_structures": ["per-sequence generated token count", "decode batch"],
            "metrics": ["tail latency", "fairness", "output_tokens/s"],
            "minimal_experiment": "compare FIFO decode with age/token-count priority on synthetic requests",
        },
    }
    key = mechanism.strip().lower()
    if key not in mappings:
        raise ValueError(f"unknown mechanism {mechanism!r}; choose one of: {', '.join(sorted(mappings))}")
    return {"mechanism": key, **mappings[key]}
