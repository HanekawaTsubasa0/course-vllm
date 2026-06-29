from __future__ import annotations

import argparse
import json

from course_vllm.engine.policies import (
    cache_aware_order,
    estimate_pd_disaggregation,
    paper_to_system_map,
    score_decode_completion_cost,
    score_order,
    tokendance_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache-aware request ordering demo for week15.")
    parser.add_argument(
        "--prompts",
        default="1,2,3,4|1,2,3,9|8,7|1,2,5",
        help="pipe-separated token id lists, e.g. '1,2|1,3|9'",
    )
    parser.add_argument(
        "--requests",
        default="128:16|2048:8|256:64|1024:12",
        help="pipe-separated prompt_len:decode_len pairs for PD/TokenDance demos",
    )
    parser.add_argument("--mechanism", default="cache-aware serving")
    args = parser.parse_args()
    mapping = paper_to_system_map(args.mechanism)
    if mapping["mechanism"] == "cache-aware serving":
        result = run_cache_aware(args.prompts)
    elif mapping["mechanism"] == "prefill-decode disaggregation":
        result = run_pd_disaggregation(args.requests)
    elif mapping["mechanism"] == "tokendance-style scheduling":
        result = run_tokendance(args.requests)
    else:
        raise RuntimeError(f"unhandled mechanism: {mapping['mechanism']}")
    result["paper_to_system"] = mapping
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_cache_aware(raw_prompts: str) -> dict:
    prompts = [[int(item) for item in prompt.split(",") if item] for prompt in raw_prompts.split("|") if prompt]
    baseline = list(range(len(prompts)))
    optimized = cache_aware_order(prompts)
    return {
        "baseline_order": baseline,
        "cache_aware_order": optimized,
        "baseline_shared_prefix_score": score_order(prompts, baseline),
        "cache_aware_shared_prefix_score": score_order(prompts, optimized),
    }


def run_pd_disaggregation(raw_requests: str) -> dict:
    requests = parse_requests(raw_requests)
    return estimate_pd_disaggregation(requests)


def run_tokendance(raw_requests: str) -> dict:
    requests = parse_requests(raw_requests)
    baseline = list(range(len(requests)))
    optimized = tokendance_order(requests)
    return {
        "baseline_order": baseline,
        "tokendance_order": optimized,
        "baseline_completion_cost_tokens": score_decode_completion_cost(requests, baseline),
        "tokendance_completion_cost_tokens": score_decode_completion_cost(requests, optimized),
    }


def parse_requests(raw_requests: str) -> list[dict[str, int]]:
    requests = []
    for index, raw_request in enumerate(item for item in raw_requests.split("|") if item):
        prompt_len, decode_len = raw_request.split(":", maxsplit=1)
        requests.append(
            {
                "prompt_len": int(prompt_len),
                "decode_len": int(decode_len),
                "age": index,
                "generated": 0,
            }
        )
    return requests


if __name__ == "__main__":
    main()
