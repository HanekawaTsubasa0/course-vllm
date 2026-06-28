from __future__ import annotations

import argparse
import json

from course_vllm.engine.policies import cache_aware_order, paper_to_system_map, score_order


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache-aware request ordering demo for week15.")
    parser.add_argument(
        "--prompts",
        default="1,2,3,4|1,2,3,9|8,7|1,2,5",
        help="pipe-separated token id lists, e.g. '1,2|1,3|9'",
    )
    parser.add_argument("--mechanism", default="cache-aware serving")
    args = parser.parse_args()
    prompts = [
        [int(item) for item in prompt.split(",") if item]
        for prompt in args.prompts.split("|")
        if prompt
    ]
    baseline = list(range(len(prompts)))
    optimized = cache_aware_order(prompts)
    print(
        json.dumps(
            {
                "baseline_order": baseline,
                "cache_aware_order": optimized,
                "baseline_shared_prefix_score": score_order(prompts, baseline),
                "cache_aware_shared_prefix_score": score_order(prompts, optimized),
                "paper_to_system": paper_to_system_map(args.mechanism),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
