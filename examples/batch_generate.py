from __future__ import annotations

import argparse

from course_vllm import Engine, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/wangqi/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompts", default="Hello|What is KV cache?")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--backend", default="paged", choices=["hf", "course", "paged"])
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    args = parser.parse_args()

    prompts = [prompt for prompt in args.prompts.split("|") if prompt]
    engine = Engine(args.model, dtype=args.dtype, backend=args.backend)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    results = engine.generate_batch(
        prompts,
        params,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    for prompt, result in zip(prompts, results):
        print(f"prompt={prompt!r}")
        print(result["text"])
        print(f"finish_reason={result['finish_reason']} token_ids={result['token_ids']}")


if __name__ == "__main__":
    main()
