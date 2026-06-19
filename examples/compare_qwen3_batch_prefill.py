from __future__ import annotations

import argparse

import torch

from course_vllm.model.qwen3_backend import Qwen3PagedBackend, Qwen3TorchBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/wangqi/huggingface/Qwen3-0.6B")
    parser.add_argument("--token-ids", default="9707,0,21806|9707,0,14582")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--backend", default="paged", choices=["course", "paged"])
    args = parser.parse_args()

    batch_token_ids = [
        [int(token_id) for token_id in row.split(",") if token_id]
        for row in args.token_ids.split("|")
        if row
    ]
    backend_cls = Qwen3PagedBackend if args.backend == "paged" else Qwen3TorchBackend
    batched_backend = backend_cls(args.model, device=args.device, dtype=args.dtype)
    batch_out = batched_backend.prefill_batch(batch_token_ids)
    batch_logits = [logits.float().cpu() for logits in batch_out.logits]
    del batched_backend
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()

    single_backend = backend_cls(args.model, device=args.device, dtype=args.dtype)
    single_logits = [
        single_backend.prefill(token_ids).logits.float().cpu()
        for token_ids in batch_token_ids
    ]

    for index, (batch_step_logits, single_step_logits) in enumerate(zip(batch_logits, single_logits)):
        diff = (batch_step_logits - single_step_logits).abs()
        print(
            f"item={index} max_abs_diff={diff.max().item():.6f} "
            f"mean_abs_diff={diff.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
