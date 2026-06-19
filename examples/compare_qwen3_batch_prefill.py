from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from course_vllm.model.qwen3_backend import Qwen3PagedBackend, Qwen3TorchBackend
from course_vllm.model.types import parse_dtype


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
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    backend_cls = Qwen3PagedBackend if args.backend == "paged" else Qwen3TorchBackend
    batched_backend = backend_cls(args.model, device=str(device), dtype=args.dtype)
    batch_out = batched_backend.prefill_batch(batch_token_ids)
    course_batch_logits = torch.stack([logits.float().cpu() for logits in batch_out.logits])
    del batched_backend
    if device.type == "cuda":
        torch.cuda.empty_cache()

    single_backend = backend_cls(args.model, device=str(device), dtype=args.dtype)
    course_single_logits = torch.stack([
        single_backend.prefill(token_ids).logits.float().cpu()
        for token_ids in batch_token_ids
    ])
    del single_backend
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    hf_model.eval()
    input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        hf_batch_logits = hf_model(input_ids=input_ids, use_cache=True).logits[:, -1].float().cpu()
        hf_single_logits = torch.stack([
            hf_model(
                input_ids=torch.tensor([token_ids], dtype=torch.long, device=device),
                use_cache=True,
            ).logits[0, -1].float().cpu()
            for token_ids in batch_token_ids
        ])

    report("course batch vs course single", course_batch_logits, course_single_logits)
    report("hf batch vs hf single", hf_batch_logits, hf_single_logits)
    report("course batch vs hf batch", course_batch_logits, hf_batch_logits)
    report("course single vs hf single", course_single_logits, hf_single_logits)


def report(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs - rhs).abs()
    print(
        f"{name}: max_abs_diff={diff.max().item():.6f} "
        f"mean_abs_diff={diff.mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
