from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from course_vllm.model.qwen3_backend import Qwen3TorchBackend
from course_vllm.model.types import parse_dtype


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/wangqi/huggingface/Qwen3-0.6B")
    parser.add_argument("--prompt-token-ids", default="9707,0,21806|9707,0,14582")
    parser.add_argument("--decode-token-ids", default="279,25")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    prompt_token_ids = [
        [int(token_id) for token_id in row.split(",") if token_id]
        for row in args.prompt_token_ids.split("|")
        if row
    ]
    decode_token_ids = [int(token_id) for token_id in args.decode_token_ids.split(",") if token_id]
    if len(prompt_token_ids) != len(decode_token_ids):
        raise ValueError("number of prompts must match number of decode tokens")

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    course = Qwen3TorchBackend(args.model, device=str(device), dtype=args.dtype)
    course_prefill = course.prefill_batch(prompt_token_ids)
    course_decode = course.decode_batch(decode_token_ids, course_prefill.past_key_values)
    course_logits = torch.stack([logits.float().cpu() for logits in course_decode.logits])
    del course
    if device.type == "cuda":
        torch.cuda.empty_cache()

    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    hf.eval()
    input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=device)
    decode_ids = torch.tensor([[token_id] for token_id in decode_token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        hf_prefill = hf(input_ids=input_ids, use_cache=True)
        hf_batch_logits = hf(
            input_ids=decode_ids,
            past_key_values=hf_prefill.past_key_values,
            use_cache=True,
        ).logits[:, -1].float().cpu()

        hf_single_logits = []
        for prompt_ids, decode_id in zip(prompt_token_ids, decode_token_ids):
            single_prefill = hf(
                input_ids=torch.tensor([prompt_ids], dtype=torch.long, device=device),
                use_cache=True,
            )
            single_decode = hf(
                input_ids=torch.tensor([[decode_id]], dtype=torch.long, device=device),
                past_key_values=single_prefill.past_key_values,
                use_cache=True,
            )
            hf_single_logits.append(single_decode.logits[0, -1].float().cpu())
        hf_single_logits = torch.stack(hf_single_logits)

    report("course batch decode vs hf batch decode", course_logits, hf_batch_logits)
    report("hf batch decode vs hf single decode", hf_batch_logits, hf_single_logits)
    report("course batch decode vs hf single decode", course_logits, hf_single_logits)


def report(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs - rhs).abs()
    print(
        f"{name}: max_abs_diff={diff.max().item():.6f} "
        f"mean_abs_diff={diff.mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
