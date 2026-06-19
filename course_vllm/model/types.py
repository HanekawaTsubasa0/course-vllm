from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    past_key_values: object | None


@dataclass(slots=True)
class BatchModelOutput:
    logits: list[torch.Tensor]
    past_key_values: list[object | None]


def parse_dtype(dtype: str) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[dtype]
