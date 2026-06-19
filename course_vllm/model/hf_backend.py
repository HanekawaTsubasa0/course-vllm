from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(slots=True)
class ModelOutput:
    logits: torch.Tensor
    past_key_values: object | None


class HFModelBackend:
    """HuggingFace-backed model runner used as the first stable reference path."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = self._parse_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id

    def _parse_dtype(self, dtype: str) -> torch.dtype:
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

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def prefill(self, token_ids: list[int]) -> ModelOutput:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        return ModelOutput(logits=out.logits[0, -1], past_key_values=out.past_key_values)

    @torch.inference_mode()
    def decode_step(self, token_id: int, past_key_values: object) -> ModelOutput:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        return ModelOutput(logits=out.logits[0, -1], past_key_values=out.past_key_values)
