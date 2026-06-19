import torch

from course_vllm.engine.engine import Engine
from course_vllm.engine.sampler import SamplingParams
from course_vllm.model.types import ModelOutput


class FakeBackend:
    eos_token_id = 99
    tokenizer = object()

    def encode(self, prompt: str) -> list[int]:
        return [ord(item) for item in prompt]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return messages[-1]["content"]

    def prefill(self, token_ids: list[int]) -> ModelOutput:
        logits = torch.zeros(128)
        logits[ord("A") + len(token_ids)] = 1.0
        return ModelOutput(logits=logits, past_key_values={"steps": 0})

    def decode_step(self, token_id: int, past_key_values: dict) -> ModelOutput:
        past_key_values["steps"] += 1
        logits = torch.zeros(128)
        logits[token_id + 1] = 1.0
        return ModelOutput(logits=logits, past_key_values=past_key_values)

    def release_cache(self, past_key_values) -> None:
        if isinstance(past_key_values, dict):
            past_key_values["released"] = True


def test_engine_generate_batch_uses_scheduler_for_multiple_prompts():
    engine = object.__new__(Engine)
    engine.backend = FakeBackend()
    engine.backend_name = "fake"

    results = engine.generate_batch(
        ["x", "yz"],
        SamplingParams(temperature=0.0, max_tokens=3),
        max_num_seqs=2,
        max_num_batched_tokens=8,
    )

    assert [result["text"] for result in results] == ["BCD", "CDE"]
    assert [result["finish_reason"] for result in results] == ["length", "length"]
