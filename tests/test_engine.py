import torch

from course_vllm.engine.engine import Engine
from course_vllm.engine.sampler import SamplingParams
from course_vllm.model.types import BatchModelOutput, ModelOutput


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


class FakeBatchBackend(FakeBackend):
    def __init__(self):
        self.prefill_batch_calls = 0
        self.decode_batch_calls = 0

    def prefill_batch(self, batch_token_ids: list[list[int]]):
        self.prefill_batch_calls += 1
        outputs = [self.prefill(token_ids) for token_ids in batch_token_ids]
        return BatchModelOutput(
            logits=[output.logits for output in outputs],
            past_key_values=[output.past_key_values for output in outputs],
        )

    def decode_batch(self, token_ids: list[int], past_key_values: list[dict]):
        self.decode_batch_calls += 1
        outputs = [
            self.decode_step(token_id, past_key_value)
            for token_id, past_key_value in zip(token_ids, past_key_values)
        ]
        return BatchModelOutput(
            logits=[output.logits for output in outputs],
            past_key_values=[output.past_key_values for output in outputs],
        )


class FakeEosBackend(FakeBackend):
    def prefill(self, token_ids: list[int]) -> ModelOutput:
        logits = torch.zeros(128)
        logits[ord("A")] = 1.0
        return ModelOutput(logits=logits, past_key_values={})

    def decode_step(self, token_id: int, past_key_values: dict) -> ModelOutput:
        logits = torch.zeros(128)
        logits[self.eos_token_id] = 1.0
        return ModelOutput(logits=logits, past_key_values=past_key_values)


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


def test_engine_default_generation_stops_on_eos_without_token_limit():
    engine = object.__new__(Engine)
    engine.backend = FakeEosBackend()
    engine.backend_name = "fake"

    result = engine.generate("x", SamplingParams(temperature=0.0))

    assert result["finish_reason"] == "eos"
    assert result["token_ids"] == [ord("A"), 99]


def test_engine_generate_batch_prefers_backend_prefill_batch():
    engine = object.__new__(Engine)
    engine.backend = FakeBatchBackend()
    engine.backend_name = "fake"

    engine.generate_batch(
        ["x", "y"],
        SamplingParams(temperature=0.0, max_tokens=1),
        max_num_seqs=2,
        max_num_batched_tokens=8,
    )

    assert engine.backend.prefill_batch_calls == 1


def test_engine_generate_batch_prefers_backend_decode_batch():
    engine = object.__new__(Engine)
    engine.backend = FakeBatchBackend()
    engine.backend_name = "fake"

    engine.generate_batch(
        ["x", "y"],
        SamplingParams(temperature=0.0, max_tokens=3),
        max_num_seqs=2,
        max_num_batched_tokens=8,
    )

    assert engine.backend.decode_batch_calls == 2


def test_engine_generate_batch_cache_aware_keeps_input_order():
    engine = object.__new__(Engine)
    engine.backend = FakeBatchBackend()
    engine.backend_name = "fake"

    results = engine.generate_batch(
        ["abcd", "x", "abce"],
        SamplingParams(temperature=0.0, max_tokens=1),
        max_num_seqs=3,
        max_num_batched_tokens=16,
        cache_aware_scheduling=True,
    )

    assert [result["text"] for result in results] == ["E", "B", "E"]
    assert engine.backend.prefill_batch_calls == 1


def test_engine_info_reports_course_stage_and_kernel_impl():
    engine = object.__new__(Engine)
    engine.backend = FakeBackend()
    engine.backend_name = "fake"
    engine.stage = "week04"
    engine.kernel_impl = "auto"
    engine.use_pinned_memory = False
    engine.use_transfer_stream = False

    info = engine.info()

    assert info["backend"] == "fake"
    assert info["kernel_impl"] == "auto"
    assert info["stage"]["key"] == "week04"
