import asyncio

from course_vllm.engine.sampler import SamplingParams
from course_vllm.server.batching import BatchingEngine


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate_batch(self, prompts, sampling_params, *, max_num_seqs, max_num_batched_tokens):
        self.calls.append((list(prompts), sampling_params, max_num_seqs, max_num_batched_tokens))
        return [
            {
                "text": prompt.upper(),
                "token_ids": [index],
                "finish_reason": "length",
            }
            for index, prompt in enumerate(prompts)
        ]


def test_batching_engine_groups_matching_sampling_params():
    asyncio.run(_test_batching_engine_groups_matching_sampling_params())


async def _test_batching_engine_groups_matching_sampling_params():
    engine = FakeEngine()
    batching = BatchingEngine(engine, max_batch_size=4, batch_wait_ms=1)
    params = SamplingParams(temperature=0.0, max_tokens=2)
    first, second = await asyncio.gather(
        batching.generate("a", params),
        batching.generate("b", params),
    )
    await batching.stop()

    assert first["text"] == "A"
    assert second["text"] == "B"
    assert len(engine.calls) == 1
    assert engine.calls[0][0] == ["a", "b"]


def test_batching_engine_separates_different_sampling_params():
    asyncio.run(_test_batching_engine_separates_different_sampling_params())


async def _test_batching_engine_separates_different_sampling_params():
    engine = FakeEngine()
    batching = BatchingEngine(engine, max_batch_size=4, batch_wait_ms=1)
    first, second = await asyncio.gather(
        batching.generate("a", SamplingParams(temperature=0.0, max_tokens=2)),
        batching.generate("b", SamplingParams(temperature=0.8, max_tokens=2)),
    )
    await batching.stop()

    assert first["text"] == "A"
    assert second["text"] == "B"
    assert [call[0] for call in engine.calls] == [["a"], ["b"]]
