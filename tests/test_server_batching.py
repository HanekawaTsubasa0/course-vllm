import asyncio
import threading
import time

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

    def generate_stream(self, prompt, sampling_params):
        yield {"event": "token", "text": prompt.upper(), "token_id": 1}
        yield {"event": "finished", "finish_reason": "length", "token_ids": [1], "text": prompt.upper()}


class BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()

    def generate_batch(self, *args, **kwargs):
        self.started.set()
        time.sleep(0.2)
        return super().generate_batch(*args, **kwargs)


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
    assert batching.stats_dict()["total_requests"] == 2
    assert batching.stats_dict()["total_batches"] == 1
    assert batching.stats_dict()["max_observed_batch_size"] == 2


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
    assert batching.stats_dict()["total_batches"] == 2


def test_batching_engine_model_worker_does_not_block_event_loop():
    asyncio.run(_test_batching_engine_model_worker_does_not_block_event_loop())


async def _test_batching_engine_model_worker_does_not_block_event_loop():
    engine = BlockingEngine()
    batching = BatchingEngine(engine, max_batch_size=1, batch_wait_ms=0)
    task = asyncio.create_task(batching.generate("a", SamplingParams(max_tokens=1)))
    await asyncio.sleep(0.05)
    assert engine.started.is_set()
    assert not task.done()
    result = await task
    await batching.stop()
    assert result["text"] == "A"


def test_batching_engine_stream_uses_model_worker():
    asyncio.run(_test_batching_engine_stream_uses_model_worker())


async def _test_batching_engine_stream_uses_model_worker():
    batching = BatchingEngine(FakeEngine(), max_batch_size=1, batch_wait_ms=0)
    events = [event async for event in batching.stream("a", SamplingParams(max_tokens=1))]
    await batching.stop()
    assert [event["event"] for event in events] == ["token", "finished"]
    assert events[0]["text"] == "A"
