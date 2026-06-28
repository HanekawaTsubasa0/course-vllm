from course_vllm.engine.request import Request, RequestStatus, Sequence
from course_vllm.engine.sampler import SamplingParams
from course_vllm.engine.scheduler import BatchKind, Scheduler


def _seq(prompt_len: int) -> Sequence:
    request = Request(prompt="x", sampling_params=SamplingParams())
    return Sequence(request=request, prompt_token_ids=list(range(prompt_len)))


def test_scheduler_prefill_respects_token_budget():
    scheduler = Scheduler(max_num_seqs=4, max_num_batched_tokens=5)
    first = _seq(3)
    second = _seq(3)
    scheduler.add(first)
    scheduler.add(second)

    batch = scheduler.schedule()

    assert batch is not None
    assert batch.kind == BatchKind.PREFILL
    assert batch.sequences == [first]
    assert batch.num_tokens == 3
    assert len(scheduler.waiting) == 1


def test_scheduler_decode_after_prefill():
    scheduler = Scheduler(max_num_seqs=2, max_num_batched_tokens=8)
    first = _seq(2)
    second = _seq(2)
    scheduler.add(first)
    scheduler.add(second)

    prefill = scheduler.schedule()
    decode = scheduler.schedule()

    assert prefill is not None and prefill.kind == BatchKind.PREFILL
    assert decode is not None and decode.kind == BatchKind.DECODE
    assert decode.sequences == [first, second]


def test_scheduler_finish_removes_running_sequence():
    scheduler = Scheduler()
    seq = _seq(2)
    scheduler.add(seq)
    scheduler.schedule()
    scheduler.finish(seq)

    assert seq.status == RequestStatus.FINISHED
    assert not scheduler.has_unfinished()


def test_scheduler_chunked_prefill_splits_long_prompt():
    scheduler = Scheduler(max_num_seqs=2, max_num_batched_tokens=3, enable_chunked_prefill=True)
    seq = _seq(5)
    scheduler.add(seq)

    first = scheduler.schedule()

    assert first is not None and first.kind == BatchKind.PREFILL
    assert first.num_tokens == 3
    assert seq.scheduled_start == 0
    assert seq.scheduled_end == 3
    seq.prefill_offset = seq.scheduled_end

    second = scheduler.schedule()
    assert second is not None and second.kind == BatchKind.PREFILL
    assert second.num_tokens == 2
    assert seq.scheduled_start == 3
    assert seq.scheduled_end == 5
    seq.prefill_offset = seq.scheduled_end

    decode = scheduler.schedule()
    assert decode is not None and decode.kind == BatchKind.DECODE


def test_scheduler_preempt_moves_sequence_back_to_waiting():
    scheduler = Scheduler()
    seq = _seq(2)
    scheduler.add(seq)
    scheduler.schedule()

    scheduler.preempt(seq)

    assert seq.status == RequestStatus.WAITING
    assert list(scheduler.waiting) == [seq]
    assert not scheduler.running
