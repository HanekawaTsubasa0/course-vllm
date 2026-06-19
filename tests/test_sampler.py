import torch

from course_vllm.engine.sampler import Sampler, SamplingParams


def test_greedy_sampler_returns_argmax():
    sampler = Sampler(SamplingParams(temperature=0.0))
    assert sampler.sample(torch.tensor([0.1, 2.0, 1.0])) == 1


def test_seeded_sampler_is_repeatable():
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
    first = Sampler(SamplingParams(temperature=1.0, seed=7)).sample(logits)
    second = Sampler(SamplingParams(temperature=1.0, seed=7)).sample(logits)
    assert first == second


def test_sampling_params_allow_unbounded_generation():
    assert SamplingParams().max_tokens is None
