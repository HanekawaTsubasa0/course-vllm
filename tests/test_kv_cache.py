import torch

from course_vllm.engine.kv_cache import ContinuousKVCache


def test_continuous_kv_cache_appends_on_sequence_dimension():
    cache = ContinuousKVCache()
    first = torch.ones(1, 2, 1, 4)
    second = torch.full((1, 2, 1, 4), 2.0)

    cache.append(seq_id=3, layer_id=0, key=first, value=first)
    cache.append(seq_id=3, layer_id=0, key=second, value=second)

    layer = cache.get(seq_id=3, layer_id=0)
    assert layer.key.shape == (1, 2, 2, 4)
    assert torch.equal(layer.key[:, :, 0], first[:, :, 0])
    assert torch.equal(layer.key[:, :, 1], second[:, :, 0])


def test_continuous_kv_cache_release_sequence():
    cache = ContinuousKVCache()
    tensor = torch.ones(1, 1, 1, 1)
    cache.append(1, 0, tensor, tensor)
    cache.append(1, 1, tensor, tensor)
    cache.release(1)
    assert cache.num_layers_for(1) == 0
