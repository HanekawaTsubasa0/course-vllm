import torch

from course_vllm.engine.paged_kv_cache import PagedKVCache, PagedKVConfig


def _cache() -> PagedKVCache:
    return PagedKVCache(
        PagedKVConfig(
            num_layers=2,
            num_blocks=4,
            block_size=2,
            num_kv_heads=2,
            head_dim=3,
        )
    )


def test_paged_kv_cache_writes_and_reads_dense_across_blocks():
    cache = _cache()
    key = torch.arange(18, dtype=torch.float32).view(1, 2, 3, 3)
    value = key + 100

    cache.allocate(seq_id=1, num_tokens=0)
    cache.append(seq_id=1, layer_id=0, key=key, value=value)

    restored_key, restored_value = cache.get_dense(seq_id=1, layer_id=0)
    assert torch.equal(restored_key, key)
    assert torch.equal(restored_value, value)
    assert len(cache.block_table(seq_id=1)) == 2


def test_paged_kv_cache_appends_decode_token():
    cache = _cache()
    prompt_key = torch.arange(12, dtype=torch.float32).view(1, 2, 2, 3)
    prompt_value = prompt_key + 10
    decode_key = torch.full((1, 2, 1, 3), 42.0)
    decode_value = torch.full((1, 2, 1, 3), 43.0)

    cache.allocate(seq_id=2, num_tokens=0)
    cache.append(seq_id=2, layer_id=0, key=prompt_key, value=prompt_value)
    cache.append(seq_id=2, layer_id=0, key=decode_key, value=decode_value)

    restored_key, restored_value = cache.get_dense(seq_id=2, layer_id=0)
    assert torch.equal(restored_key, torch.cat([prompt_key, decode_key], dim=-2))
    assert torch.equal(restored_value, torch.cat([prompt_value, decode_value], dim=-2))


def test_paged_kv_cache_keeps_layers_separate():
    cache = _cache()
    layer_0 = torch.zeros(1, 2, 1, 3)
    layer_1 = torch.ones(1, 2, 1, 3)

    cache.allocate(seq_id=3, num_tokens=0)
    cache.append(seq_id=3, layer_id=0, key=layer_0, value=layer_0)
    cache.write(seq_id=3, layer_id=1, positions=[0], key=layer_1, value=layer_1)

    restored_0, _ = cache.get_dense(seq_id=3, layer_id=0)
    restored_1, _ = cache.get_dense(seq_id=3, layer_id=1)
    assert torch.equal(restored_0, layer_0)
    assert torch.equal(restored_1, layer_1)


def test_paged_kv_cache_release_returns_blocks():
    cache = _cache()
    tensor = torch.ones(1, 2, 3, 3)

    cache.allocate(seq_id=4, num_tokens=0)
    cache.append(seq_id=4, layer_id=0, key=tensor, value=tensor)
    assert cache.block_manager.num_free_blocks == 2

    cache.release(seq_id=4)
    assert cache.block_manager.num_free_blocks == 4
