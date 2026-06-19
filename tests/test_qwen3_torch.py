import torch

from course_vllm.engine.kv_cache import ContinuousKVCache
from course_vllm.engine.paged_kv_cache import PagedKVCache, PagedKVConfig
from course_vllm.model.qwen3_backend import Qwen3PagedBackend, Qwen3TorchBackend
from course_vllm.model.qwen3_torch import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3KVCache,
    Qwen3RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )


def test_rms_norm_preserves_shape_and_dtype():
    norm = Qwen3RMSNorm(4, eps=1e-6).to(dtype=torch.bfloat16)
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    y = norm(x)
    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16


def test_repeat_kv_expands_kv_heads():
    x = torch.randn(2, 2, 3, 4)
    y = repeat_kv(x, num_groups=3)
    assert y.shape == (2, 6, 3, 4)
    assert torch.equal(y[:, 0], x[:, 0])
    assert torch.equal(y[:, 3], x[:, 1])


def test_rotary_embedding_keeps_qk_shapes():
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 2, 3, 8)
    cos = torch.randn(2, 3, 8)
    sin = torch.randn(2, 3, 8)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_tiny_qwen3_forward_shape():
    model = Qwen3ForCausalLM(tiny_config())
    input_ids = torch.tensor([[1, 2, 3]])
    logits = model(input_ids)
    assert logits.shape == (1, 3, 32)


def test_tiny_qwen3_cache_matches_full_forward():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    full_logits = model(input_ids)

    logits, cache = model.forward_with_cache(input_ids[:, :2])
    assert torch.allclose(logits[:, -1], full_logits[:, 1], atol=1e-5, rtol=1e-5)

    logits, cache = model.forward_with_cache(input_ids[:, 2:3], past_key_values=cache)
    assert torch.allclose(logits[:, -1], full_logits[:, 2], atol=1e-5, rtol=1e-5)

    logits, _ = model.forward_with_cache(input_ids[:, 3:4], past_key_values=cache)
    assert torch.allclose(logits[:, -1], full_logits[:, 3], atol=1e-5, rtol=1e-5)


def test_qwen3_backend_stores_incremental_tokens_in_continuous_cache():
    backend = object.__new__(Qwen3TorchBackend)
    backend.kv_cache = ContinuousKVCache()
    backend._cache_ids = iter([7])
    key = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1)
    value = key + 10

    handle = backend._store_cache(Qwen3KVCache(key_values=[(key[:, :, :3], value[:, :, :3])], seq_len=3))
    assert handle.seq_id == 7
    assert handle.seq_len == 3

    handle = backend._store_cache(
        Qwen3KVCache(key_values=[(key, value)], seq_len=4),
        seq_id=handle.seq_id,
        append_from=3,
    )

    restored = backend._load_cache(handle)
    restored_key, restored_value = restored.key_values[0]
    assert restored.seq_len == 4
    assert torch.equal(restored_key, key)
    assert torch.equal(restored_value, value)


def test_qwen3_backend_batch_prefill_matches_single_prefill():
    backend = object.__new__(Qwen3TorchBackend)
    backend.device = torch.device("cpu")
    backend.model = Qwen3ForCausalLM(tiny_config()).eval()
    backend.kv_cache = ContinuousKVCache()
    backend._cache_ids = iter(range(10))

    torch.manual_seed(0)
    batch = [[1, 2, 3], [4, 5, 6]]
    batch_out = backend.prefill_batch(batch)
    backend_single = object.__new__(Qwen3TorchBackend)
    backend_single.device = torch.device("cpu")
    backend_single.model = backend.model
    backend_single.kv_cache = ContinuousKVCache()
    backend_single._cache_ids = iter(range(10, 20))
    single_out = [backend_single.prefill(token_ids) for token_ids in batch]

    assert len(batch_out.logits) == 2
    for batch_logits, single in zip(batch_out.logits, single_out):
        assert torch.allclose(batch_logits, single.logits)


def test_qwen3_paged_backend_stores_layers_without_growing_length_per_layer():
    backend = object.__new__(Qwen3PagedBackend)
    backend._cache_ids = iter([11])
    backend.model = Qwen3ForCausalLM(tiny_config())
    backend.kv_cache = PagedKVCache(
        PagedKVConfig(
            num_layers=2,
            num_blocks=4,
            block_size=2,
            num_kv_heads=2,
            head_dim=4,
        )
    )
    first_key = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)
    first_value = first_key + 100
    second_key = first_key + 200
    second_value = first_key + 300

    handle = backend._store_cache(
        Qwen3KVCache(
            key_values=[
                (first_key[:, :, :3], first_value[:, :, :3]),
                (second_key[:, :, :3], second_value[:, :, :3]),
            ],
            seq_len=3,
        )
    )
    assert backend.kv_cache.block_manager.tables[handle.seq_id].length == 3

    handle = backend._store_cache(
        Qwen3KVCache(
            key_values=[
                (first_key, first_value),
                (second_key, second_value),
            ],
            seq_len=4,
        ),
        seq_id=handle.seq_id,
        append_from=3,
    )

    restored = backend._load_cache(handle)
    assert restored.seq_len == 4
    assert backend.kv_cache.block_manager.tables[handle.seq_id].length == 4
    assert torch.equal(restored.key_values[0][0], first_key)
    assert torch.equal(restored.key_values[1][0], second_key)
