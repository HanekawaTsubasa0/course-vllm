import pytest
import torch
from torch.nn import functional as F

from course_vllm.engine.paged_kv_cache import PagedKVCache, PagedKVConfig
from course_vllm.kernels import benchmark_cuda, cuda_dense_attention_decode, cuda_paged_attention_decode
from course_vllm.model.attention import paged_attention_decode, paged_attention_decode_reference
from course_vllm.model.ops import dense_attention_decode, dense_attention_decode_reference
from course_vllm.model.qwen3_torch import repeat_kv


def _dense_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    key = repeat_kv(key.unsqueeze(0), query.shape[0] // key.shape[0]).squeeze(0)
    value = repeat_kv(value.unsqueeze(0), query.shape[0] // value.shape[0]).squeeze(0)
    scores = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1)).squeeze(1)
    weights = F.softmax((scores * scale).float(), dim=-1).to(query.dtype)
    return torch.matmul(weights.unsqueeze(1), value).squeeze(1)


def _cache(device: str | torch.device = "cpu") -> PagedKVCache:
    return PagedKVCache(
        PagedKVConfig(
            num_layers=1,
            num_blocks=8,
            block_size=3,
            num_kv_heads=2,
            head_dim=4,
            device=device,
        )
    )


def test_paged_attention_decode_matches_dense_attention_for_variable_lengths():
    torch.manual_seed(0)
    cache = _cache()
    seq_ids = [101, 102, 103]
    lengths = [1, 4, 6]
    dense_keys = []
    dense_values = []
    for seq_id, length in zip(seq_ids, lengths):
        key = torch.randn(1, 2, length, 4)
        value = torch.randn(1, 2, length, 4)
        cache.allocate(seq_id=seq_id, num_tokens=0)
        cache.append(seq_id=seq_id, layer_id=0, key=key, value=value)
        dense_keys.append(key.squeeze(0))
        dense_values.append(value.squeeze(0))

    query = torch.randn(len(seq_ids), 4, 4)
    actual = paged_attention_decode(
        query=query,
        key_cache=cache.key_cache[0],
        value_cache=cache.value_cache[0],
        block_tables=[cache.block_table(seq_id) for seq_id in seq_ids],
        context_lens=lengths,
        block_size=cache.config.block_size,
    )
    expected = torch.stack(
        [
            _dense_decode_attention(query[index], dense_keys[index], dense_values[index])
            for index in range(len(seq_ids))
        ]
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_paged_attention_decode_accepts_tensor_metadata():
    torch.manual_seed(1)
    cache = _cache()
    seq_ids = [201, 202]
    lengths = [3, 5]
    for seq_id, length in zip(seq_ids, lengths):
        cache.allocate(seq_id=seq_id, num_tokens=0)
        cache.append(
            seq_id=seq_id,
            layer_id=0,
            key=torch.randn(1, 2, length, 4),
            value=torch.randn(1, 2, length, 4),
        )
    max_blocks = max(len(cache.block_table(seq_id)) for seq_id in seq_ids)
    block_tables = torch.zeros(len(seq_ids), max_blocks, dtype=torch.long)
    for index, seq_id in enumerate(seq_ids):
        table = torch.tensor(cache.block_table(seq_id), dtype=torch.long)
        block_tables[index, : table.numel()] = table

    out = paged_attention_decode(
        query=torch.randn(2, 4, 4),
        key_cache=cache.key_cache[0],
        value_cache=cache.value_cache[0],
        block_tables=block_tables,
        context_lens=torch.tensor(lengths),
        block_size=cache.config.block_size,
    )

    assert out.shape == (2, 4, 4)


def test_paged_attention_decode_rejects_incomplete_block_table():
    cache = _cache()
    cache.allocate(seq_id=301, num_tokens=0)
    cache.append(
        seq_id=301,
        layer_id=0,
        key=torch.randn(1, 2, 4, 4),
        value=torch.randn(1, 2, 4, 4),
    )

    with pytest.raises(ValueError, match="block table"):
        paged_attention_decode(
            query=torch.randn(1, 4, 4),
            key_cache=cache.key_cache[0],
            value_cache=cache.value_cache[0],
            block_tables=[cache.block_table(301)[:1]],
            context_lens=[4],
            block_size=cache.config.block_size,
        )


def test_dense_attention_decode_reference_matches_torch_attention():
    torch.manual_seed(5)
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 4, 7, 8)
    value = torch.randn(2, 4, 7, 8)
    actual = dense_attention_decode_reference(query, key, value, scale=8**-0.5)
    scores = torch.matmul(query.unsqueeze(2), key.transpose(-2, -1)).squeeze(2) * 8**-0.5
    weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    expected = torch.matmul(weights.unsqueeze(2), value).squeeze(2)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_dense_attention_decode_dispatch_falls_back_on_cpu():
    query = torch.randn(1, 2, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    actual = dense_attention_decode(query, key, value, scale=0.5, kernel_impl="auto")
    expected = dense_attention_decode_reference(query, key, value, scale=0.5)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_cuda_dense_attention_decode_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(6)
    query = torch.randn(2, 4, 16, device="cuda")
    key = torch.randn(2, 4, 9, 16, device="cuda")
    value = torch.randn(2, 4, 9, 16, device="cuda")
    actual = cuda_dense_attention_decode(query, key, value)
    expected = dense_attention_decode_reference(query, key, value, scale=16**-0.5)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cuda_paged_attention_decode_matches_dense_attention():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(2)
    cache = _cache(device="cuda")
    seq_ids = [401, 402]
    lengths = [4, 6]
    dense_keys = []
    dense_values = []
    for seq_id, length in zip(seq_ids, lengths):
        key = torch.randn(1, 2, length, 4, device="cuda")
        value = torch.randn(1, 2, length, 4, device="cuda")
        cache.allocate(seq_id=seq_id, num_tokens=0)
        cache.append(seq_id=seq_id, layer_id=0, key=key, value=value)
        dense_keys.append(key.squeeze(0))
        dense_values.append(value.squeeze(0))

    query = torch.randn(len(seq_ids), 4, 4, device="cuda")
    actual = cuda_paged_attention_decode(
        query=query,
        key_cache=cache.key_cache[0],
        value_cache=cache.value_cache[0],
        block_tables=[cache.block_table(seq_id) for seq_id in seq_ids],
        context_lens=lengths,
        block_size=cache.config.block_size,
    )

    expected = torch.stack(
        [
            _dense_decode_attention(query[index], dense_keys[index], dense_values[index])
            for index in range(len(seq_ids))
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_cuda_paged_attention_decode_supports_bfloat16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported")
    torch.manual_seed(4)
    cache = PagedKVCache(
        PagedKVConfig(
            num_layers=1,
            num_blocks=16,
            block_size=4,
            num_kv_heads=2,
            head_dim=16,
            dtype=torch.bfloat16,
            device="cuda",
        )
    )
    seq_id = 501
    length = 8
    key = torch.randn(1, 2, length, 16, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 2, length, 16, device="cuda", dtype=torch.bfloat16)
    cache.allocate(seq_id=seq_id, num_tokens=0)
    cache.append(seq_id=seq_id, layer_id=0, key=key, value=value)
    query = torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16)
    actual = cuda_paged_attention_decode(
        query=query,
        key_cache=cache.key_cache[0],
        value_cache=cache.value_cache[0],
        block_tables=[cache.block_table(seq_id)],
        context_lens=[length],
        block_size=cache.config.block_size,
    )
    expected = _dense_decode_attention(query[0], key.squeeze(0), value.squeeze(0)).unsqueeze(0)
    assert actual.dtype == torch.bfloat16
    assert torch.allclose(actual.float(), expected.float(), atol=4e-3, rtol=4e-3)


def test_cuda_paged_attention_decode_not_much_slower_than_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(3)
    cache = PagedKVCache(
        PagedKVConfig(
            num_layers=1,
            num_blocks=128,
            block_size=16,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )
    )
    seq_ids = list(range(8))
    lengths = [96, 80, 64, 112, 72, 88, 104, 56]
    for seq_id, length in zip(seq_ids, lengths):
        cache.allocate(seq_id=seq_id, num_tokens=0)
        cache.append(
            seq_id=seq_id,
            layer_id=0,
            key=torch.randn(1, 4, length, 32, device="cuda"),
            value=torch.randn(1, 4, length, 32, device="cuda"),
        )
    query = torch.randn(len(seq_ids), 8, 32, device="cuda")
    block_tables = [cache.block_table(seq_id) for seq_id in seq_ids]
    actual = cuda_paged_attention_decode(
        query=query,
        key_cache=cache.key_cache[0],
        value_cache=cache.value_cache[0],
        block_tables=block_tables,
        context_lens=lengths,
        block_size=cache.config.block_size,
    )
    expected = paged_attention_decode_reference(
        query=query,
        key_cache=cache.key_cache[0],
        value_cache=cache.value_cache[0],
        block_tables=block_tables,
        context_lens=lengths,
        block_size=cache.config.block_size,
    )
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    cuda_ms = benchmark_cuda(
        lambda: cuda_paged_attention_decode(
            query=query,
            key_cache=cache.key_cache[0],
            value_cache=cache.value_cache[0],
            block_tables=block_tables,
            context_lens=lengths,
            block_size=cache.config.block_size,
        ),
        warmup=3,
        repeat=10,
    )
    torch_ms = benchmark_cuda(
        lambda: paged_attention_decode_reference(
            query=query,
            key_cache=cache.key_cache[0],
            value_cache=cache.value_cache[0],
            block_tables=block_tables,
            context_lens=lengths,
            block_size=cache.config.block_size,
        ),
        warmup=3,
        repeat=10,
    )
    assert cuda_ms < torch_ms * 3 + 0.05
