import torch

from course_vllm.model.qwen3_torch import (
    Qwen3Config,
    Qwen3ForCausalLM,
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
