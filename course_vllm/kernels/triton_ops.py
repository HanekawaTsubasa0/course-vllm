from __future__ import annotations

from collections.abc import Sequence

import torch

from course_vllm.kernels.harness import KernelUnavailable

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - depends on optional runtime.
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: Exception | None = exc
else:
    _TRITON_IMPORT_ERROR = None


if triton is not None:

    @triton.jit
    def _softmax_kernel(x, out, n_cols: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        values = tl.load(x + row * n_cols + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        values = values - tl.max(values, axis=0)
        numerator = tl.exp(values)
        result = numerator / tl.sum(numerator, axis=0)
        tl.store(out + row * n_cols + offsets, result, mask=mask)

    @triton.jit
    def _rms_norm_kernel(x, weight, out, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        values = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / n_cols
        result = values * tl.rsqrt(variance + eps) * weights
        tl.store(out + row * n_cols + offsets, result, mask=mask)

    @triton.jit
    def _rope_kernel(x, cos, sin, out, n_cols: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        half = n_cols // 2
        mask = offsets < n_cols
        rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        sign = tl.where(offsets < half, -1.0, 1.0)
        values = tl.load(x + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        rotated = tl.load(x + row * n_cols + rotated_offsets, mask=mask, other=0.0).to(tl.float32)
        cos_values = tl.load(cos + row * n_cols + offsets, mask=mask, other=1.0).to(tl.float32)
        sin_values = tl.load(sin + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + row * n_cols + offsets, values * cos_values + sign * rotated * sin_values, mask=mask)

    @triton.jit
    def _matmul_kernel(
        a,
        b,
        out,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            a_values = tl.load(
                a + offs_m[:, None] * K + k_idx[None, :],
                mask=(offs_m[:, None] < M) & (k_idx[None, :] < K),
                other=0.0,
            )
            b_values = tl.load(
                b + k_idx[:, None] * N + offs_n[None, :],
                mask=(k_idx[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a_values, b_values)
        tl.store(
            out + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _paged_attention_decode_kernel(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        out,
        scale: tl.constexpr,
        H_Q: tl.constexpr,
        H_KV: tl.constexpr,
        D: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_CONTEXT: tl.constexpr,
        BLOCK_T: tl.constexpr,
        D_BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0)
        head = tl.program_id(1)
        group_size = H_Q // H_KV
        kv_head = head // group_size
        d_offsets = tl.arange(0, D_BLOCK)
        d_mask = d_offsets < D
        q = tl.load(query + (batch * H_Q + head) * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        context_len = tl.load(context_lens + batch)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((D_BLOCK,), tl.float32)
        token_offsets = tl.arange(0, BLOCK_T)
        for start in range(0, MAX_CONTEXT, BLOCK_T):
            positions = start + token_offsets
            valid_tokens = positions < context_len
            table_offsets = positions // BLOCK_SIZE
            block_ids = tl.load(
                block_tables + batch * MAX_BLOCKS + table_offsets,
                mask=valid_tokens,
                other=0,
            )
            slots = block_ids * BLOCK_SIZE + positions % BLOCK_SIZE
            kv_offsets = (slots[:, None] * H_KV + kv_head) * D + d_offsets[None, :]
            mask = valid_tokens[:, None] & d_mask[None, :]
            keys = tl.load(key_cache + kv_offsets, mask=mask, other=0.0).to(tl.float32)
            values = tl.load(value_cache + kv_offsets, mask=mask, other=0.0).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            scores = tl.where(valid_tokens, scores, -float("inf"))

            m_next = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_next)
            probs = tl.exp(scores - m_next)
            acc = acc * alpha + tl.sum(values * probs[:, None], axis=0)
            l_i = l_i * alpha + tl.sum(probs, axis=0)
            m_i = m_next

        tl.store(out + (batch * H_Q + head) * D + d_offsets, acc / l_i, mask=d_mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    x = _require_2d_cuda("x", x)
    rows, cols = x.shape
    out = torch.empty_like(x)
    block = _next_power_of_2(cols)
    _softmax_kernel[(rows,)](x, out, cols, BLOCK=block)
    return out


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    x = _require_2d_cuda("x", x)
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight must have shape [hidden_size]")
    weight = weight.to(device=x.device).contiguous()
    rows, cols = x.shape
    out = torch.empty_like(x)
    _rms_norm_kernel[(rows,)](x, weight, out, cols, eps, BLOCK=_next_power_of_2(cols))
    return out


def triton_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = _require_2d_cuda("x", x)
    if x.shape[1] % 2 != 0:
        raise ValueError("RoPE head dimension must be even")
    cos = _rope_table("cos", cos, x)
    sin = _rope_table("sin", sin, x)
    rows, cols = x.shape
    out = torch.empty_like(x)
    _rope_kernel[(rows,)](x, cos, sin, out, cols, BLOCK=_next_power_of_2(cols))
    return out


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _require_2d_cuda("a", a)
    b = _require_2d_cuda("b", b)
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 16))
    _matmul_kernel[grid](a, b, out, M=m, N=n, K=k, BLOCK_M=16, BLOCK_N=16, BLOCK_K=32)
    return out


def triton_paged_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: Sequence[Sequence[int]] | torch.Tensor,
    context_lens: Sequence[int] | torch.Tensor,
    block_size: int,
    *,
    scale: float | None = None,
    block_t: int = 64,
) -> torch.Tensor:
    _require_triton()
    if not query.is_cuda:
        raise KernelUnavailable("paged attention Triton kernel requires CUDA tensors")
    if query.ndim != 3 or key_cache.ndim != 3 or value_cache.shape != key_cache.shape:
        raise ValueError("expected query [batch, heads, dim] and KV cache [slots, kv_heads, dim]")
    batch, num_heads, head_dim = query.shape
    _, num_kv_heads, kv_dim = key_cache.shape
    if head_dim != kv_dim or num_heads % num_kv_heads != 0:
        raise ValueError("query heads/dim must match grouped KV cache")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    query = query.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    tables, lens = _metadata_tensors(block_tables, context_lens, batch, query.device)
    scale = head_dim**-0.5 if scale is None else scale
    max_context = int(tables.shape[1]) * block_size
    if max_context <= 0:
        raise ValueError("block_tables must contain at least one block")
    if int(lens.min().item()) <= 0:
        raise ValueError("context lengths must be > 0")
    if int(lens.max().item()) > max_context:
        raise ValueError("block table does not cover context length")

    out = torch.empty_like(query)
    d_block = _next_power_of_2(head_dim)
    if d_block > 256:
        raise KernelUnavailable("teaching paged attention kernel supports head_dim <= 256")
    _paged_attention_decode_kernel[(batch, num_heads)](
        query,
        key_cache,
        value_cache,
        tables,
        lens,
        out,
        scale,
        H_Q=num_heads,
        H_KV=num_kv_heads,
        D=head_dim,
        MAX_BLOCKS=int(tables.shape[1]),
        BLOCK_SIZE=block_size,
        MAX_CONTEXT=max_context,
        BLOCK_T=block_t,
        D_BLOCK=d_block,
    )
    return out


def _require_triton() -> None:
    if triton is None:
        raise KernelUnavailable(f"Triton is not available: {_TRITON_IMPORT_ERROR}")
    if not torch.cuda.is_available():
        raise KernelUnavailable("CUDA is not available")


def _require_2d_cuda(name: str, tensor: torch.Tensor) -> torch.Tensor:
    _require_triton()
    if not tensor.is_cuda:
        raise KernelUnavailable(f"{name} must be a CUDA tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D")
    return tensor.contiguous()


def _rope_table(name: str, table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if table.ndim == 1:
        table = table.reshape(1, -1).expand(x.shape[0], -1)
    if table.shape != x.shape:
        raise ValueError(f"{name} must have shape [dim] or match x")
    return table.to(device=x.device).contiguous()


def _metadata_tensors(
    block_tables: Sequence[Sequence[int]] | torch.Tensor,
    context_lens: Sequence[int] | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(block_tables):
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            raise ValueError("block_tables tensor must have shape [batch, max_blocks]")
        tables = block_tables.to(device=device, dtype=torch.int64).contiguous()
    else:
        if len(block_tables) != batch_size:
            raise ValueError("block_tables batch size must match query batch size")
        max_blocks = max(len(table) for table in block_tables)
        tables = torch.zeros((batch_size, max_blocks), dtype=torch.int64, device=device)
        for index, table in enumerate(block_tables):
            tables[index, : len(table)] = torch.tensor(table, dtype=torch.int64, device=device)

    if torch.is_tensor(context_lens):
        if context_lens.ndim != 1 or context_lens.shape[0] != batch_size:
            raise ValueError("context_lens tensor must have shape [batch]")
        lens = context_lens.to(device=device, dtype=torch.int64).contiguous()
    else:
        if len(context_lens) != batch_size:
            raise ValueError("context_lens batch size must match query batch size")
        lens = torch.tensor(context_lens, dtype=torch.int64, device=device)
    return tables, lens


def _next_power_of_2(value: int) -> int:
    return int(triton.next_power_of_2(value))


__all__ = [
    "triton_matmul",
    "triton_paged_attention_decode",
    "triton_rms_norm",
    "triton_rope",
    "triton_softmax",
]
