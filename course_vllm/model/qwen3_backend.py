from __future__ import annotations

from itertools import count

import torch
from transformers import AutoTokenizer

from course_vllm.engine.kv_cache import ContinuousKVCache, KVCacheHandle
from course_vllm.engine.paged_kv_cache import PagedKVCache, PagedKVConfig
from course_vllm.model.qwen3_torch import Qwen3ForCausalLM, Qwen3KVCache
from course_vllm.model.types import BatchModelOutput, ModelOutput, bucket_by_length, parse_dtype


class Qwen3TorchBackend:
    """Course-owned Qwen3 runner with an explicit PyTorch KV-cache path."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = parse_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            device=self.device,
            dtype=self.dtype,
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        self.kv_cache = ContinuousKVCache()
        self._cache_ids = count()

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def prefill(self, token_ids: list[int]) -> ModelOutput:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        logits, cache = self.model.forward_with_cache(input_ids)
        handle = self._store_cache(cache)
        return ModelOutput(logits=logits[0, -1], past_key_values=handle)

    @torch.inference_mode()
    def prefill_batch(self, batch_token_ids: list[list[int]]) -> BatchModelOutput:
        if not batch_token_ids:
            return BatchModelOutput(logits=[], past_key_values=[])
        logits: list[torch.Tensor | None] = [None] * len(batch_token_ids)
        past_key_values: list[object | None] = [None] * len(batch_token_ids)
        for indices in bucket_by_length(batch_token_ids).values():
            bucket_token_ids = [batch_token_ids[index] for index in indices]
            bucket_out = self._prefill_same_length_batch(bucket_token_ids)
            for index, bucket_index in enumerate(indices):
                logits[bucket_index] = bucket_out.logits[index]
                past_key_values[bucket_index] = bucket_out.past_key_values[index]
        if any(item is None for item in logits):
            raise RuntimeError("internal error: missing batch logits")
        return BatchModelOutput(
            logits=[item for item in logits if item is not None],
            past_key_values=past_key_values,
        )

    @torch.inference_mode()
    def _prefill_same_length_batch(self, batch_token_ids: list[list[int]]) -> BatchModelOutput:
        input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
        logits, cache = self.model.forward_with_cache(input_ids)
        handles = [
            self._store_cache(self._slice_cache(cache, batch_index))
            for batch_index in range(len(batch_token_ids))
        ]
        return BatchModelOutput(
            logits=[logits[batch_index, -1] for batch_index in range(len(batch_token_ids))],
            past_key_values=handles,
        )

    @torch.inference_mode()
    def decode_step(self, token_id: int, past_key_values: KVCacheHandle) -> ModelOutput:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        cache = self._load_cache(past_key_values)
        logits, cache = self.model.forward_with_cache(input_ids, past_key_values=cache)
        handle = self._store_cache(
            cache,
            seq_id=past_key_values.seq_id,
            append_from=past_key_values.seq_len,
        )
        return ModelOutput(logits=logits[0, -1], past_key_values=handle)

    @torch.inference_mode()
    def decode_batch(
        self,
        token_ids: list[int],
        past_key_values: list[object | None],
    ) -> BatchModelOutput:
        outputs = [
            self.decode_step(token_id, past_key_value)
            for token_id, past_key_value in zip(token_ids, past_key_values)
        ]
        return BatchModelOutput(
            logits=[output.logits for output in outputs],
            past_key_values=[output.past_key_values for output in outputs],
        )

    def release_cache(self, past_key_values: object | None) -> None:
        if isinstance(past_key_values, KVCacheHandle):
            self.kv_cache.release(past_key_values.seq_id)

    def _store_cache(
        self,
        cache: Qwen3KVCache,
        seq_id: int | None = None,
        append_from: int = 0,
    ) -> KVCacheHandle:
        seq_id = next(self._cache_ids) if seq_id is None else seq_id
        for layer_id, (key, value) in enumerate(cache.key_values):
            self.kv_cache.append(
                seq_id=seq_id,
                layer_id=layer_id,
                key=key[:, :, append_from:, :],
                value=value[:, :, append_from:, :],
            )
        return KVCacheHandle(seq_id=seq_id, seq_len=cache.seq_len)

    def _load_cache(self, handle: KVCacheHandle) -> Qwen3KVCache:
        layer_kvs = self.kv_cache.layers(handle.seq_id)
        return Qwen3KVCache(
            key_values=[(layer.key, layer.value) for layer in layer_kvs],
            seq_len=handle.seq_len,
        )

    def _slice_cache(self, cache: Qwen3KVCache, batch_index: int) -> Qwen3KVCache:
        return Qwen3KVCache(
            key_values=[
                (key[batch_index : batch_index + 1], value[batch_index : batch_index + 1])
                for key, value in cache.key_values
            ],
            seq_len=cache.seq_len,
        )


class Qwen3PagedBackend(Qwen3TorchBackend):
    """Qwen3 runner that stores KV tensors in physical paged slots."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        trust_remote_code: bool = True,
        num_blocks: int = 512,
        block_size: int = 16,
    ):
        super().__init__(
            model_path,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        config = self.model.config
        self.kv_cache = PagedKVCache(
            PagedKVConfig(
                num_layers=config.num_hidden_layers,
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def release_cache(self, past_key_values: object | None) -> None:
        if isinstance(past_key_values, KVCacheHandle):
            self.kv_cache.release(past_key_values.seq_id)

    def _store_cache(
        self,
        cache: Qwen3KVCache,
        seq_id: int | None = None,
        append_from: int = 0,
    ) -> KVCacheHandle:
        if seq_id is None:
            seq_id = next(self._cache_ids)
            self.kv_cache.allocate(seq_id=seq_id, num_tokens=0)
        num_new_tokens = cache.seq_len - append_from
        positions = self.kv_cache.reserve(seq_id=seq_id, num_new_tokens=num_new_tokens)
        for layer_id, (key, value) in enumerate(cache.key_values):
            self.kv_cache.write(
                seq_id=seq_id,
                layer_id=layer_id,
                positions=positions,
                key=key[:, :, append_from:, :],
                value=value[:, :, append_from:, :],
            )
        return KVCacheHandle(seq_id=seq_id, seq_len=cache.seq_len)

    def _load_cache(self, handle: KVCacheHandle) -> Qwen3KVCache:
        key_values = [
            self.kv_cache.get_dense(seq_id=handle.seq_id, layer_id=layer_id)
            for layer_id in range(self.model.config.num_hidden_layers)
        ]
        return Qwen3KVCache(key_values=key_values, seq_len=handle.seq_len)
