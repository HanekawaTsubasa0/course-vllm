from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class KVCacheHandle:
    seq_id: int
    seq_len: int


@dataclass(slots=True)
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


class ContinuousKVCache:
    """Per-sequence dense KV cache used before paged KV is introduced."""

    def __init__(self):
        self._cache: dict[tuple[int, int], LayerKV] = {}

    def append(self, seq_id: int, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        cache_key = (seq_id, layer_id)
        if cache_key not in self._cache:
            self._cache[cache_key] = LayerKV(key=key.clone(), value=value.clone())
            return
        layer = self._cache[cache_key]
        self._cache[cache_key] = LayerKV(
            key=torch.cat([layer.key, key], dim=-2),
            value=torch.cat([layer.value, value], dim=-2),
        )

    def get(self, seq_id: int, layer_id: int) -> LayerKV:
        return self._cache[(seq_id, layer_id)]

    def layers(self, seq_id: int) -> list[LayerKV]:
        layer_ids = sorted(layer_id for item_seq_id, layer_id in self._cache if item_seq_id == seq_id)
        return [self._cache[(seq_id, layer_id)] for layer_id in layer_ids]

    def contains(self, seq_id: int, layer_id: int) -> bool:
        return (seq_id, layer_id) in self._cache

    def release(self, seq_id: int) -> None:
        for key in [key for key in self._cache if key[0] == seq_id]:
            del self._cache[key]

    def num_layers_for(self, seq_id: int) -> int:
        return sum(1 for item in self._cache if item[0] == seq_id)
