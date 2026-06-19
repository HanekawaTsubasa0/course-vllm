from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil


@dataclass(slots=True)
class Block:
    block_id: int
    ref_count: int = 0
    token_hash: int | None = None

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


@dataclass(slots=True)
class BlockTable:
    block_size: int
    block_ids: list[int] = field(default_factory=list)
    length: int = 0

    def required_blocks(self, new_length: int) -> int:
        return ceil(new_length / self.block_size) if new_length > 0 else 0


class BlockManager:
    """Small vLLM-style block allocator for paged KV experiments."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.block_size = block_size
        self.blocks = [Block(block_id=i) for i in range(num_blocks)]
        self.free_block_ids = list(range(num_blocks))
        self.tables: dict[int, BlockTable] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return len(self.blocks) - self.num_free_blocks

    def allocate(self, seq_id: int, num_tokens: int) -> BlockTable:
        if seq_id in self.tables:
            raise ValueError(f"sequence {seq_id} already has a block table")
        table = BlockTable(block_size=self.block_size)
        self.tables[seq_id] = table
        self.ensure_capacity(seq_id, num_tokens)
        table.length = num_tokens
        return table

    def ensure_capacity(self, seq_id: int, new_length: int) -> BlockTable:
        table = self.tables[seq_id]
        required = table.required_blocks(new_length)
        missing = required - len(table.block_ids)
        if missing <= 0:
            table.length = max(table.length, new_length)
            return table
        if missing > self.num_free_blocks:
            raise RuntimeError(
                f"not enough KV blocks: need {missing}, free {self.num_free_blocks}"
            )
        for _ in range(missing):
            block_id = self.free_block_ids.pop()
            block = self.blocks[block_id]
            block.ref_count = 1
            table.block_ids.append(block_id)
        table.length = max(table.length, new_length)
        return table

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> BlockTable:
        table = self.tables[seq_id]
        return self.ensure_capacity(seq_id, table.length + num_new_tokens)

    def block_table(self, seq_id: int) -> list[int]:
        return list(self.tables[seq_id].block_ids)

    def slot_mapping(self, seq_id: int, positions: list[int]) -> list[int]:
        table = self.tables[seq_id]
        slots: list[int] = []
        for position in positions:
            if position < 0 or position >= table.length:
                raise IndexError(f"position {position} outside sequence length {table.length}")
            block_index = position // self.block_size
            block_offset = position % self.block_size
            slots.append(table.block_ids[block_index] * self.block_size + block_offset)
        return slots

    def release(self, seq_id: int) -> None:
        table = self.tables.pop(seq_id)
        for block_id in table.block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count < 0:
                raise RuntimeError(f"negative ref_count for block {block_id}")
            if block.ref_count == 0:
                block.token_hash = None
                self.free_block_ids.append(block_id)
