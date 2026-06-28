import pytest

from course_vllm.engine.block_manager import BlockManager


def test_block_manager_allocates_block_table_and_slots():
    manager = BlockManager(num_blocks=4, block_size=4)
    table = manager.allocate(seq_id=10, num_tokens=6)

    assert len(table.block_ids) == 2
    assert manager.num_free_blocks == 2

    slots = manager.slot_mapping(10, [0, 3, 4, 5])
    assert slots == [
        table.block_ids[0] * 4 + 0,
        table.block_ids[0] * 4 + 3,
        table.block_ids[1] * 4 + 0,
        table.block_ids[1] * 4 + 1,
    ]


def test_block_manager_appends_and_releases():
    manager = BlockManager(num_blocks=3, block_size=2)
    manager.allocate(seq_id=1, num_tokens=1)
    manager.append_tokens(seq_id=1, num_new_tokens=4)
    assert len(manager.block_table(1)) == 3

    manager.release(1)
    assert manager.num_free_blocks == 3


def test_block_manager_raises_when_out_of_blocks():
    manager = BlockManager(num_blocks=1, block_size=2)
    manager.allocate(seq_id=1, num_tokens=2)
    with pytest.raises(RuntimeError):
        manager.append_tokens(seq_id=1, num_new_tokens=1)


def test_block_manager_usage_stats_reports_fragmentation():
    manager = BlockManager(num_blocks=4, block_size=4)
    manager.allocate(seq_id=1, num_tokens=5)
    manager.allocate(seq_id=2, num_tokens=1)

    stats = manager.usage_stats()

    assert stats["active_sequences"] == 2
    assert stats["used_blocks"] == 3
    assert stats["live_tokens"] == 6
    assert stats["allocated_slots"] == 12
    assert stats["wasted_slots"] == 6
    assert stats["fragmentation_ratio"] == 0.5


def test_block_manager_reuses_full_prefix_blocks():
    manager = BlockManager(num_blocks=4, block_size=2)
    first = manager.allocate(seq_id=1, num_tokens=3, token_ids=[1, 2, 3])
    second = manager.allocate(seq_id=2, num_tokens=3, token_ids=[1, 2, 9])

    assert first.block_ids[0] == second.block_ids[0]
    assert manager.blocks[first.block_ids[0]].ref_count == 2
    assert manager.usage_stats()["prefix_cached_blocks"] == 1

    manager.release(1)
    assert manager.blocks[second.block_ids[0]].ref_count == 1


def test_block_manager_keeps_released_prefix_blocks_for_later_reuse():
    manager = BlockManager(num_blocks=4, block_size=2)
    first = manager.allocate(seq_id=1, num_tokens=2, token_ids=[1, 2])
    first_block = first.block_ids[0]
    manager.release(1)

    second = manager.allocate(seq_id=2, num_tokens=2, token_ids=[1, 2])

    assert second.block_ids[0] == first_block
    assert manager.blocks[first_block].ref_count == 1
    assert manager.usage_stats()["cached_free_blocks"] == 0
