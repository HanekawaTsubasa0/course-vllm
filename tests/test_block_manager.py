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
