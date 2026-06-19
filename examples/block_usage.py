from __future__ import annotations

import argparse

from course_vllm.engine.block_manager import BlockManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--prompt-lens", default="3,6,9")
    parser.add_argument("--decode-steps", type=int, default=2)
    args = parser.parse_args()

    prompt_lens = [int(item) for item in args.prompt_lens.split(",") if item]
    manager = BlockManager(num_blocks=args.num_blocks, block_size=args.block_size)

    print(f"num_blocks={args.num_blocks} block_size={args.block_size}")
    print("prefill")
    for seq_id, prompt_len in enumerate(prompt_lens):
        manager.allocate(seq_id=seq_id, num_tokens=prompt_len)
        print_sequence(manager, seq_id)

    print(f"used={manager.num_used_blocks} free={manager.num_free_blocks}")
    print("decode")
    for step in range(args.decode_steps):
        print(f"step={step + 1}")
        for seq_id in range(len(prompt_lens)):
            manager.append_tokens(seq_id=seq_id, num_new_tokens=1)
            print_sequence(manager, seq_id)
        print(f"used={manager.num_used_blocks} free={manager.num_free_blocks}")


def print_sequence(manager: BlockManager, seq_id: int) -> None:
    table = manager.tables[seq_id]
    positions = list(range(table.length))
    slots = manager.slot_mapping(seq_id, positions) if positions else []
    print(
        f"  seq={seq_id} len={table.length} "
        f"blocks={table.block_ids} slots={slots}"
    )


if __name__ == "__main__":
    main()
