def sample_blocks(path, n_first=30, n_each=10, n_last=30, n_blocks=20):
    """
    Print:
      - first n_first lines
      - n_each lines per block at regular intervals (n_blocks blocks total)
      - last n_last lines
    With markers for position in the file.
    """
    # Count total lines first
    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    interval = (total_lines - n_first - n_last) // n_blocks

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # First part
            if i < n_first:
                if i == 0:
                    print(f"\n--- START of file (first {n_first} lines) ---\n")
                print(f"[line {i+1}] {line.strip()}")

            # Middle blocks
            elif n_first <= i < total_lines - n_last:
                block_index = (i - n_first) // interval
                offset_in_block = (i - n_first) % interval
                if offset_in_block == 0:
                    print(f"\n--- Block {block_index+1} at line {i+1} ---\n")
                if offset_in_block < n_each and block_index < n_blocks:
                    print(f"[line {i+1}] {line.strip()}")

            # Last part
            elif i >= total_lines - n_last:
                if i == total_lines - n_last:
                    print(f"\n--- END of file (last {n_last} lines) ---\n")
                print(f"[line {i+1}] {line.strip()}")


if __name__ == "__main__":
    sample_blocks(r"phonology_curriculum_no_scores.txt")
