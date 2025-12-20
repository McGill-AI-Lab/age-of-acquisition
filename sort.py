from collections import deque

with open("all_sorted_by_phono.txt", "r", encoding="utf-8") as f:
    last_lines = deque(f, maxlen=100)

for line in last_lines:
    print(line, end="")