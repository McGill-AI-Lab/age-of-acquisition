input_file = "phonology_curriculum.txt"
output_file = "phono_curriculum.txt"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        # split at the first tab
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            fout.write(parts[1] + "\n")