import re
from itertools import islice

from complexity_ranker.phonology import PhonologicalComplexity

chunk_size = 50000
chunk_files = []
chunk = []
chunk_index = 0
start_line = chunk_index * chunk_size  # only matters if resuming

valid_count = 0  # count of sentences with score > 0

with open(r"refined_corpus_shuffled.txt", "r", encoding="utf-8") as f:
    f = islice(f, start_line, None)

    for line in f:
        line_clean = re.sub(r"[^\w\s']", " ", line)
        list_words = line_clean.strip().split()
        try:
            phonology_score = PhonologicalComplexity.get_phonology_stats(list_words).phonology_score
        except ValueError:
            continue  # skip bad lines silently

        if phonology_score == 0:
            continue  # skip zero-score sentences

        # add to current chunk
        chunk.append(f"{phonology_score}\t{line.strip()}\n")
        valid_count += 1

        # flush to file when we hit chunk_size valid sentences
        if valid_count % chunk_size == 0:
            chunk_file = f"chunk_{chunk_index}.txt"
            with open(chunk_file, "w", encoding="utf-8") as cf:
                cf.writelines(chunk)
            chunk_files.append(chunk_file)
            chunk = []
            chunk_index += 1

    # save leftover valid sentences (if any)
    if chunk:
        chunk_file = f"chunk_{chunk_index}.txt"
        with open(chunk_file, "w", encoding="utf-8") as cf:
            cf.writelines(chunk)
        chunk_files.append(chunk_file)
