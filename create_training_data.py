#!/usr/bin/env python
from __future__ import annotations

"""
Combines the processed CHILDES and BookCorpus datasets into a single
large training dataset in Parquet format.

This script performs the following steps:
1.  Reads the `childes_filtered.txt` file, sentencizes and tokenizes the text
    using spaCy.
2.  Reads all Parquet shards from the processed BookCorpus directory
    (`data/processed/refined-bookcorpus-dataset/`).
3.  Merges the tokenized sentences from both datasets.
4.  Saves the combined data as a single Parquet file with a 'tokens' column,
    where each row is a list of tokens representing a sentence.

Prerequisites:
- Run `process_childes_dataset.py` to generate `data/childes_filtered.txt`.
- Run `refined_book_dataset.py` to generate the BookCorpus Parquet shards.
- Install required packages: `pip install pandas pyarrow spacy tqdm`
"""

import argparse
import gc
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import spacy
from tqdm import tqdm


def _load_sentencizer():
    """Loads a spaCy model with a sentencizer."""
    print("Loading spaCy sentencizer...")
    nlp = spacy.blank("en")
    if not nlp.has_pipe("sentencizer"):
        nlp.add_pipe("sentencizer")
    # Set a high max_length to handle the single large text block from CHILDES
    nlp.max_length = 80_000_000
    print("spaCy model loaded.")
    return nlp


def _batch_list(data: list, batch_size: int):
    """Yield successive batch-sized chunks from a list."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def process_childes_data(childes_file: Path, nlp) -> list[list[str]]:
    """
    Reads the CHILDES text file, sentencizes and tokenizes it.
    """
    if not childes_file.exists():
        print(f"Warning: CHILDES file not found at {childes_file}. Skipping.")
        return []

    print(f"Processing CHILDES data from {childes_file}...")
    with open(childes_file, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)
    tokenized_sents = []
    for sent in tqdm(doc.sents, desc="Sentencizing CHILDES"):
        tokens = [t.text.lower() for t in sent if not t.is_space]
        if tokens:
            tokenized_sents.append(tokens)

    print(f"Processed {len(tokenized_sents):,} sentences from CHILDES data.")
    return tokenized_sents


def process_book_corpus_data(book_corpus_dir: Path) -> list[list[str]]:
    """
    Reads all Parquet files from the book corpus directory and combines them.
    """
    if not book_corpus_dir.exists():
        print(f"Warning: Book corpus directory not found at {book_corpus_dir}. Skipping.")
        return []

    parquet_files = sorted(book_corpus_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"Warning: No Parquet files found in {book_corpus_dir}. Skipping.")
        return []

    print(f"Found {len(parquet_files)} Parquet files in {book_corpus_dir}.")

    all_tokens = []
    for file_path in tqdm(parquet_files, desc="Loading book corpus shards"):
        df = pd.read_parquet(file_path)
        all_tokens.extend(df["tokens"].tolist())

    print(f"Loaded {len(all_tokens):,} sentences from book corpus.")
    return all_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Combine processed BookCorpus and CHILDES data into a single training dataset."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="combined_training_data.parquet",
        help="Name for the output Parquet file.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    book_corpus_dir = data_dir / "processed" / "refined-bookcorpus-dataset"
    childes_file = data_dir / "childes_filtered.txt"

    output_dir = data_dir / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / args.output_filename

    nlp = _load_sentencizer()
    childes_tokens = process_childes_data(childes_file, nlp)
    del nlp
    gc.collect()

    book_tokens = process_book_corpus_data(book_corpus_dir)

    print("Combining datasets...")
    combined_tokens = childes_tokens + book_tokens

    if not combined_tokens:
        print("No data to save. Exiting.")
        return

    print(f"Total sentences in combined dataset: {len(combined_tokens):,}")
    print(f"Saving combined dataset to {output_file}...")

    # Define schema for the Parquet file
    schema = pa.schema([pa.field("tokens", pa.list_(pa.string()))])

    # A reasonable batch size to balance memory and I/O overhead
    batch_size = 100_000
    num_batches = (len(combined_tokens) + batch_size - 1) // batch_size

    with pq.ParquetWriter(output_file, schema) as writer:
        # Create a generator for batches
        batches = _batch_list(combined_tokens, batch_size)
        # Wrap the generator with tqdm for a progress bar
        for batch in tqdm(batches, total=num_batches, desc="Saving to Parquet"):
            table = pa.Table.from_pydict({"tokens": batch}, schema=schema)
            writer.write_table(table)

    print("✅ Done.")


if __name__ == "__main__":
    main()