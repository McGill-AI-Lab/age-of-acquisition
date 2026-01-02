#!/usr/bin/env python
"""
Train Word2Vec on tranche-based curriculum and save embeddings per tranche.

This script:
1. Reads parquet shards from data/processed/corpora/curricula where each shard
   represents one tranche (a set of sentences)
2. Trains a single Word2Vec model on the entire corpus
3. Saves word embeddings organized by tranche - one parquet file per tranche
   containing all word embeddings that appear in that tranche

Expected input format:
    Each tranche parquet file should contain a 'text' or 'tokens' column.
    - If 'text': sentences as strings (will be tokenized)
    - If 'tokens': pre-tokenized lists of strings

Output format:
    One parquet file per tranche saved to the output directory with schema:
        - word: str (the word)
        - embedding: list[float] (the word vector)

Hyperparameters (defaults):
    - Epochs: 1
    - Vector size: 300 (also supports 50)
    - Context window: 5
    - Negative sampling: 20 negatives per positive
    - Minimum word count: 20
    - Learning rate: 0.025 (fixed, no decay)

Example usage:
    # Train with 300-dimensional vectors
    python -m src.training.train_word2vec_tranches \
        --input_dir data/processed/corpora/curricula \
        --output_dir outputs/embeddings/tranches_300d

    # Train with 50-dimensional vectors
    python -m src.training.train_word2vec_tranches \
        --input_dir data/processed/corpora/curricula \
        --output_dir outputs/embeddings/tranches_50d \
        --vector_size 50
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gensim.models import Word2Vec
from tqdm import tqdm


# Simple tokenizer pattern - matches word-like tokens including contractions
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


def simple_tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def discover_tranches(input_dir: Path) -> list[Path]:
    """
    Discover tranche parquet files in the input directory.
    
    Returns files sorted by name to ensure consistent tranche ordering.
    """
    parquet_files = sorted(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        # Check for partitioned dataset (directory with part-*.parquet files)
        part_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        parquet_files = part_dirs if part_dirs else []
    
    return parquet_files


def read_tranche_sentences(tranche_path: Path) -> Iterator[list[str]]:
    """
    Read sentences from a tranche parquet file.
    
    Handles both 'text' column (strings to tokenize) and 'tokens' column
    (pre-tokenized lists).
    """
    if tranche_path.is_dir():
        # Partitioned dataset
        dataset = pq.ParquetDataset(tranche_path)
        table = dataset.read()
    else:
        table = pq.read_table(tranche_path)
    
    columns = table.column_names
    
    if "tokens" in columns:
        # Pre-tokenized data
        tokens_list = table.column("tokens").to_pylist()
        for tokens in tokens_list:
            if tokens:
                yield [t.lower() for t in tokens]
    elif "text" in columns:
        # Raw text to tokenize
        texts = table.column("text").to_pylist()
        for text in texts:
            if text:
                tokens = simple_tokenize(text)
                if tokens:
                    yield tokens
    else:
        raise ValueError(
            f"Tranche {tranche_path} has no 'text' or 'tokens' column. "
            f"Found columns: {columns}"
        )


def collect_tranche_vocabulary(tranche_path: Path) -> set[str]:
    """Collect all unique words from a tranche."""
    vocab = set()
    for sentence in read_tranche_sentences(tranche_path):
        vocab.update(sentence)
    return vocab


class TrancheCorpusIterator:
    """
    Iterator over all tranches for Word2Vec training.
    
    Allows multiple passes over the corpus (required for Word2Vec training).
    """
    
    def __init__(self, tranche_paths: list[Path]):
        self.tranche_paths = tranche_paths
    
    def __iter__(self) -> Iterator[list[str]]:
        for tranche_path in self.tranche_paths:
            yield from read_tranche_sentences(tranche_path)


def save_embeddings_parquet(
    words: list[str],
    embeddings: np.ndarray,
    output_path: Path
) -> None:
    """
    Save word embeddings to a parquet file.
    
    Args:
        words: List of words
        embeddings: Array of shape (n_words, vector_size)
        output_path: Path to output parquet file
    """
    # Convert embeddings to list of lists for parquet storage
    embedding_lists = [emb.tolist() for emb in embeddings]
    
    table = pa.table({
        "word": pa.array(words, type=pa.string()),
        "embedding": pa.array(embedding_lists, type=pa.list_(pa.float32()))
    })
    
    pq.write_table(table, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec on tranches and save embeddings per tranche."
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed/corpora/curricula",
        help="Directory containing tranche parquet shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/embeddings/tranches",
        help="Directory to save per-tranche embedding parquet files.",
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=300,
        help="Dimensionality of word vectors (50 or 300).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Context window size.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=20,
        help="Minimum word frequency threshold.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads.",
    )
    parser.add_argument(
        "--sg",
        type=int,
        default=1,
        choices=[0, 1],
        help="Training algorithm: 1 for Skip-gram, 0 for CBOW.",
    )
    parser.add_argument(
        "--negative",
        type=int,
        default=20,
        help="Number of negative samples per positive sample.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.025,
        help="Learning rate (fixed throughout training).",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO
    )
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover tranches
    logging.info(f"Discovering tranches in {input_dir}")
    tranche_paths = discover_tranches(input_dir)
    
    if not tranche_paths:
        logging.error(f"No parquet files found in {input_dir}")
        return
    
    logging.info(f"Found {len(tranche_paths)} tranches:")
    for i, path in enumerate(tranche_paths):
        logging.info(f"  Tranche {i}: {path.name}")
    
    # Create corpus iterator
    corpus = TrancheCorpusIterator(tranche_paths)
    
    # Train Word2Vec
    logging.info("Training Word2Vec model...")
    logging.info(f"  vector_size={args.vector_size}")
    logging.info(f"  window={args.window}")
    logging.info(f"  min_count={args.min_count}")
    logging.info(f"  epochs={args.epochs}")
    logging.info(f"  sg={args.sg} ({'Skip-gram' if args.sg else 'CBOW'})")
    logging.info(f"  negative={args.negative}")
    logging.info(f"  alpha={args.alpha} (fixed)")
    
    model = Word2Vec(
        sentences=corpus,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        epochs=args.epochs,
        negative=args.negative,
        alpha=args.alpha,
        min_alpha=args.alpha,  # Keep learning rate constant (no decay)
    )
    
    vocab_size = len(model.wv)
    logging.info(f"Training complete. Vocabulary size: {vocab_size:,}")
    
    # Save model
    model_path = output_dir / "word2vec.model"
    model.save(str(model_path))
    logging.info(f"Saved model to {model_path}")
    
    # Extract and save embeddings per tranche
    logging.info("Extracting embeddings per tranche...")
    
    for i, tranche_path in enumerate(tqdm(tranche_paths, desc="Processing tranches")):
        tranche_name = tranche_path.stem if tranche_path.is_file() else tranche_path.name
        
        # Collect vocabulary for this tranche
        tranche_vocab = collect_tranche_vocabulary(tranche_path)
        
        # Filter to words in the model vocabulary
        words_in_model = [w for w in tranche_vocab if w in model.wv]
        
        if not words_in_model:
            logging.warning(f"Tranche {i} ({tranche_name}): No words in model vocabulary")
            continue
        
        # Extract embeddings
        embeddings = np.array([model.wv[w] for w in words_in_model])
        
        # Save to parquet
        output_path = output_dir / f"tranche_{i:04d}_{tranche_name}.parquet"
        save_embeddings_parquet(words_in_model, embeddings, output_path)
        
        logging.info(
            f"Tranche {i} ({tranche_name}): "
            f"{len(words_in_model):,} words saved to {output_path.name}"
        )
    
    logging.info("Done!")


if __name__ == "__main__":
    main()

