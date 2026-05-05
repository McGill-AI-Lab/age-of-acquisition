"""
Generate normalized word-pair distance deltas from embedding tranche parquet files.

Expected input:
  embeddings/
    aoa_50d_0/
    conc_50d_0/
    freq_50d_0/
    phon_50d_0/
    ...
    aoa_50d_4/
    ...

Expected output:
  deltas/
    run0/
      word_pairs.txt
      aoa/tranche_0001.parquet
      conc/tranche_0001.parquet
      ...
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd


CURRICULA = ["aoa", "conc", "freq", "phon"]

def read_embedding_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if "word" not in df.columns:
        raise ValueError(f"{path} must contain a 'word' column.")

    return df

def get_vector_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "word"]

def get_vocab_from_file(path: Path) -> set[str]:
    df = pd.read_parquet(path, columns=["word"])
    return set(df["word"].astype(str))

def update_embedding_cache(
    embedding_cache: dict[str, np.ndarray],
    tranche_file: Path,
) -> None:
    df = read_embedding_parquet(tranche_file)
    df["word"] = df["word"].astype(str)

    if "embedding" not in df.columns:
        raise ValueError(f"{tranche_file} must contain an 'embedding' column.")

    for word, emb in zip(df["word"], df["embedding"]):
        embedding_cache[word] = np.asarray(emb, dtype=np.float32)

def build_shared_vocab(embeddings_dir: Path, dim: int) -> list[str]:
    vocab_by_curriculum: dict[str, set[str]] = {}

    for curriculum in CURRICULA:
        curriculum_vocab: set[str] = set()

        folder = embeddings_dir / f"{curriculum}_{dim}d_{0}"
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        tranche_files = sorted(folder.glob("tranche_*.parquet"))
        if not tranche_files:
            raise FileNotFoundError(f"No tranche parquet files found in {folder}")

        for tranche_file in tqdm(tranche_files, desc=f"Scanning {curriculum}"):
            vocab = get_vocab_from_file(tranche_file)
            curriculum_vocab.update(vocab)

        vocab_by_curriculum[curriculum] = curriculum_vocab
        print(f"{curriculum}: {len(curriculum_vocab):,} unique words")

    shared_vocab = set.intersection(*vocab_by_curriculum.values())

    if not shared_vocab:
        raise ValueError("Shared vocabulary across curricula is empty.")

    return sorted(shared_vocab)

def sample_word_pairs(
    vocab: list[str],
    n_pairs: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)

    if len(vocab) < 2:
        raise ValueError("Vocabulary must contain at least two words.")

    seen = set()
    pairs = []

    max_possible_pairs = len(vocab) * (len(vocab) - 1) // 2
    if n_pairs > max_possible_pairs:
        raise ValueError(
            f"Requested {n_pairs} pairs, but only {max_possible_pairs} unique pairs "
            f"are possible from vocabulary size {len(vocab)}."
        )

    while len(pairs) < n_pairs:
        w1, w2 = rng.sample(vocab, 2)
        pair = tuple(sorted((w1, w2)))

        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    return pairs

def write_word_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("word_i\tword_j\n")
        for w1, w2 in pairs:
            f.write(f"{w1}\t{w2}\n")

def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)

def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)

def compute_deltas(
    embedding_cache: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    metric: str,
) -> pd.DataFrame:
    available_pairs = [
        (w1, w2)
        for w1, w2 in pairs
        if w1 in embedding_cache and w2 in embedding_cache
    ]

    if not available_pairs:
        return pd.DataFrame(columns=["word_i", "word_j", "delta"])

    words_i = [p[0] for p in available_pairs]
    words_j = [p[1] for p in available_pairs]

    vec_i = np.stack([embedding_cache[w] for w in words_i]).astype(np.float32)
    vec_j = np.stack([embedding_cache[w] for w in words_j]).astype(np.float32)

    if metric == "cosine":
        vec_i = normalize_rows(vec_i)
        vec_j = normalize_rows(vec_j)
        deltas = 1.0 - np.sum(vec_i * vec_j, axis=1)

    elif metric == "euclidean_normalized":
        vec_i = normalize_rows(vec_i)
        vec_j = normalize_rows(vec_j)
        deltas = np.linalg.norm(vec_i - vec_j, axis=1)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return pd.DataFrame(
        {
            "word_i": words_i,
            "word_j": words_j,
            "delta": deltas.astype(np.float32),
        }
    )

def validate_tranches(embeddings_dir: Path, dim: int, runs: int) -> dict[int, list[str]]:
    tranches_by_run = {}

    for run in range(runs):
        tranche_sets = []

        for curriculum in CURRICULA:
            folder = embeddings_dir / f"{curriculum}_{dim}d_{run}"
            tranche_names = sorted(p.name for p in folder.glob("tranche_*.parquet"))

            if not tranche_names:
                raise FileNotFoundError(f"No tranche files found in {folder}")

            tranche_sets.append(set(tranche_names))

        shared_tranches = sorted(set.intersection(*tranche_sets))

        if not shared_tranches:
            raise ValueError(f"No shared tranche files found across curricula for run {run}.")

        tranches_by_run[run] = shared_tranches

        dropped_counts = {
            curriculum: len(tranche_sets[i] - set(shared_tranches))
            for i, curriculum in enumerate(CURRICULA)
        }

        if any(count > 0 for count in dropped_counts.values()):
            print(
                f"Run {run}: keeping {len(shared_tranches)} shared tranches; "
                f"dropped non-shared tranches: {dropped_counts}"
            )

    return tranches_by_run

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("deltas"))
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--pairs-per-run", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean_normalized"],
        default="cosine",
        help="cosine = 1 - cosine similarity; euclidean_normalized = Euclidean distance after L2-normalizing embeddings.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Validating tranche structure...")
    tranches_by_run = validate_tranches(args.embeddings_dir, args.dim, args.runs)

    print("Building shared vocabulary across all curricula, runs, and tranches...")
    shared_vocab = build_shared_vocab(args.embeddings_dir, args.dim)
    print(f"Shared vocabulary size: {len(shared_vocab):,}")

    for run in range(args.runs):
        print(f"\nProcessing run {run}...")

        run_dir = args.output_dir / f"run{run}"
        run_dir.mkdir(parents=True, exist_ok=True)

        pairs = sample_word_pairs(
            shared_vocab,
            n_pairs=args.pairs_per_run,
            seed=args.seed + run,
        )

        write_word_pairs(run_dir / "word_pairs.txt", pairs)

        for curriculum in CURRICULA:
            print(f"  Curriculum: {curriculum}")

            input_folder = args.embeddings_dir / f"{curriculum}_{args.dim}d_{run}"
            output_folder = run_dir / curriculum
            output_folder.mkdir(parents=True, exist_ok=True)

            embedding_cache: dict[str, np.ndarray] = {}

            for tranche_name in tranches_by_run[run]:
                input_file = input_folder / tranche_name
                output_file = output_folder / tranche_name

                update_embedding_cache(embedding_cache, input_file)

                delta_df = compute_deltas(
                    embedding_cache,
                    pairs=pairs,
                    metric=args.metric,
                )

                delta_df.to_parquet(output_file, index=False)

                print(
                    f"    {tranche_name}: "
                    f"{len(delta_df)}/{len(pairs)} pairs available, "
                    f"{len(embedding_cache):,} words seen so far"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()