#!/usr/bin/env python
"""
Plot nearest neighbor overlap across multiple training runs.

This script measures the stability of word embeddings by computing the overlap
of k-nearest neighbors between different training runs. Higher overlap indicates
more stable/reproducible embeddings.

For each tranche:
1. Load embeddings from all runs (e.g., aoa_50d_0, aoa_50d_1, ..., aoa_50d_4)
2. Find common words across all runs
3. Compute k=30 nearest neighbors for each word in each run
4. Calculate pairwise overlap between runs
5. Average overlap across all word pairs

Output:
    A plot with tranche number on x-axis and average neighbor overlap on y-axis.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def load_tranche_embeddings(tranche_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Load embeddings from a tranche parquet file.
    
    Returns:
        words: List of words
        embeddings: numpy array of shape (n_words, embedding_dim)
    """
    table = pq.read_table(tranche_path)
    df = table.to_pandas()
    
    words = df["word"].tolist()
    embeddings = np.array(df["embedding"].tolist())
    
    return words, embeddings


def compute_knn(embeddings: np.ndarray, k: int) -> np.ndarray:
    """
    Compute k-nearest neighbors for each embedding.
    
    Returns:
        indices: Array of shape (n_words, k) containing neighbor indices
    """
    # k+1 because each word is its own nearest neighbor
    n_neighbors = min(k + 1, len(embeddings))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    nbrs.fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    
    # Exclude self (first column)
    return indices[:, 1:k+1] if indices.shape[1] > 1 else indices


def compute_overlap(neighbors1: set, neighbors2: set, k: int) -> float:
    """Compute overlap between two neighbor sets."""
    if k == 0:
        return 0.0
    return len(neighbors1 & neighbors2) / k


def compute_pairwise_overlap(
    run_embeddings: dict[int, tuple[list[str], np.ndarray]],
    k: int = 30
) -> float:
    """
    Compute average pairwise overlap of k-nearest neighbors across runs.
    
    Args:
        run_embeddings: Dict mapping run_id -> (words, embeddings)
        k: Number of nearest neighbors
    
    Returns:
        Average overlap across all pairs of runs and words
    """
    run_ids = list(run_embeddings.keys())
    if len(run_ids) < 2:
        return 0.0
    
    # Find common words across all runs
    word_sets = [set(run_embeddings[rid][0]) for rid in run_ids]
    common_words = set.intersection(*word_sets)
    
    if len(common_words) < k + 1:
        return np.nan  # Not enough common words
    
    # Build word -> index mapping for each run
    word_to_idx = {}
    for rid in run_ids:
        words = run_embeddings[rid][0]
        word_to_idx[rid] = {w: i for i, w in enumerate(words)}
    
    # Compute KNN for each run (only for common words)
    run_neighbors = {}  # run_id -> {word: set of neighbor words}
    
    for rid in run_ids:
        words, embeddings = run_embeddings[rid]
        
        # Get indices of common words
        common_indices = [word_to_idx[rid][w] for w in common_words]
        common_embeddings = embeddings[common_indices]
        common_words_list = [words[i] for i in common_indices]
        
        # Compute KNN on common words only
        knn_indices = compute_knn(common_embeddings, k)
        
        # Convert indices to word sets
        run_neighbors[rid] = {}
        for i, word in enumerate(common_words_list):
            neighbor_indices = knn_indices[i]
            neighbor_words = {common_words_list[j] for j in neighbor_indices if j < len(common_words_list)}
            run_neighbors[rid][word] = neighbor_words
    
    # Compute pairwise overlap
    overlaps = []
    for rid1, rid2 in combinations(run_ids, 2):
        for word in common_words:
            neighbors1 = run_neighbors[rid1].get(word, set())
            neighbors2 = run_neighbors[rid2].get(word, set())
            overlap = compute_overlap(neighbors1, neighbors2, k)
            overlaps.append(overlap)
    
    return np.mean(overlaps) if overlaps else np.nan


def discover_runs(base_dir: Path, pattern: str = "aoa_50d_") -> list[Path]:
    """Discover run directories matching the pattern."""
    runs = sorted(base_dir.glob(f"{pattern}*"))
    return [r for r in runs if r.is_dir()]


def discover_tranches(run_dir: Path) -> list[Path]:
    """Discover tranche parquet files in a run directory."""
    return sorted(run_dir.glob("tranche_*.parquet"))


def main():
    parser = argparse.ArgumentParser(
        description="Plot nearest neighbor overlap across training runs."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default="outputs/embeddings",
        help="Base directory containing run subdirectories.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="aoa_50d_",
        help="Pattern to match run directories (e.g., 'aoa_50d_' matches aoa_50d_0, aoa_50d_1, ...).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="Number of nearest neighbors to consider.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/figures/neighbor_overlap.png",
        help="Output path for the plot.",
    )
    parser.add_argument(
        "--sample_tranches",
        type=int,
        default=None,
        help="Sample every N tranches (for faster computation).",
    )
    
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Discover runs
    runs = discover_runs(embeddings_dir, args.pattern)
    print(f"Found {len(runs)} runs: {[r.name for r in runs]}")
    
    if len(runs) < 2:
        print("Error: Need at least 2 runs to compute overlap.")
        return
    
    # Get tranche list from first run
    tranches = discover_tranches(runs[0])
    print(f"Found {len(tranches)} tranches")
    
    if args.sample_tranches:
        tranches = tranches[::args.sample_tranches]
        print(f"Sampling every {args.sample_tranches} tranches: {len(tranches)} tranches")
    
    # Compute overlap for each tranche
    tranche_numbers = []
    overlaps = []
    
    for tranche_path in tqdm(tranches, desc="Computing overlaps"):
        tranche_name = tranche_path.stem  # e.g., "tranche_0001"
        tranche_num = int(tranche_name.split("_")[1])
        
        # Load embeddings from all runs for this tranche
        run_embeddings = {}
        valid_runs = 0
        
        for i, run_dir in enumerate(runs):
            run_tranche_path = run_dir / tranche_path.name
            if run_tranche_path.exists():
                try:
                    words, embeddings = load_tranche_embeddings(run_tranche_path)
                    if len(words) > args.k:
                        run_embeddings[i] = (words, embeddings)
                        valid_runs += 1
                except Exception as e:
                    print(f"Warning: Failed to load {run_tranche_path}: {e}")
        
        if valid_runs < 2:
            continue
        
        # Compute pairwise overlap
        overlap = compute_pairwise_overlap(run_embeddings, k=args.k)
        
        if not np.isnan(overlap):
            tranche_numbers.append(tranche_num)
            overlaps.append(overlap)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(tranche_numbers, overlaps, linewidth=1.5, color="#2E86AB", alpha=0.8)
    plt.scatter(tranche_numbers, overlaps, s=10, color="#2E86AB", alpha=0.6)
    
    plt.xlabel("Training Tranche", fontsize=12)
    plt.ylabel(f"Average k={args.k} Nearest Neighbor Overlap", fontsize=12)
    plt.title(f"Embedding Stability Across {len(runs)} Runs\n(Higher = More Consistent)", fontsize=14)
    
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    if len(tranche_numbers) > 1:
        z = np.polyfit(tranche_numbers, overlaps, 1)
        p = np.poly1d(z)
        plt.plot(tranche_numbers, p(tranche_numbers), "--", color="#E94F37", 
                 alpha=0.7, label=f"Trend (slope={z[0]:.4f})")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")
    print(f"Average overlap: {np.mean(overlaps):.4f}")
    print(f"Min overlap: {np.min(overlaps):.4f} (tranche {tranche_numbers[np.argmin(overlaps)]})")
    print(f"Max overlap: {np.max(overlaps):.4f} (tranche {tranche_numbers[np.argmax(overlaps)]})")


if __name__ == "__main__":
    main()

