#!/usr/bin/env python
"""
Plot nearest neighbor overlap comparing AoA vs Shuffled curricula.

This script compares embedding stability between two curriculum types:
- AoA curriculum: Sentences ordered by age-of-acquisition (easy → hard)
- Shuffled curriculum: Random sentence ordering

For each curriculum, it computes the k=30 nearest neighbor overlap across
two training runs (with different random seeds for within-tranche shuffling).
Overlap is computed as the pairwise overlap between the two runs.

Output:
    A plot with two lines:
    - Blue: AoA curriculum overlap (between 2 runs)
    - Red: Shuffled curriculum overlap (between 2 runs)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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


def compute_knn_overlap(
    words1: list[str],
    embeddings1: np.ndarray,
    words2: list[str],
    embeddings2: np.ndarray,
    k: int = 30
) -> float:
    """
    Compute k-nearest neighbor overlap between two embedding sets.
    
    Only considers words that appear in both sets.
    """
    # Find common words
    word_set1 = set(words1)
    word_set2 = set(words2)
    common_words = list(word_set1 & word_set2)
    
    if len(common_words) < k + 1:
        return np.nan
    
    # Build word -> index mappings
    idx1 = {w: i for i, w in enumerate(words1)}
    idx2 = {w: i for i, w in enumerate(words2)}
    
    # Get embeddings for common words
    common_indices1 = [idx1[w] for w in common_words]
    common_indices2 = [idx2[w] for w in common_words]
    
    emb1 = embeddings1[common_indices1]
    emb2 = embeddings2[common_indices2]
    
    # Compute KNN for each set
    n_neighbors = min(k + 1, len(common_words))
    
    nbrs1 = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    nbrs1.fit(emb1)
    _, indices1 = nbrs1.kneighbors(emb1)
    
    nbrs2 = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    nbrs2.fit(emb2)
    _, indices2 = nbrs2.kneighbors(emb2)
    
    # Compute overlap for each word
    overlaps = []
    for i, word in enumerate(common_words):
        # Get neighbor words (excluding self at index 0)
        neighbors1 = {common_words[j] for j in indices1[i][1:k+1]}
        neighbors2 = {common_words[j] for j in indices2[i][1:k+1]}
        
        overlap = len(neighbors1 & neighbors2) / k
        overlaps.append(overlap)
    
    return np.mean(overlaps)


def discover_tranches(run_dir: Path) -> list[Path]:
    """Discover tranche parquet files in a run directory."""
    return sorted(run_dir.glob("tranche_*.parquet"))


def compute_knn_overlap_two_runs(
    words1: list[str],
    embeddings1: np.ndarray,
    words2: list[str],
    embeddings2: np.ndarray,
    k: int = 30
) -> float:
    """
    Compute k-nearest neighbor overlap between two embedding sets.
    
    Only considers words that appear in both sets.
    """
    # Find common words across both runs
    word_set1 = set(words1)
    word_set2 = set(words2)
    common_words = list(word_set1 & word_set2)
    
    if len(common_words) < k + 1:
        return np.nan
    
    # Build word -> index mappings
    idx1 = {w: i for i, w in enumerate(words1)}
    idx2 = {w: i for i, w in enumerate(words2)}
    
    # Get embeddings for common words
    common_indices1 = [idx1[w] for w in common_words]
    common_indices2 = [idx2[w] for w in common_words]
    
    emb1 = embeddings1[common_indices1]
    emb2 = embeddings2[common_indices2]
    
    # Compute KNN for each set
    n_neighbors = min(k + 1, len(common_words))
    
    nbrs1 = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    nbrs1.fit(emb1)
    _, indices1 = nbrs1.kneighbors(emb1)
    
    nbrs2 = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    nbrs2.fit(emb2)
    _, indices2 = nbrs2.kneighbors(emb2)
    
    # Compute overlap for each word
    overlaps = []
    for i, word in enumerate(common_words):
        # Get neighbor words (excluding self at index 0)
        neighbors1 = {common_words[j] for j in indices1[i][1:k+1]}
        neighbors2 = {common_words[j] for j in indices2[i][1:k+1]}
        
        overlap = len(neighbors1 & neighbors2) / k
        overlaps.append(overlap)
    
    return np.mean(overlaps)


def compute_curriculum_overlap(
    run1_dir: Path,
    run2_dir: Path,
    k: int = 30,
    sample_every: int = 1
) -> tuple[list[int], list[float]]:
    """
    Compute per-tranche overlap across two runs.
    
    Returns:
        tranche_numbers: List of tranche indices
        overlaps: List of overlap values (pairwise between 2 runs)
    """
    tranches1 = discover_tranches(run1_dir)
    tranches2 = discover_tranches(run2_dir)
    
    # Build mapping from tranche name to path
    tranche_map2 = {t.name: t for t in tranches2}
    
    tranche_numbers = []
    overlaps = []
    
    for tranche_path1 in tqdm(tranches1[::sample_every], desc=f"Processing {run1_dir.name}"):
        tranche_name = tranche_path1.name
        tranche_num = int(tranche_name.split("_")[1].split(".")[0])
        
        if tranche_name not in tranche_map2:
            continue
        
        tranche_path2 = tranche_map2[tranche_name]
        
        try:
            words1, emb1 = load_tranche_embeddings(tranche_path1)
            words2, emb2 = load_tranche_embeddings(tranche_path2)
            
            overlap = compute_knn_overlap_two_runs(
                words1, emb1, words2, emb2, k=k
            )
            
            if not np.isnan(overlap):
                tranche_numbers.append(tranche_num)
                overlaps.append(overlap)
        except Exception as e:
            print(f"Warning: Failed to process {tranche_name}: {e}")
    
    return tranche_numbers, overlaps


def main():
    parser = argparse.ArgumentParser(
        description="Plot nearest neighbor overlap: AoA vs Shuffled curriculum."
    )
    parser.add_argument(
        "--aoa_run1",
        type=str,
        default="outputs/embeddings/aoa_50d_0",
        help="Path to AoA curriculum run 1.",
    )
    parser.add_argument(
        "--aoa_run2",
        type=str,
        default="outputs/embeddings/aoa_50d_1",
        help="Path to AoA curriculum run 2.",
    )
    parser.add_argument(
        "--shuffled_run1",
        type=str,
        default="outputs/embeddings/shuffled_50d_0",
        help="Path to Shuffled curriculum run 1.",
    )
    parser.add_argument(
        "--shuffled_run2",
        type=str,
        default="outputs/embeddings/shuffled_50d_1",
        help="Path to Shuffled curriculum run 2.",
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
        default="outputs/figures/aoa_vs_shuffled_overlap.png",
        help="Output path for the plot.",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=1,
        help="Sample every N tranches (for faster computation).",
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compute overlaps for AoA curriculum (across 2 runs)
    print("\n=== AoA Curriculum (2 runs) ===")
    aoa_tranches, aoa_overlaps = compute_curriculum_overlap(
        Path(args.aoa_run1),
        Path(args.aoa_run2),
        k=args.k,
        sample_every=args.sample_every
    )
    
    # Compute overlaps for Shuffled curriculum (across 2 runs)
    print("\n=== Shuffled Curriculum (2 runs) ===")
    shuffled_tranches, shuffled_overlaps = compute_curriculum_overlap(
        Path(args.shuffled_run1),
        Path(args.shuffled_run2),
        k=args.k,
        sample_every=args.sample_every
    )
    
    # Plot
    plt.figure(figsize=(14, 7))
    
    # AoA line
    plt.plot(aoa_tranches, aoa_overlaps, 
             linewidth=1.5, color="#2E86AB", alpha=0.8, label="AoA Curriculum")
    
    # Shuffled line
    plt.plot(shuffled_tranches, shuffled_overlaps, 
             linewidth=1.5, color="#E94F37", alpha=0.8, label="Shuffled Curriculum")
    
    plt.xlabel("Training Tranche", fontsize=12)
    plt.ylabel(f"k={args.k} Nearest Neighbor Overlap", fontsize=12)
    plt.title("Embedding Stability: AoA vs Shuffled Curriculum\n(Pairwise Overlap Between 2 Runs)", fontsize=14)
    
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add mean annotations
    if aoa_overlaps:
        aoa_mean = np.mean(aoa_overlaps)
        plt.axhline(y=aoa_mean, color="#2E86AB", linestyle="--", alpha=0.5)
        plt.text(max(aoa_tranches) * 0.02, aoa_mean + 0.02, 
                 f"AoA mean: {aoa_mean:.3f}", color="#2E86AB", fontsize=10)
    
    if shuffled_overlaps:
        shuffled_mean = np.mean(shuffled_overlaps)
        plt.axhline(y=shuffled_mean, color="#E94F37", linestyle="--", alpha=0.5)
        plt.text(max(shuffled_tranches) * 0.02, shuffled_mean - 0.04, 
                 f"Shuffled mean: {shuffled_mean:.3f}", color="#E94F37", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n=== Results ===")
    print(f"Plot saved to: {output_path}")
    if aoa_overlaps:
        print(f"AoA: mean={np.mean(aoa_overlaps):.4f}, min={np.min(aoa_overlaps):.4f}, max={np.max(aoa_overlaps):.4f}")
    if shuffled_overlaps:
        print(f"Shuffled: mean={np.mean(shuffled_overlaps):.4f}, min={np.min(shuffled_overlaps):.4f}, max={np.max(shuffled_overlaps):.4f}")


if __name__ == "__main__":
    main()
