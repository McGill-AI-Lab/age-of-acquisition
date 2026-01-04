#!/usr/bin/env python
"""
Fast version of neighbor overlap plotting with word and tranche sampling.

This script compares embedding stability between two curriculum types:
- AoA curriculum: Sentences ordered by age-of-acquisition (easy → hard)
- Shuffled curriculum: Random sentence ordering

This version supports:
- Word sampling: Only use a fraction of words in each tranche
- Tranche sampling: Only process every N tranches
- 3 runs per curriculum: Computes average pairwise overlap across all 3 pairs

Default settings for fast computation:
- 50% word sample
- Every 10 tranches
- k=30 nearest neighbors
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


def discover_tranches(run_dir: Path) -> list[Path]:
    """Discover tranche parquet files in a run directory."""
    return sorted(run_dir.glob("tranche_*.parquet"))


def compute_knn_overlap_sampled_three_runs(
    words1: list[str],
    embeddings1: np.ndarray,
    words2: list[str],
    embeddings2: np.ndarray,
    words3: list[str],
    embeddings3: np.ndarray,
    k: int = 30,
    word_sample_frac: float = 0.5,
    seed: int = 42
) -> float:
    """
    Compute k-nearest neighbor overlap across three embedding sets with word sampling.
    
    Returns the average pairwise overlap across all 3 pairs.
    Only considers words that appear in all three sets.
    """
    # Find common words across all three runs
    word_set1 = set(words1)
    word_set2 = set(words2)
    word_set3 = set(words3)
    common_words_full = list(word_set1 & word_set2 & word_set3)
    
    if len(common_words_full) < k + 1:
        return np.nan
    
    # Sample words if requested
    rng = np.random.default_rng(seed)
    n_sample = max(k + 1, int(len(common_words_full) * word_sample_frac))
    n_sample = min(n_sample, len(common_words_full))
    
    if n_sample < len(common_words_full):
        sample_indices = rng.choice(len(common_words_full), size=n_sample, replace=False)
        common_words = [common_words_full[i] for i in sample_indices]
    else:
        common_words = common_words_full
    
    if len(common_words) < k + 1:
        return np.nan
    
    # Build word -> index mappings
    idx1 = {w: i for i, w in enumerate(words1)}
    idx2 = {w: i for i, w in enumerate(words2)}
    idx3 = {w: i for i, w in enumerate(words3)}
    
    # Get embeddings for common words
    common_indices1 = [idx1[w] for w in common_words]
    common_indices2 = [idx2[w] for w in common_words]
    common_indices3 = [idx3[w] for w in common_words]
    
    emb1 = embeddings1[common_indices1]
    emb2 = embeddings2[common_indices2]
    emb3 = embeddings3[common_indices3]
    
    # Compute KNN for each set
    n_neighbors = min(k + 1, len(common_words))
    
    nbrs1 = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    nbrs1.fit(emb1)
    _, indices1 = nbrs1.kneighbors(emb1)
    
    nbrs2 = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    nbrs2.fit(emb2)
    _, indices2 = nbrs2.kneighbors(emb2)
    
    nbrs3 = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
    nbrs3.fit(emb3)
    _, indices3 = nbrs3.kneighbors(emb3)
    
    # Compute pairwise overlaps for each word, then average
    all_overlaps = []
    for i, word in enumerate(common_words):
        # Get neighbor words (excluding self at index 0)
        neighbors1 = {common_words[j] for j in indices1[i][1:k+1]}
        neighbors2 = {common_words[j] for j in indices2[i][1:k+1]}
        neighbors3 = {common_words[j] for j in indices3[i][1:k+1]}
        
        # Compute pairwise overlaps
        overlap_12 = len(neighbors1 & neighbors2) / k
        overlap_13 = len(neighbors1 & neighbors3) / k
        overlap_23 = len(neighbors2 & neighbors3) / k
        
        # Average across all 3 pairs
        avg_overlap = (overlap_12 + overlap_13 + overlap_23) / 3
        all_overlaps.append(avg_overlap)
    
    return np.mean(all_overlaps)


def compute_curriculum_overlap_three_runs(
    run1_dir: Path,
    run2_dir: Path,
    run3_dir: Path,
    k: int = 30,
    sample_every: int = 10,
    word_sample_frac: float = 0.5,
    seed: int = 42
) -> tuple[list[int], list[float]]:
    """
    Compute per-tranche overlap across three runs with sampling.
    
    Returns:
        tranche_numbers: List of tranche indices
        overlaps: List of overlap values (average pairwise across 3 runs)
    """
    tranches1 = discover_tranches(run1_dir)
    tranches2 = discover_tranches(run2_dir)
    tranches3 = discover_tranches(run3_dir)
    
    # Build mapping from tranche name to path
    tranche_map2 = {t.name: t for t in tranches2}
    tranche_map3 = {t.name: t for t in tranches3}
    
    tranche_numbers = []
    overlaps = []
    
    sampled_tranches = tranches1[::sample_every]
    desc = f"Processing {run1_dir.name} (every {sample_every}, {word_sample_frac*100:.0f}% words)"
    
    for tranche_path1 in tqdm(sampled_tranches, desc=desc):
        tranche_name = tranche_path1.name
        tranche_num = int(tranche_name.split("_")[1].split(".")[0])
        
        if tranche_name not in tranche_map2 or tranche_name not in tranche_map3:
            continue
        
        tranche_path2 = tranche_map2[tranche_name]
        tranche_path3 = tranche_map3[tranche_name]
        
        try:
            words1, emb1 = load_tranche_embeddings(tranche_path1)
            words2, emb2 = load_tranche_embeddings(tranche_path2)
            words3, emb3 = load_tranche_embeddings(tranche_path3)
            
            # Use tranche number as additional seed component for reproducibility
            tranche_seed = seed + tranche_num
            
            overlap = compute_knn_overlap_sampled_three_runs(
                words1, emb1, words2, emb2, words3, emb3,
                k=k, 
                word_sample_frac=word_sample_frac,
                seed=tranche_seed
            )
            
            if not np.isnan(overlap):
                tranche_numbers.append(tranche_num)
                overlaps.append(overlap)
        except Exception as e:
            print(f"Warning: Failed to process {tranche_name}: {e}")
    
    return tranche_numbers, overlaps


def main():
    parser = argparse.ArgumentParser(
        description="Fast overlap plot with word and tranche sampling (3 runs)."
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
        "--aoa_run3",
        type=str,
        default="outputs/embeddings/aoa_50d_2",
        help="Path to AoA curriculum run 3.",
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
        "--shuffled_run3",
        type=str,
        default="outputs/embeddings/shuffled_50d_2",
        help="Path to Shuffled curriculum run 3.",
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
        default="outputs/figures/aoa_vs_shuffled_overlap_fast.png",
        help="Output path for the plot.",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=10,
        help="Sample every N tranches (default: 10).",
    )
    parser.add_argument(
        "--word_sample_frac",
        type=float,
        default=0.5,
        help="Fraction of words to sample in each tranche (default: 0.5 = 50%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for word sampling.",
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Fast Overlap Computation (3 runs) ===")
    print(f"  Tranche sampling: every {args.sample_every} tranches")
    print(f"  Word sampling: {args.word_sample_frac*100:.0f}% of words per tranche")
    print(f"  k={args.k} nearest neighbors")
    
    # Compute overlaps for AoA curriculum (across 3 runs)
    print("\n=== AoA Curriculum (3 runs) ===")
    aoa_tranches, aoa_overlaps = compute_curriculum_overlap_three_runs(
        Path(args.aoa_run1),
        Path(args.aoa_run2),
        Path(args.aoa_run3),
        k=args.k,
        sample_every=args.sample_every,
        word_sample_frac=args.word_sample_frac,
        seed=args.seed
    )
    
    # Compute overlaps for Shuffled curriculum (across 3 runs)
    print("\n=== Shuffled Curriculum (3 runs) ===")
    shuffled_tranches, shuffled_overlaps = compute_curriculum_overlap_three_runs(
        Path(args.shuffled_run1),
        Path(args.shuffled_run2),
        Path(args.shuffled_run3),
        k=args.k,
        sample_every=args.sample_every,
        word_sample_frac=args.word_sample_frac,
        seed=args.seed
    )
    
    # Plot
    plt.figure(figsize=(14, 7))
    
    # AoA line
    plt.plot(aoa_tranches, aoa_overlaps, 
             linewidth=1.5, color="#2E86AB", alpha=0.8, label="AoA Curriculum", marker='o', markersize=4)
    
    # Shuffled line
    plt.plot(shuffled_tranches, shuffled_overlaps, 
             linewidth=1.5, color="#E94F37", alpha=0.8, label="Shuffled Curriculum", marker='s', markersize=4)
    
    plt.xlabel("Training Tranche", fontsize=12)
    plt.ylabel(f"k={args.k} Nearest Neighbor Overlap", fontsize=12)
    plt.title(f"Embedding Stability: AoA vs Shuffled Curriculum\n(Average Pairwise Overlap Across 3 Runs, Every {args.sample_every} tranches, {args.word_sample_frac*100:.0f}% word sample)", fontsize=14)
    
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
    print(f"Tranches processed: AoA={len(aoa_tranches)}, Shuffled={len(shuffled_tranches)}")
    if aoa_overlaps:
        print(f"AoA: mean={np.mean(aoa_overlaps):.4f}, min={np.min(aoa_overlaps):.4f}, max={np.max(aoa_overlaps):.4f}")
    if shuffled_overlaps:
        print(f"Shuffled: mean={np.mean(shuffled_overlaps):.4f}, min={np.min(shuffled_overlaps):.4f}, max={np.max(shuffled_overlaps):.4f}")


if __name__ == "__main__":
    main()
