#!/usr/bin/env python
"""
Late tranche neighbor overlap analysis.

This script analyzes embedding stability for LATE tranches (950+),
with configurable word sampling.

Default settings:
- Tranches 950 to end
- 50% of words (sampled)
- k=30 nearest neighbors
- 3 runs per curriculum
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


def compute_knn_overlap_three_runs_sampled(
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
    common_words = list(word_set1 & word_set2 & word_set3)
    
    if len(common_words) < k + 1:
        return np.nan
    
    # Sample words if requested
    if word_sample_frac < 1.0:
        rng = np.random.default_rng(seed)
        n_sample = max(k + 1, int(len(common_words) * word_sample_frac))
        sample_indices = rng.choice(len(common_words), size=min(n_sample, len(common_words)), replace=False)
        common_words = [common_words[i] for i in sample_indices]
    
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


def compute_curriculum_overlap_range(
    run1_dir: Path,
    run2_dir: Path,
    run3_dir: Path,
    k: int = 30,
    start_tranche: int = 950,
    end_tranche: int = 99999,
    word_sample_frac: float = 0.5,
    seed: int = 42
) -> tuple[list[int], list[float]]:
    """
    Compute per-tranche overlap across three runs for a range of tranches.
    
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
    
    # Filter tranches by range (tranche numbers are 1-indexed in filenames)
    range_tranches = [t for t in tranches1 
                      if start_tranche <= int(t.name.split("_")[1].split(".")[0]) <= end_tranche]
    
    sample_pct = int(word_sample_frac * 100)
    desc = f"Processing {run1_dir.name} (tranches {start_tranche}+, {sample_pct}% words)"
    
    for tranche_path1 in tqdm(range_tranches, desc=desc):
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
            
            overlap = compute_knn_overlap_three_runs_sampled(
                words1, emb1, words2, emb2, words3, emb3, 
                k=k, word_sample_frac=word_sample_frac, seed=seed + tranche_num
            )
            
            if not np.isnan(overlap):
                tranche_numbers.append(tranche_num)
                overlaps.append(overlap)
        except Exception as e:
            print(f"Warning: Failed to process {tranche_name}: {e}")
    
    return tranche_numbers, overlaps


def main():
    parser = argparse.ArgumentParser(
        description="Late tranches overlap analysis (950+, with word sampling)."
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
        "--start_tranche",
        type=int,
        default=950,
        help="Start tranche number (default: 950).",
    )
    parser.add_argument(
        "--end_tranche",
        type=int,
        default=99999,
        help="End tranche number (default: 99999 = to the end).",
    )
    parser.add_argument(
        "--word_sample_frac",
        type=float,
        default=0.5,
        help="Fraction of words to sample per tranche (default: 0.5 = 50%).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/figures/aoa_vs_shuffled_overlap_late.png",
        help="Output path for the plot.",
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sample_pct = int(args.word_sample_frac * 100)
    
    print(f"\n=== Late Tranche Overlap Analysis ===")
    print(f"  Tranches {args.start_tranche} to end")
    print(f"  {sample_pct}% of words (sampled)")
    print(f"  k={args.k} nearest neighbors")
    print(f"  3 runs per curriculum")
    
    # Compute overlaps for AoA curriculum (across 3 runs)
    print("\n=== AoA Curriculum (3 runs) ===")
    aoa_tranches, aoa_overlaps = compute_curriculum_overlap_range(
        Path(args.aoa_run1),
        Path(args.aoa_run2),
        Path(args.aoa_run3),
        k=args.k,
        start_tranche=args.start_tranche,
        end_tranche=args.end_tranche,
        word_sample_frac=args.word_sample_frac
    )
    
    # Compute overlaps for Shuffled curriculum (across 3 runs)
    print("\n=== Shuffled Curriculum (3 runs) ===")
    shuffled_tranches, shuffled_overlaps = compute_curriculum_overlap_range(
        Path(args.shuffled_run1),
        Path(args.shuffled_run2),
        Path(args.shuffled_run3),
        k=args.k,
        start_tranche=args.start_tranche,
        end_tranche=args.end_tranche,
        word_sample_frac=args.word_sample_frac
    )
    
    # Plot
    plt.figure(figsize=(14, 7))
    
    # AoA line
    plt.plot(aoa_tranches, aoa_overlaps, 
             linewidth=2, color="#2E86AB", alpha=0.8, label="AoA Curriculum", marker='o', markersize=3)
    
    # Shuffled line
    plt.plot(shuffled_tranches, shuffled_overlaps, 
             linewidth=2, color="#E94F37", alpha=0.8, label="Shuffled Curriculum", marker='s', markersize=3)
    
    plt.xlabel("Training Tranche", fontsize=12)
    plt.ylabel(f"k={args.k} Nearest Neighbor Overlap", fontsize=12)
    plt.title(f"Embedding Stability: AoA vs Shuffled Curriculum\n(Late Tranches {args.start_tranche}+, {sample_pct}% Words, Average Pairwise Overlap Across 3 Runs)", fontsize=13)
    
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add mean annotations
    if aoa_overlaps:
        aoa_mean = np.mean(aoa_overlaps)
        plt.axhline(y=aoa_mean, color="#2E86AB", linestyle="--", alpha=0.5)
        plt.text(min(aoa_tranches) + (max(aoa_tranches) - min(aoa_tranches)) * 0.02, aoa_mean + 0.02, 
                 f"AoA mean: {aoa_mean:.3f}", color="#2E86AB", fontsize=10)
    
    if shuffled_overlaps:
        shuffled_mean = np.mean(shuffled_overlaps)
        plt.axhline(y=shuffled_mean, color="#E94F37", linestyle="--", alpha=0.5)
        plt.text(min(shuffled_tranches) + (max(shuffled_tranches) - min(shuffled_tranches)) * 0.02, shuffled_mean - 0.04, 
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


