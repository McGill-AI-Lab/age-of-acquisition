#!/usr/bin/env python
"""
Unified nearest neighbor overlap analysis for curriculum comparison.

This script compares embedding stability between curriculum types (e.g., AoA vs Shuffled)
by computing k-nearest neighbor overlap across multiple training runs (2-5 runs supported).

Supports multiple analysis modes via the --mode parameter or custom configuration:
  - full:  All tranches, 100% words, 2 runs (default)
  - fast:  Every 10 tranches, 50% words, 3 runs
  - early: Tranches 30-200, 100% words, 3 runs
  - late:  Tranches 950+, 50% words, 3 runs

Example usage:
  # Full analysis with 3 runs
  python plot_neighbor_overlap.py --mode full --num_runs 3

  # Fast analysis (every 10 tranches, 50% word sample)
  python plot_neighbor_overlap.py --mode fast

  # Early tranches only
  python plot_neighbor_overlap.py --mode early

  # Analysis with 5 runs
  python plot_neighbor_overlap.py --num_runs 5

  # Custom configuration
  python plot_neighbor_overlap.py --start_tranche 100 --end_tranche 500 --word_sample_frac 0.75 --sample_every 5
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Optional

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


def compute_knn_overlap_n_runs(
    embeddings_list: list[tuple[list[str], np.ndarray]],
    k: int = 30,
    word_sample_frac: float = 1.0,
    seed: int = 42
) -> float:
    """
    Compute k-nearest neighbor overlap across N embedding sets (2-5 runs).
    
    Returns the average pairwise overlap across all C(n,2) pairs.
    Only considers words that appear in all sets.
    Optionally samples a fraction of common words.
    
    Args:
        embeddings_list: List of (words, embeddings) tuples, one per run
        k: Number of nearest neighbors to consider
        word_sample_frac: Fraction of common words to sample (1.0 = all)
        seed: Random seed for word sampling
    
    Returns:
        Average pairwise overlap across all pairs of runs
    """
    num_runs = len(embeddings_list)
    
    # Find common words across all runs
    word_sets = [set(words) for words, _ in embeddings_list]
    common_words = list(set.intersection(*word_sets))
    
    if len(common_words) < k + 1:
        return np.nan
    
    # Sample words if requested
    if word_sample_frac < 1.0:
        rng = np.random.default_rng(seed)
        n_sample = max(k + 1, int(len(common_words) * word_sample_frac))
        n_sample = min(n_sample, len(common_words))
        sample_indices = rng.choice(len(common_words), size=n_sample, replace=False)
        common_words = [common_words[i] for i in sample_indices]
    
    if len(common_words) < k + 1:
        return np.nan
    
    # Build word -> index mappings for each run
    idx_maps = [{w: i for i, w in enumerate(words)} for words, _ in embeddings_list]
    
    # Get embeddings for common words from each run
    common_embeddings = []
    for (words, embeddings), idx_map in zip(embeddings_list, idx_maps):
        common_indices = [idx_map[w] for w in common_words]
        common_embeddings.append(embeddings[common_indices])
    
    # Compute KNN for each run
    n_neighbors = min(k + 1, len(common_words))
    all_indices = []
    
    for emb in common_embeddings:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine")
        nbrs.fit(emb)
        _, indices = nbrs.kneighbors(emb)
        all_indices.append(indices)
    
    # Compute pairwise overlaps for each word, then average across all pairs
    # Number of pairs = C(num_runs, 2) = num_runs * (num_runs - 1) / 2
    num_pairs = num_runs * (num_runs - 1) // 2
    
    all_overlaps = []
    for word_idx in range(len(common_words)):
        # Get neighbor words for this word from each run
        neighbor_sets = []
        for run_indices in all_indices:
            neighbors = {common_words[j] for j in run_indices[word_idx][1:k+1]}
            neighbor_sets.append(neighbors)
        
        # Compute pairwise overlaps
        pairwise_overlaps = []
        for i, j in combinations(range(num_runs), 2):
            overlap = len(neighbor_sets[i] & neighbor_sets[j]) / k
            pairwise_overlaps.append(overlap)
        
        # Average across all pairs for this word
        avg_overlap = sum(pairwise_overlaps) / num_pairs
        all_overlaps.append(avg_overlap)
    
    return np.mean(all_overlaps)


def compute_curriculum_overlap(
    run_dirs: list[Path],
    k: int = 30,
    start_tranche: int = 1,
    end_tranche: int = 99999,
    sample_every: int = 1,
    word_sample_frac: float = 1.0,
    seed: int = 42
) -> tuple[list[int], list[float]]:
    """
    Compute per-tranche overlap across multiple runs.
    
    Args:
        run_dirs: List of 2-5 run directories
        k: Number of nearest neighbors
        start_tranche: First tranche to include (inclusive)
        end_tranche: Last tranche to include (inclusive)
        sample_every: Process every N-th tranche within the range
        word_sample_frac: Fraction of words to sample (1.0 = all)
        seed: Random seed for word sampling
    
    Returns:
        tranche_numbers: List of tranche indices
        overlaps: List of overlap values
    """
    num_runs = len(run_dirs)
    assert 2 <= num_runs <= 5, "Must provide 2-5 run directories"
    
    # Discover tranches for each run
    all_tranches = [discover_tranches(run_dir) for run_dir in run_dirs]
    tranches1 = all_tranches[0]
    
    # Build mapping from tranche name to path for runs 2+
    tranche_maps = [{t.name: t for t in tranches} for tranches in all_tranches[1:]]
    
    # Filter tranches by range
    range_tranches = [
        t for t in tranches1 
        if start_tranche <= int(t.name.split("_")[1].split(".")[0]) <= end_tranche
    ]
    
    # Apply tranche sampling
    sampled_tranches = range_tranches[::sample_every]
    
    tranche_numbers = []
    overlaps = []
    
    # Build description string
    sample_pct = int(word_sample_frac * 100)
    range_str = f"{start_tranche}-{end_tranche}" if end_tranche < 99999 else f"{start_tranche}+"
    desc = f"Processing {run_dirs[0].name} ({range_str}, every {sample_every}, {sample_pct}% words)"
    
    for tranche_path1 in tqdm(sampled_tranches, desc=desc):
        tranche_name = tranche_path1.name
        tranche_num = int(tranche_name.split("_")[1].split(".")[0])
        
        # Check if tranche exists in all runs
        if not all(tranche_name in tm for tm in tranche_maps):
            continue
        
        # Get paths for all runs
        tranche_paths = [tranche_path1] + [tm[tranche_name] for tm in tranche_maps]
        
        try:
            # Load embeddings for all runs
            embeddings_data = [load_tranche_embeddings(p) for p in tranche_paths]
            
            # Use tranche number as additional seed component for reproducibility
            tranche_seed = seed + tranche_num
            
            overlap = compute_knn_overlap_n_runs(
                embeddings_data,
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


def get_mode_defaults(mode: str) -> dict:
    """Get default parameters for each analysis mode."""
    modes = {
        "full": {
            "start_tranche": 1,
            "end_tranche": 99999,
            "sample_every": 1,
            "word_sample_frac": 1.0,
            "num_runs": 2,
            "output_suffix": "",
        },
        "fast": {
            "start_tranche": 1,
            "end_tranche": 99999,
            "sample_every": 10,
            "word_sample_frac": 0.5,
            "num_runs": 3,
            "output_suffix": "_fast",
        },
        "early": {
            "start_tranche": 30,
            "end_tranche": 200,
            "sample_every": 1,
            "word_sample_frac": 1.0,
            "num_runs": 3,
            "output_suffix": "_early",
        },
        "late": {
            "start_tranche": 950,
            "end_tranche": 99999,
            "sample_every": 1,
            "word_sample_frac": 0.5,
            "num_runs": 3,
            "output_suffix": "_late",
        },
    }
    return modes.get(mode, modes["full"])


def main():
    parser = argparse.ArgumentParser(
        description="Unified neighbor overlap analysis for curriculum comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full   All tranches, 100%% words, 2 runs (default)
  fast   Every 10 tranches, 50%% words, 3 runs
  early  Tranches 30-200, 100%% words, 3 runs
  late   Tranches 950+, 50%% words, 3 runs

Examples:
  python plot_neighbor_overlap.py --mode fast
  python plot_neighbor_overlap.py --mode early --num_runs 2
  python plot_neighbor_overlap.py --num_runs 5
  python plot_neighbor_overlap.py --start_tranche 100 --end_tranche 500
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "fast", "early", "late"],
        default="full",
        help="Analysis mode preset (default: full). Custom args override mode defaults.",
    )
    
    # Run paths - AoA curriculum (up to 5 runs)
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
        help="Path to AoA curriculum run 3 (only used if --num_runs >= 3).",
    )
    parser.add_argument(
        "--aoa_run4",
        type=str,
        default="outputs/embeddings/aoa_50d_3",
        help="Path to AoA curriculum run 4 (only used if --num_runs >= 4).",
    )
    parser.add_argument(
        "--aoa_run5",
        type=str,
        default="outputs/embeddings/aoa_50d_4",
        help="Path to AoA curriculum run 5 (only used if --num_runs >= 5).",
    )
    
    # Run paths - Shuffled curriculum (up to 5 runs)
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
        help="Path to Shuffled curriculum run 3 (only used if --num_runs >= 3).",
    )
    parser.add_argument(
        "--shuffled_run4",
        type=str,
        default="outputs/embeddings/shuffled_50d_3",
        help="Path to Shuffled curriculum run 4 (only used if --num_runs >= 4).",
    )
    parser.add_argument(
        "--shuffled_run5",
        type=str,
        default="outputs/embeddings/shuffled_50d_4",
        help="Path to Shuffled curriculum run 5 (only used if --num_runs >= 5).",
    )
    
    # Analysis parameters
    parser.add_argument(
        "--num_runs",
        type=int,
        default=None,
        choices=[2, 3, 4, 5],
        help="Number of runs to compare (2-5). Defaults to mode setting.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="Number of nearest neighbors to consider (default: 30).",
    )
    parser.add_argument(
        "--start_tranche",
        type=int,
        default=None,
        help="First tranche to include (default: mode setting).",
    )
    parser.add_argument(
        "--end_tranche",
        type=int,
        default=None,
        help="Last tranche to include (default: mode setting).",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=None,
        help="Process every N-th tranche (default: mode setting).",
    )
    parser.add_argument(
        "--word_sample_frac",
        type=float,
        default=None,
        help="Fraction of words to sample per tranche (default: mode setting).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for word sampling (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot. Defaults to outputs/figures/aoa_vs_shuffled_overlap{suffix}.png",
    )
    
    args = parser.parse_args()
    
    # Get mode defaults and apply custom overrides
    mode_defaults = get_mode_defaults(args.mode)
    
    num_runs = args.num_runs if args.num_runs is not None else mode_defaults["num_runs"]
    start_tranche = args.start_tranche if args.start_tranche is not None else mode_defaults["start_tranche"]
    end_tranche = args.end_tranche if args.end_tranche is not None else mode_defaults["end_tranche"]
    sample_every = args.sample_every if args.sample_every is not None else mode_defaults["sample_every"]
    word_sample_frac = args.word_sample_frac if args.word_sample_frac is not None else mode_defaults["word_sample_frac"]
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"outputs/figures/aoa_vs_shuffled_overlap{mode_defaults['output_suffix']}.png")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect run directories based on num_runs
    aoa_run_paths = [
        args.aoa_run1, args.aoa_run2, args.aoa_run3, args.aoa_run4, args.aoa_run5
    ]
    shuffled_run_paths = [
        args.shuffled_run1, args.shuffled_run2, args.shuffled_run3, 
        args.shuffled_run4, args.shuffled_run5
    ]
    
    aoa_runs = [Path(p) for p in aoa_run_paths[:num_runs]]
    shuffled_runs = [Path(p) for p in shuffled_run_paths[:num_runs]]
    
    # Calculate number of pairs for display
    num_pairs = num_runs * (num_runs - 1) // 2
    
    # Print configuration
    sample_pct = int(word_sample_frac * 100)
    range_str = f"{start_tranche}-{end_tranche}" if end_tranche < 99999 else f"{start_tranche}+"
    
    print(f"\n{'='*60}")
    print(f"Neighbor Overlap Analysis")
    print(f"{'='*60}")
    print(f"  Mode: {args.mode}")
    print(f"  Runs: {num_runs} ({num_pairs} pairwise comparisons)")
    print(f"  Tranche range: {range_str}")
    print(f"  Tranche sampling: every {sample_every}")
    print(f"  Word sampling: {sample_pct}%")
    print(f"  k: {args.k} nearest neighbors")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    # Compute overlaps for AoA curriculum
    print(f"\n=== AoA Curriculum ({num_runs} runs) ===")
    aoa_tranches, aoa_overlaps = compute_curriculum_overlap(
        aoa_runs,
        k=args.k,
        start_tranche=start_tranche,
        end_tranche=end_tranche,
        sample_every=sample_every,
        word_sample_frac=word_sample_frac,
        seed=args.seed
    )
    
    # Compute overlaps for Shuffled curriculum
    print(f"\n=== Shuffled Curriculum ({num_runs} runs) ===")
    shuffled_tranches, shuffled_overlaps = compute_curriculum_overlap(
        shuffled_runs,
        k=args.k,
        start_tranche=start_tranche,
        end_tranche=end_tranche,
        sample_every=sample_every,
        word_sample_frac=word_sample_frac,
        seed=args.seed
    )
    
    # Plot
    plt.figure(figsize=(14, 7))
    
    # Determine marker size based on number of points
    n_points = max(len(aoa_tranches), len(shuffled_tranches))
    marker_size = 4 if n_points < 200 else 2 if n_points < 500 else 1
    use_markers = n_points < 300
    
    # AoA line
    plt.plot(
        aoa_tranches, aoa_overlaps,
        linewidth=1.5, color="#2E86AB", alpha=0.8, label="AoA Curriculum",
        marker='o' if use_markers else None, markersize=marker_size
    )
    
    # Shuffled line
    plt.plot(
        shuffled_tranches, shuffled_overlaps,
        linewidth=1.5, color="#E94F37", alpha=0.8, label="Shuffled Curriculum",
        marker='s' if use_markers else None, markersize=marker_size
    )
    
    plt.xlabel("Training Tranche", fontsize=12)
    plt.ylabel(f"k={args.k} Nearest Neighbor Overlap", fontsize=12)
    
    # Build title
    run_str = f"{num_runs} Runs"
    if sample_every > 1:
        title_range = f"Every {sample_every} Tranches"
    else:
        title_range = f"Tranches {range_str}"
    title_words = f"{sample_pct}% Words" if sample_pct < 100 else "All Words"
    
    plt.title(
        f"Embedding Stability: AoA vs Shuffled Curriculum\n({title_range}, {title_words}, {run_str})",
        fontsize=13
    )
    
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add mean annotations
    if aoa_overlaps:
        aoa_mean = np.mean(aoa_overlaps)
        plt.axhline(y=aoa_mean, color="#2E86AB", linestyle="--", alpha=0.5)
        x_pos = min(aoa_tranches) + (max(aoa_tranches) - min(aoa_tranches)) * 0.02
        plt.text(x_pos, aoa_mean + 0.02, f"AoA mean: {aoa_mean:.3f}", color="#2E86AB", fontsize=10)
    
    if shuffled_overlaps:
        shuffled_mean = np.mean(shuffled_overlaps)
        plt.axhline(y=shuffled_mean, color="#E94F37", linestyle="--", alpha=0.5)
        x_pos = min(shuffled_tranches) + (max(shuffled_tranches) - min(shuffled_tranches)) * 0.02
        plt.text(x_pos, shuffled_mean - 0.04, f"Shuffled mean: {shuffled_mean:.3f}", color="#E94F37", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Plot saved to: {output_path}")
    print(f"Tranches processed: AoA={len(aoa_tranches)}, Shuffled={len(shuffled_tranches)}")
    if aoa_overlaps:
        print(f"AoA: mean={np.mean(aoa_overlaps):.4f}, min={np.min(aoa_overlaps):.4f}, max={np.max(aoa_overlaps):.4f}")
    if shuffled_overlaps:
        print(f"Shuffled: mean={np.mean(shuffled_overlaps):.4f}, min={np.min(shuffled_overlaps):.4f}, max={np.max(shuffled_overlaps):.4f}")


if __name__ == "__main__":
    main()
