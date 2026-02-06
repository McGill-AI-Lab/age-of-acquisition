#!/usr/bin/env python
"""
Nearest neighbor overlap analysis with unique word counts on x-axis.

This script compares embedding stability between curriculum types (e.g., AoA vs Shuffled)
by computing k-nearest neighbor overlap across multiple training runs (2-5 runs supported).
Unlike plot_neighbor_overlap.py, this version plots overlap against cumulative unique word
counts rather than tranche numbers.

The script requires curriculum paths/indices to load unique_word_counts.json files that
map tranche numbers to cumulative unique word counts.

Example usage:
  # Full analysis with 3 runs
  python plot_neighbor_overlap_by_words.py --curriculum1_idx 0 --curriculum2_idx 1 --num_runs 3

  # Fast analysis (every 10 tranches, 50% word sample)
  python plot_neighbor_overlap_by_words.py --curriculum1_idx 0 --curriculum2_idx 1 --mode fast

  # Custom curriculum paths
  python plot_neighbor_overlap_by_words.py \
      --curriculum1_path data/processed/corpora/training/000_c=aoa_... \
      --curriculum2_path data/processed/corpora/training/001_c=shuffled_...
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from curricula.path_utils import resolve_curriculum_path
from curricula.word_counts import write_unique_word_counts


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


def load_unique_word_counts(curriculum_path: str | Path | int, auto_generate: bool = True) -> dict[int, int]:
    """
    Load unique_word_counts.json and create a mapping from tranche_index to cumulative_unique_words.
    
    If the file doesn't exist and auto_generate is True, automatically generates it using
    write_unique_word_counts().
    
    Args:
        curriculum_path: Curriculum path, index (int or str), or Path
        auto_generate: If True, automatically generate the file if it doesn't exist
        
    Returns:
        Dictionary mapping tranche_index -> cumulative_unique_words
    """
    curriculum_root = resolve_curriculum_path(curriculum_path)
    counts_file = curriculum_root / "unique_word_counts.json"
    
    if not counts_file.exists():
        if auto_generate:
            print(f"  Generating unique_word_counts.json for curriculum at {curriculum_root}...")
            try:
                write_unique_word_counts(curriculum_path)
                print(f"  Successfully generated {counts_file}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate unique_word_counts.json for curriculum at {curriculum_root}: {e}"
                ) from e
        else:
            raise FileNotFoundError(
                f"unique_word_counts.json not found at {counts_file}. "
                f"Run write_unique_word_counts() first for curriculum at {curriculum_root}"
            )
    
    with open(counts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mapping: tranche_index -> cumulative_unique_words
    mapping = {}
    for entry in data.get("per_tranche", []):
        tranche_idx = entry.get("tranche_index")
        cumulative = entry.get("cumulative_unique_words")
        if tranche_idx is not None and cumulative is not None:
            mapping[tranche_idx] = cumulative
    
    return mapping


def load_word_counts_from_metadata(
    embeddings_dir: Path,
    word_count_type: str = "seen"
) -> dict[int, int]:
    """
    Load word counts from training metadata.json file.
    
    Args:
        embeddings_dir: Path to embeddings directory containing metadata.json
        word_count_type: Type of word count to use:
            - "seen": cumulative unique words encountered during training
            - "trained": number of words in the model vocabulary (above min_count)
        
    Returns:
        Dictionary mapping tranche_index -> word_count
    """
    metadata_file = embeddings_dir / "metadata.json"
    
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_file}. "
            f"Training must be run with word count tracking enabled."
        )
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine which field to use based on word_count_type
    if word_count_type == "seen":
        field_name = "unique_words_seen"
    elif word_count_type == "trained":
        field_name = "unique_words_trained"
    else:
        raise ValueError(f"Unknown word_count_type: {word_count_type}. Must be 'seen' or 'trained'.")
    
    # Create mapping: tranche_index -> word_count
    mapping = {}
    for entry in data.get("tranches", []):
        tranche_idx = entry.get("tranche_index")
        word_count = entry.get(field_name)
        if tranche_idx is not None and word_count is not None:
            mapping[tranche_idx] = word_count
    
    return mapping


def compute_knn_overlap_n_runs(
    embeddings_list: list[tuple[list[str], np.ndarray]],
    k: int = 30,
    word_sample_frac: float = 1.0,
    seed: int = 42
) -> tuple[float, float, float]:
    """
    Compute k-nearest neighbor overlap across N embedding sets (2-5 runs).
    
    Returns the average pairwise overlap across all C(n,2) run pairs and the
    standard error/standard deviation computed over those pairwise averages
    (run-to-run variability). Only considers words that appear in all sets.
    Optionally samples a fraction of common words.
    
    Args:
        embeddings_list: List of (words, embeddings) tuples, one per run
        k: Number of nearest neighbors to consider
        word_sample_frac: Fraction of common words to sample (1.0 = all)
        seed: Random seed for word sampling
    
    Returns:
        (mean_overlap, std_error, std_dev): Average pairwise overlap, its
        standard error (SEM over run-pair means), and the standard deviation of
        run-pair means
    """
    num_runs = len(embeddings_list)
    
    # Find common words across all runs
    word_sets = [set(words) for words, _ in embeddings_list]
    common_words = list(set.intersection(*word_sets))
    
    if len(common_words) < k + 1:
        return np.nan, np.nan, np.nan
    
    # Sample words if requested
    if word_sample_frac < 1.0:
        rng = np.random.default_rng(seed)
        n_sample = max(k + 1, int(len(common_words) * word_sample_frac))
        n_sample = min(n_sample, len(common_words))
        sample_indices = rng.choice(len(common_words), size=n_sample, replace=False)
        common_words = [common_words[i] for i in sample_indices]
    
    if len(common_words) < k + 1:
        return np.nan, np.nan, np.nan
    
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
    
    # Compute overlaps per run pair across words, then aggregate across pairs.
    pair_overlaps: dict[tuple[int, int], list[float]] = {
        (i, j): [] for i, j in combinations(range(num_runs), 2)
    }
    
    for word_idx in range(len(common_words)):
        # Get neighbor words for this word from each run
        neighbor_sets = []
        for run_indices in all_indices:
            neighbors = {common_words[j] for j in run_indices[word_idx][1:k+1]}
            neighbor_sets.append(neighbors)
        
        # Collect overlap for each pair for this word
        for i, j in combinations(range(num_runs), 2):
            overlap = len(neighbor_sets[i] & neighbor_sets[j]) / k
            pair_overlaps[(i, j)].append(overlap)
    
    # Average overlap per run pair across words
    pairwise_means = []
    for overlaps in pair_overlaps.values():
        if overlaps:
            pairwise_means.append(np.mean(overlaps))
    
    if not pairwise_means:
        return np.nan, np.nan, np.nan
    
    mean_overlap = float(np.mean(pairwise_means))
    # Standard deviation over run-pair means (population) and corresponding SEM
    std_dev = float(np.std(pairwise_means, ddof=0))
    std_error = float(std_dev / np.sqrt(len(pairwise_means)))
    
    return mean_overlap, std_error, std_dev


def compute_curriculum_overlap(
    run_dirs: list[Path],
    word_counts_map: dict[int, int],
    k: int = 30,
    start_tranche: int = 1,
    end_tranche: int = 99999,
    sample_every: int = 1,
    word_sample_frac: float = 1.0,
    seed: int = 42
) -> tuple[list[int], list[float], list[float], list[float]]:
    """
    Compute per-tranche overlap across multiple runs, returning unique word counts as x-values.
    
    Args:
        run_dirs: List of 2-5 run directories
        word_counts_map: Dictionary mapping tranche_index -> cumulative_unique_words
        k: Number of nearest neighbors
        start_tranche: First tranche to include (inclusive)
        end_tranche: Last tranche to include (inclusive)
        sample_every: Process every N-th tranche within the range
        word_sample_frac: Fraction of words to sample (1.0 = all)
        seed: Random seed for word sampling
    
    Returns:
        unique_word_counts: List of cumulative unique word counts (x-axis values)
        overlaps: List of overlap values
        errors: List of standard errors for each overlap
        std_devs: List of standard deviations for each overlap
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
    
    unique_word_counts = []
    overlaps = []
    errors = []
    std_devs = []
    
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
        
        # Get cumulative unique word count for this tranche
        if tranche_num not in word_counts_map:
            print(f"Warning: Tranche {tranche_num} not found in word_counts_map, skipping")
            continue
        
        cumulative_words = word_counts_map[tranche_num]
        
        # Get paths for all runs
        tranche_paths = [tranche_path1] + [tm[tranche_name] for tm in tranche_maps]
        
        try:
            # Load embeddings for all runs
            embeddings_data = [load_tranche_embeddings(p) for p in tranche_paths]
            
            # Use tranche number as additional seed component for reproducibility
            tranche_seed = seed + tranche_num
            
            overlap, error, std_dev = compute_knn_overlap_n_runs(
                embeddings_data,
                k=k,
                word_sample_frac=word_sample_frac,
                seed=tranche_seed
            )
            
            if not np.isnan(overlap):
                unique_word_counts.append(cumulative_words)
                overlaps.append(overlap)
                errors.append(error)
                std_devs.append(std_dev)
        except Exception as e:
            print(f"Warning: Failed to process {tranche_name}: {e}")
    
    return unique_word_counts, overlaps, errors, std_devs


def format_tranche_type_display(tranche_type: str | None) -> str:
    """
    Format tranche type for display in plot title.
    """
    if tranche_type is None:
        return ""
    
    mapping = {
        "word-based": "Unique Word-Based",
        "sentence-based": "Sentence-Based",
        "word-count": "Word-Count",
        "matching": "Matching",
    }
    return mapping.get(tranche_type, tranche_type.replace("-", " ").title())


def extract_curriculum_name_from_path(path: str | Path) -> str:
    """
    Extract curriculum name from a run path.
    Examples:
        "outputs/embeddings/aoa_50d_0" -> "AoA"
        "outputs/embeddings/freq_50d_0" -> "Frequency"
        "outputs/embeddings/shuffled_50d_0" -> "Shuffled"
        "outputs/embeddings/conc_50d_0" -> "Concreteness"
    """
    path_str = str(path)
    # Extract the directory name (last component)
    dir_name = Path(path_str).name
    
    # Common curriculum name mappings
    curriculum_map = {
        "aoa": "AoA",
        "freq": "Frequency",
        "frequency": "Frequency",
        "shuffled": "Shuffled",
        "random": "Shuffled",
        "conc": "Concreteness",
        "concreteness": "Concreteness",
        "phon": "Phonology",
        "phonology": "Phonology",
    }
    
    # Try to find curriculum name in path
    path_lower = path_str.lower()
    for key, display_name in curriculum_map.items():
        if key in path_lower:
            return display_name
    
    # Fallback: capitalize first part of directory name before underscore
    if "_" in dir_name:
        base_name = dir_name.split("_")[0]
        return base_name.capitalize()
    
    return dir_name.capitalize()


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
        description="Neighbor overlap analysis with unique word counts on x-axis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full   All tranches, 100%% words, 2 runs (default)
  fast   Every 10 tranches, 50%% words, 3 runs
  early  Tranches 30-200, 100%% words, 3 runs
  late   Tranches 950+, 50%% words, 3 runs

Examples:
  python plot_neighbor_overlap_by_words.py --curriculum1_idx 0 --curriculum2_idx 1 --mode fast
  python plot_neighbor_overlap_by_words.py --curriculum1_idx 0 --curriculum2_idx 1 --num_runs 5
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
    
    # Curriculum paths/indices - required
    parser.add_argument(
        "--curriculum1_idx",
        type=str,
        default=None,
        help="Curriculum index (e.g., '0' or '1') for first curriculum. Alternative to --curriculum1_path.",
    )
    parser.add_argument(
        "--curriculum1_path",
        type=str,
        default=None,
        help="Full path to first curriculum directory. Alternative to --curriculum1_idx.",
    )
    parser.add_argument(
        "--curriculum2_idx",
        type=str,
        default=None,
        help="Curriculum index (e.g., '0' or '1') for second curriculum. Alternative to --curriculum2_path.",
    )
    parser.add_argument(
        "--curriculum2_path",
        type=str,
        default=None,
        help="Full path to second curriculum directory. Alternative to --curriculum2_idx.",
    )
    
    # Run paths - AoA curriculum (up to 5 runs)
    parser.add_argument(
        "--curriculum1_run1",
        type=str,
        default="outputs/embeddings/aoa_50d_0",
        help="Path to first curriculum run 1.",
    )
    parser.add_argument(
        "--curriculum1_run2",
        type=str,
        default="outputs/embeddings/aoa_50d_1",
        help="Path to first curriculum run 2.",
    )
    parser.add_argument(
        "--curriculum1_run3",
        type=str,
        default="outputs/embeddings/aoa_50d_2",
        help="Path to first curriculum run 3 (only used if --num_runs >= 3).",
    )
    parser.add_argument(
        "--curriculum1_run4",
        type=str,
        default="outputs/embeddings/aoa_50d_3",
        help="Path to first curriculum run 4 (only used if --num_runs >= 4).",
    )
    parser.add_argument(
        "--curriculum1_run5",
        type=str,
        default="outputs/embeddings/aoa_50d_4",
        help="Path to first curriculum run 5 (only used if --num_runs >= 5).",
    )
    
    # Run paths - Shuffled curriculum (up to 5 runs)
    parser.add_argument(
        "--curriculum2_run1",
        type=str,
        default="outputs/embeddings/shuffled_50d_0",
        help="Path to second curriculum run 1.",
    )
    parser.add_argument(
        "--curriculum2_run2",
        type=str,
        default="outputs/embeddings/shuffled_50d_1",
        help="Path to second curriculum run 2.",
    )
    parser.add_argument(
        "--curriculum2_run3",
        type=str,
        default="outputs/embeddings/shuffled_50d_2",
        help="Path to second curriculum run 3 (only used if --num_runs >= 3).",
    )
    parser.add_argument(
        "--curriculum2_run4",
        type=str,
        default="outputs/embeddings/shuffled_50d_3",
        help="Path to second curriculum run 4 (only used if --num_runs >= 4).",
    )
    parser.add_argument(
        "--curriculum2_run5",
        type=str,
        default="outputs/embeddings/shuffled_50d_4",
        help="Path to second curriculum run 5 (only used if --num_runs >= 5).",
    )
    
    # Analysis parameters
    parser.add_argument(
        "--curriculum1_num_runs",
        type=int,
        default=None,
        choices=[2, 3, 4, 5],
        help="Number of first curriculum runs to compare (2-5). Defaults to --num_runs if set, else mode setting.",
    )
    parser.add_argument(
        "--curriculum2_num_runs",
        type=int,
        default=None,
        choices=[2, 3, 4, 5],
        help="Number of second curriculum runs to compare (2-5). Defaults to --num_runs if set, else mode setting.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=None,
        choices=[2, 3, 4, 5],
        help="Number of runs to compare for both curricula (2-5). Overridden by --curriculum1_num_runs and --curriculum2_num_runs. Defaults to mode setting.",
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
        "--curriculum1_name",
        type=str,
        default=None,
        help="Display name for first curriculum (e.g., 'AoA', 'Frequency', 'Concreteness'). Defaults to extracting from run path.",
    )
    parser.add_argument(
        "--curriculum2_name",
        type=str,
        default=None,
        help="Display name for second curriculum (e.g., 'Shuffled', 'Random'). Defaults to extracting from run path.",
    )
    parser.add_argument(
        "--curriculum1_tranche_type",
        type=str,
        default=None,
        choices=["word-based", "sentence-based", "word-count", "matching"],
        help="Tranche type for first curriculum: 'word-based' (unique words), 'sentence-based', 'word-count' (total words), or 'matching'.",
    )
    parser.add_argument(
        "--curriculum2_tranche_type",
        type=str,
        default=None,
        choices=["word-based", "sentence-based", "word-count", "matching"],
        help="Tranche type for second curriculum: 'word-based' (unique words), 'sentence-based', 'word-count' (total words), or 'matching'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot. Defaults to outputs/figures/{curriculum1}_vs_{curriculum2}_overlap_by_words{suffix}.png",
    )
    parser.add_argument(
        "--word_count_source",
        type=str,
        choices=["curriculum", "training"],
        default="curriculum",
        help="Source of word counts: 'curriculum' uses unique_word_counts.json from curriculum dir, "
             "'training' uses metadata.json from embeddings dir (default: curriculum).",
    )
    parser.add_argument(
        "--word_count_type",
        type=str,
        choices=["seen", "trained"],
        default="seen",
        help="Type of word count when using --word_count_source=training: "
             "'seen' = cumulative unique words encountered, "
             "'trained' = words in model vocabulary (above min_count). "
             "(default: seen). Ignored when --word_count_source=curriculum.",
    )
    
    args = parser.parse_args()
    
    # Validate curriculum paths/indices
    if args.curriculum1_idx is None and args.curriculum1_path is None:
        parser.error("Must provide either --curriculum1_idx or --curriculum1_path")
    if args.curriculum2_idx is None and args.curriculum2_path is None:
        parser.error("Must provide either --curriculum2_idx or --curriculum2_path")
    
    # Resolve curriculum paths
    curriculum1_path = args.curriculum1_path if args.curriculum1_path else args.curriculum1_idx
    curriculum2_path = args.curriculum2_path if args.curriculum2_path else args.curriculum2_idx
    
    # Get mode defaults and apply custom overrides
    mode_defaults = get_mode_defaults(args.mode)
    
    num_runs = args.num_runs if args.num_runs is not None else mode_defaults["num_runs"]
    curriculum1_num_runs = args.curriculum1_num_runs if args.curriculum1_num_runs is not None else num_runs
    curriculum2_num_runs = args.curriculum2_num_runs if args.curriculum2_num_runs is not None else num_runs
    start_tranche = args.start_tranche if args.start_tranche is not None else mode_defaults["start_tranche"]
    end_tranche = args.end_tranche if args.end_tranche is not None else mode_defaults["end_tranche"]
    sample_every = args.sample_every if args.sample_every is not None else mode_defaults["sample_every"]
    word_sample_frac = args.word_sample_frac if args.word_sample_frac is not None else mode_defaults["word_sample_frac"]
    
    # Collect run directories based on separate num_runs
    curriculum1_run_paths = [
        args.curriculum1_run1, args.curriculum1_run2, args.curriculum1_run3, 
        args.curriculum1_run4, args.curriculum1_run5
    ]
    curriculum2_run_paths = [
        args.curriculum2_run1, args.curriculum2_run2, args.curriculum2_run3, 
        args.curriculum2_run4, args.curriculum2_run5
    ]
    
    curriculum1_runs = [Path(p) for p in curriculum1_run_paths[:curriculum1_num_runs]]
    curriculum2_runs = [Path(p) for p in curriculum2_run_paths[:curriculum2_num_runs]]
    
    # Extract or use provided curriculum names
    curriculum1_name = args.curriculum1_name
    if curriculum1_name is None and curriculum1_runs:
        curriculum1_name = extract_curriculum_name_from_path(curriculum1_runs[0])
    
    curriculum2_name = args.curriculum2_name
    if curriculum2_name is None and curriculum2_runs:
        curriculum2_name = extract_curriculum_name_from_path(curriculum2_runs[0])
    
    # Fallback defaults if extraction fails
    if curriculum1_name is None:
        curriculum1_name = "Curriculum 1"
    if curriculum2_name is None:
        curriculum2_name = "Curriculum 2"
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Create safe filename from curriculum names
        safe_name1 = curriculum1_name.lower().replace(" ", "_")
        safe_name2 = curriculum2_name.lower().replace(" ", "_")
        output_path = Path(f"outputs/figures/{safe_name1}_vs_{safe_name2}_overlap_by_words{mode_defaults['output_suffix']}.png")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load word counts based on source selection
    print(f"\n{'='*60}")
    print(f"Loading word counts")
    print(f"{'='*60}")
    print(f"  Source: {args.word_count_source}")
    if args.word_count_source == "training":
        print(f"  Type: {args.word_count_type} (unique_words_{args.word_count_type})")
    
    if args.word_count_source == "curriculum":
        # Load from curriculum's unique_word_counts.json
        print(f"  Curriculum 1 ({curriculum1_name}): {curriculum1_path}")
        try:
            curriculum1_word_counts = load_unique_word_counts(curriculum1_path)
            print(f"    Loaded {len(curriculum1_word_counts)} tranche mappings")
        except Exception as e:
            print(f"    ERROR: {e}")
            return
        
        print(f"  Curriculum 2 ({curriculum2_name}): {curriculum2_path}")
        try:
            curriculum2_word_counts = load_unique_word_counts(curriculum2_path)
            print(f"    Loaded {len(curriculum2_word_counts)} tranche mappings")
        except Exception as e:
            print(f"    ERROR: {e}")
            return
    else:
        # Load from training's metadata.json (use first run directory as source)
        print(f"  Curriculum 1 ({curriculum1_name}): {curriculum1_runs[0]}")
        try:
            curriculum1_word_counts = load_word_counts_from_metadata(
                curriculum1_runs[0], 
                word_count_type=args.word_count_type
            )
            print(f"    Loaded {len(curriculum1_word_counts)} tranche mappings (unique_words_{args.word_count_type})")
        except Exception as e:
            print(f"    ERROR: {e}")
            return
        
        print(f"  Curriculum 2 ({curriculum2_name}): {curriculum2_runs[0]}")
        try:
            curriculum2_word_counts = load_word_counts_from_metadata(
                curriculum2_runs[0],
                word_count_type=args.word_count_type
            )
            print(f"    Loaded {len(curriculum2_word_counts)} tranche mappings (unique_words_{args.word_count_type})")
        except Exception as e:
            print(f"    ERROR: {e}")
            return
    
    # Calculate number of pairs for display
    curriculum1_num_pairs = curriculum1_num_runs * (curriculum1_num_runs - 1) // 2
    curriculum2_num_pairs = curriculum2_num_runs * (curriculum2_num_runs - 1) // 2
    
    # Print configuration
    sample_pct = int(word_sample_frac * 100)
    range_str = f"{start_tranche}-{end_tranche}" if end_tranche < 99999 else f"{start_tranche}+"
    
    print(f"\n{'='*60}")
    print(f"Neighbor Overlap Analysis (by Unique Word Count)")
    print(f"{'='*60}")
    print(f"  Mode: {args.mode}")
    print(f"  {curriculum1_name} runs: {curriculum1_num_runs} ({curriculum1_num_pairs} pairwise comparisons)")
    if args.curriculum1_tranche_type:
        print(f"    Tranche type: {format_tranche_type_display(args.curriculum1_tranche_type)}")
    print(f"  {curriculum2_name} runs: {curriculum2_num_runs} ({curriculum2_num_pairs} pairwise comparisons)")
    if args.curriculum2_tranche_type:
        print(f"    Tranche type: {format_tranche_type_display(args.curriculum2_tranche_type)}")
    print(f"  Tranche range: {range_str}")
    print(f"  Tranche sampling: every {sample_every}")
    print(f"  Word sampling: {sample_pct}%")
    print(f"  k: {args.k} nearest neighbors")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    # Compute overlaps for first curriculum
    print(f"\n=== {curriculum1_name} Curriculum ({curriculum1_num_runs} runs) ===")
    curriculum1_word_counts_list, curriculum1_overlaps, curriculum1_errors, curriculum1_std_devs = compute_curriculum_overlap(
        curriculum1_runs,
        curriculum1_word_counts,
        k=args.k,
        start_tranche=start_tranche,
        end_tranche=end_tranche,
        sample_every=sample_every,
        word_sample_frac=word_sample_frac,
        seed=args.seed
    )
    
    # Compute overlaps for second curriculum
    print(f"\n=== {curriculum2_name} Curriculum ({curriculum2_num_runs} runs) ===")
    curriculum2_word_counts_list, curriculum2_overlaps, curriculum2_errors, curriculum2_std_devs = compute_curriculum_overlap(
        curriculum2_runs,
        curriculum2_word_counts,
        k=args.k,
        start_tranche=start_tranche,
        end_tranche=end_tranche,
        sample_every=sample_every,
        word_sample_frac=word_sample_frac,
        seed=args.seed
    )
    
    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(14, 7))
    
    # Determine marker size based on number of points
    n_points = max(len(curriculum1_word_counts_list), len(curriculum2_word_counts_list))
    marker_size = 4 if n_points < 200 else 2 if n_points < 500 else 1
    use_markers = n_points < 300
    
    # Convert errors/std devs to numpy arrays for plotting
    curriculum1_errors_array = np.array(curriculum1_errors)
    curriculum2_errors_array = np.array(curriculum2_errors)
    curriculum1_std_dev = np.array(curriculum1_std_devs)
    curriculum2_std_dev = np.array(curriculum2_std_devs)
    
    # Draw shaded error bands first (behind lines) so they are visible
    # Upper/lower = mean ± std (run-pair std). Replace NaN with 0 so bands plot.
    curriculum1_upper = np.array(curriculum1_overlaps) + np.nan_to_num(curriculum1_std_dev, nan=0.0)
    curriculum1_lower = np.array(curriculum1_overlaps) - np.nan_to_num(curriculum1_std_dev, nan=0.0)
    plt.fill_between(
        curriculum1_word_counts_list, curriculum1_lower, curriculum1_upper,
        color="#87CEEB", alpha=0.2, label=f"{curriculum1_name} ± std"
    )
    curriculum2_upper = np.array(curriculum2_overlaps) + np.nan_to_num(curriculum2_std_dev, nan=0.0)
    curriculum2_lower = np.array(curriculum2_overlaps) - np.nan_to_num(curriculum2_std_dev, nan=0.0)
    plt.fill_between(
        curriculum2_word_counts_list, curriculum2_lower, curriculum2_upper,
        color="#FFB366", alpha=0.2, label=f"{curriculum2_name} ± std"
    )
    
    # Plot mean lines on top of bands
    plt.plot(
        curriculum1_word_counts_list, curriculum1_overlaps,
        linewidth=2, color="#2E86AB", alpha=1.0, label=f"{curriculum1_name} Curriculum",
        marker='o' if use_markers else None, markersize=marker_size
    )
    plt.plot(
        curriculum2_word_counts_list, curriculum2_overlaps,
        linewidth=2, color="#E94F37", alpha=1.0, label=f"{curriculum2_name} Curriculum",
        marker='s' if use_markers else None, markersize=marker_size
    )
    
    # Determine x-axis label based on word count source/type
    if args.word_count_source == "curriculum":
        x_label = "Cumulative Unique Words (from Curriculum)"
        x_axis_subtitle = "(X-axis: Cumulative Unique Words)"
    elif args.word_count_type == "seen":
        x_label = "Unique Words Seen During Training"
        x_axis_subtitle = "(X-axis: Unique Words Seen)"
    else:  # trained
        x_label = "Unique Words Trained (in Vocabulary)"
        x_axis_subtitle = "(X-axis: Unique Words Trained)"
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(f"k={args.k} Nearest Neighbor Overlap", fontsize=12)
    
    # Build title with tranche type information
    run_str = f"{num_runs} Runs"
    if sample_every > 1:
        title_range = f"Every {sample_every} Tranches"
    else:
        title_range = f"Tranches {range_str}"
    title_words = f"{sample_pct}% Words" if sample_pct < 100 else "All Words"
    
    # Format tranche types for display
    tranche_type1_display = format_tranche_type_display(args.curriculum1_tranche_type)
    tranche_type2_display = format_tranche_type_display(args.curriculum2_tranche_type)
    
    # Build tranche type string
    if tranche_type1_display and tranche_type2_display:
        if tranche_type1_display == tranche_type2_display:
            tranche_type_str = f"({tranche_type1_display} Tranches)"
        else:
            tranche_type_str = f"({tranche_type1_display} vs {tranche_type2_display} Tranches)"
    elif tranche_type1_display:
        tranche_type_str = f"({tranche_type1_display} Tranches)"
    elif tranche_type2_display:
        tranche_type_str = f"({tranche_type2_display} Tranches)"
    else:
        tranche_type_str = ""
    
    # Combine title components
    title_parts = [
        f"Embedding Stability: {curriculum1_name} vs {curriculum2_name} Curriculum",
        tranche_type_str,
        f"{title_range}, {title_words}, {run_str}",
        x_axis_subtitle
    ]
    title = "\n".join([p for p in title_parts if p])  # Remove empty parts
    
    plt.title(title, fontsize=13)
    
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add mean annotations
    if curriculum1_overlaps:
        curriculum1_mean = np.mean(curriculum1_overlaps)
        plt.axhline(y=curriculum1_mean, color="#2E86AB", linestyle="--", alpha=0.5)
        x_pos = min(curriculum1_word_counts_list) + (max(curriculum1_word_counts_list) - min(curriculum1_word_counts_list)) * 0.02
        plt.text(x_pos, curriculum1_mean + 0.02, f"{curriculum1_name} mean: {curriculum1_mean:.3f}", color="#2E86AB", fontsize=10)
    
    if curriculum2_overlaps:
        curriculum2_mean = np.mean(curriculum2_overlaps)
        plt.axhline(y=curriculum2_mean, color="#E94F37", linestyle="--", alpha=0.5)
        x_pos = min(curriculum2_word_counts_list) + (max(curriculum2_word_counts_list) - min(curriculum2_word_counts_list)) * 0.02
        plt.text(x_pos, curriculum2_mean - 0.04, f"{curriculum2_name} mean: {curriculum2_mean:.3f}", color="#E94F37", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Plot saved to: {output_path}")
    print(f"Points processed: {curriculum1_name}={len(curriculum1_word_counts_list)}, {curriculum2_name}={len(curriculum2_word_counts_list)}")
    if curriculum1_overlaps:
        print(f"{curriculum1_name}: mean={np.mean(curriculum1_overlaps):.4f}, min={np.min(curriculum1_overlaps):.4f}, max={np.max(curriculum1_overlaps):.4f}")
    if curriculum2_overlaps:
        print(f"{curriculum2_name}: mean={np.mean(curriculum2_overlaps):.4f}, min={np.min(curriculum2_overlaps):.4f}, max={np.max(curriculum2_overlaps):.4f}")
    if curriculum1_std_dev.size:
        print(f"{curriculum1_name} std deviation (run-pair): mean={np.mean(curriculum1_std_dev):.4f}, min={np.min(curriculum1_std_dev):.4f}, max={np.max(curriculum1_std_dev):.4f}")
    if curriculum2_std_dev.size:
        print(f"{curriculum2_name} std deviation (run-pair): mean={np.mean(curriculum2_std_dev):.4f}, min={np.min(curriculum2_std_dev):.4f}, max={np.max(curriculum2_std_dev):.4f}")


if __name__ == "__main__":
    main()
