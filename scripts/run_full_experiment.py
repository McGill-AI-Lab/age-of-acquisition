#!/usr/bin/env python
"""
Full experiment automation script.

This script automates the entire training and evaluation pipeline:

1. Clear existing curricula and embeddings
2. Build AoA curriculum (ordered by age-of-acquisition)
3. Train 3 models with different within-tranche shuffles
4. Build Shuffled curriculum (random order)
5. Train 3 models with different within-tranche shuffles
6. Run neighbor overlap comparison plot

Usage:
    python scripts/run_full_experiment.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
TRAINING_DIR = PROJECT_ROOT / "data" / "processed" / "corpora" / "training"
EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Training parameters
VECTOR_SIZE = 50
NUM_RUNS = 3


def run_python_code(code: str, description: str) -> None:
    """Run Python code as a subprocess."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"[DONE] {description} completed")


def run_command(cmd: list[str], description: str) -> None:
    """Run a command as a subprocess."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print('='*60)
    
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"[DONE] {description} completed")


def safe_rmtree(path: Path, max_retries: int = 3) -> bool:
    """Safely remove a directory tree with retries for Windows file locks."""
    import time
    for attempt in range(max_retries):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} (file locked)...")
                time.sleep(2)
            else:
                print(f"    Warning: Could not remove {path.name}, skipping")
                return False
    return False


def clear_directories() -> None:
    """Clear existing curricula and embeddings directories."""
    print("\n" + "="*60)
    print("STEP: Clearing existing data")
    print("="*60)
    
    # Clear training curricula
    if TRAINING_DIR.exists():
        for item in TRAINING_DIR.iterdir():
            if item.is_dir():
                print(f"  Removing {item.name}")
                safe_rmtree(item)
    
    # Clear embeddings
    if EMBEDDINGS_DIR.exists():
        for item in EMBEDDINGS_DIR.iterdir():
            if item.is_dir():
                print(f"  Removing {item.name}")
                safe_rmtree(item)
    
    # Create fresh directories
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[DONE] Directories cleared")


def build_aoa_curriculum() -> str:
    """Build AoA curriculum and return its index."""
    code = """
from curricula import build_curriculum

idx = build_curriculum(
    curriculum='aoa',
    scoring_method='max',
    sort_order='asc',
    tranche_type='word-based',
    tranche_size=500,
    aoa_agnostic=True,
    multiword=False,
    skip_stopwords=True,
    inflect=True
)
print(f"AOA_CURRICULUM_INDEX={idx}")
"""
    run_python_code(code, "Building AoA curriculum")
    return "0"  # First curriculum will be index 0


def build_shuffled_curriculum() -> str:
    """Build shuffled curriculum and return its index."""
    code = """
from curricula import build_curriculum

idx = build_curriculum(
    curriculum='shuffled',
    scoring_method='max',
    sort_order='asc',
    tranche_type='word-based',
    tranche_size=500,
    aoa_agnostic=True,
    multiword=False,
    skip_stopwords=True,
    inflect=True
)
print(f"SHUFFLED_CURRICULUM_INDEX={idx}")
"""
    run_python_code(code, "Building Shuffled curriculum")
    return "1"  # Second curriculum will be index 1


def shuffle_curriculum(idx: str) -> None:
    """Shuffle sentences within tranches of a curriculum."""
    code = f"""
from curricula import shuffle_tranches
seed = shuffle_tranches('{idx}')
print(f"Shuffled with seed: {{seed}}")
"""
    run_python_code(code, f"Shuffling sentences in curriculum {idx}")


def get_curriculum_tranches_path(idx: str) -> Path:
    """Get the tranches path for a curriculum by index."""
    for item in TRAINING_DIR.iterdir():
        if item.is_dir() and item.name.startswith(f"{idx:>03s}_"):
            return item / "tranches"
        if item.is_dir() and item.name.startswith(f"{int(idx):03d}_"):
            return item / "tranches"
    
    # Fallback: search for curriculum
    for item in TRAINING_DIR.iterdir():
        if item.is_dir():
            parts = item.name.split("_")
            if parts[0] == idx or parts[0] == f"{int(idx):03d}":
                return item / "tranches"
    
    raise FileNotFoundError(f"Could not find curriculum with index {idx}")


def train_model(curriculum_idx: str, output_name: str) -> None:
    """Train Word2Vec model on a curriculum."""
    tranches_path = get_curriculum_tranches_path(curriculum_idx)
    output_dir = EMBEDDINGS_DIR / output_name
    
    cmd = [
        sys.executable, "-m", "src.training.train_word2vec_tranches",
        "--input_dir", str(tranches_path),
        "--output_dir", str(output_dir),
        "--vector_size", str(VECTOR_SIZE),
    ]
    
    run_command(cmd, f"Training model: {output_name}")


def run_overlap_plot() -> None:
    """Run the neighbor overlap comparison plot."""
    cmd = [
        sys.executable, "scripts/plot_neighbor_overlap.py",
        "--aoa_run1", str(EMBEDDINGS_DIR / "aoa_50d_0"),
        "--aoa_run2", str(EMBEDDINGS_DIR / "aoa_50d_1"),
        "--aoa_run3", str(EMBEDDINGS_DIR / "aoa_50d_2"),
        "--shuffled_run1", str(EMBEDDINGS_DIR / "shuffled_50d_0"),
        "--shuffled_run2", str(EMBEDDINGS_DIR / "shuffled_50d_1"),
        "--shuffled_run3", str(EMBEDDINGS_DIR / "shuffled_50d_2"),
        "--output", str(FIGURES_DIR / "aoa_vs_shuffled_overlap.png"),
        "--sample_every", "10",  # Sample every 10 tranches for speed
    ]
    
    run_command(cmd, "Generating overlap comparison plot")


def main():
    print("\n" + "#"*60)
    print("# FULL EXPERIMENT AUTOMATION")
    print("# AoA vs Shuffled Curriculum Comparison")
    print("#"*60)
    
    # Step 1: Clear existing data
    clear_directories()
    
    # Step 2: Build and train AoA curriculum (3 runs)
    print("\n" + "#"*60)
    print("# PHASE 1: AoA CURRICULUM")
    print("#"*60)
    
    aoa_idx = build_aoa_curriculum()
    
    for run in range(NUM_RUNS):
        if run > 0:
            # Shuffle sentences within tranches for runs 1 and 2
            shuffle_curriculum(aoa_idx)
        
        train_model(aoa_idx, f"aoa_50d_{run}")
    
    # Step 3: Build and train Shuffled curriculum (3 runs)
    print("\n" + "#"*60)
    print("# PHASE 2: SHUFFLED CURRICULUM")
    print("#"*60)
    
    shuffled_idx = build_shuffled_curriculum()
    
    for run in range(NUM_RUNS):
        if run > 0:
            # Shuffle sentences within tranches for runs 1 and 2
            shuffle_curriculum(shuffled_idx)
        
        train_model(shuffled_idx, f"shuffled_50d_{run}")
    
    # Step 4: Generate comparison plot
    print("\n" + "#"*60)
    print("# PHASE 3: ANALYSIS")
    print("#"*60)
    
    run_overlap_plot()
    
    # Done
    print("\n" + "#"*60)
    print("# EXPERIMENT COMPLETE!")
    print("#"*60)
    print(f"\nResults:")
    print(f"  - AoA models: {EMBEDDINGS_DIR / 'aoa_50d_*'}")
    print(f"  - Shuffled models: {EMBEDDINGS_DIR / 'shuffled_50d_*'}")
    print(f"  - Overlap plot: {FIGURES_DIR / 'aoa_vs_shuffled_overlap.png'}")
    print()


if __name__ == "__main__":
    main()

