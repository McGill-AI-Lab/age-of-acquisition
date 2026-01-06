# Nearest Neighbor Overlap Analysis

This script (`plot_neighbor_overlap.py`) measures **embedding stability** by comparing word embeddings across multiple training runs. It computes the k-nearest neighbor overlap between runs to quantify how consistently words cluster together in the embedding space.

## How It Works

1. **Load Embeddings**: For each tranche, load word embeddings from multiple training runs
2. **Find Common Words**: Identify words present in all runs being compared
3. **Compute k-NN**: For each word, find its k nearest neighbors in each run's embedding space
4. **Calculate Overlap**: Measure how many neighbors are shared between runs (pairwise)
5. **Average**: Report the mean overlap across all word pairs and all run pairs

Higher overlap = more stable embeddings (words have consistent neighbors across runs).

## Quick Start

```bash
# Basic usage with default settings (2 runs, all tranches)
python scripts/plot_neighbor_overlap.py

# Fast analysis (recommended for initial exploration)
python scripts/plot_neighbor_overlap.py --mode fast

# Early tranches only (where AoA effect is strongest)
python scripts/plot_neighbor_overlap.py --mode early

# Late tranches analysis
python scripts/plot_neighbor_overlap.py --mode late
```

## Analysis Modes

| Mode | Tranches | Word Sampling | Default Runs | Use Case |
|------|----------|---------------|--------------|----------|
| `full` | All (1+) | 100% | 2 | Complete analysis |
| `fast` | Every 10th | 50% | 3 | Quick overview |
| `early` | 30-200 | 100% | 3 | Early training dynamics |
| `late` | 950+ | 50% | 3 | Late training stability |

## Command Line Arguments

### Mode Selection

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `full` | Preset mode: `full`, `fast`, `early`, `late` |

### Run Paths

The script compares two curricula (AoA vs Shuffled), each with up to 5 training runs:

| Argument | Default | Description |
|----------|---------|-------------|
| `--aoa_run1` | `outputs/embeddings/aoa_50d_0` | AoA curriculum run 1 |
| `--aoa_run2` | `outputs/embeddings/aoa_50d_1` | AoA curriculum run 2 |
| `--aoa_run3` | `outputs/embeddings/aoa_50d_2` | AoA curriculum run 3 |
| `--aoa_run4` | `outputs/embeddings/aoa_50d_3` | AoA curriculum run 4 |
| `--aoa_run5` | `outputs/embeddings/aoa_50d_4` | AoA curriculum run 5 |
| `--shuffled_run1` | `outputs/embeddings/shuffled_50d_0` | Shuffled curriculum run 1 |
| `--shuffled_run2` | `outputs/embeddings/shuffled_50d_1` | Shuffled curriculum run 2 |
| `--shuffled_run3` | `outputs/embeddings/shuffled_50d_2` | Shuffled curriculum run 3 |
| `--shuffled_run4` | `outputs/embeddings/shuffled_50d_3` | Shuffled curriculum run 4 |
| `--shuffled_run5` | `outputs/embeddings/shuffled_50d_4` | Shuffled curriculum run 5 |

### Analysis Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_runs` | int | mode default | Number of runs to compare (2-5) |
| `--k` | int | 30 | Number of nearest neighbors |
| `--start_tranche` | int | mode default | First tranche to include |
| `--end_tranche` | int | mode default | Last tranche to include |
| `--sample_every` | int | mode default | Process every N-th tranche |
| `--word_sample_frac` | float | mode default | Fraction of words to sample (0.0-1.0) |
| `--seed` | int | 42 | Random seed for word sampling |
| `--output` | str | auto | Output path for the plot |

## Examples

### Basic Examples

```bash
# Full analysis with 3 runs
python scripts/plot_neighbor_overlap.py --mode full --num_runs 3

# Fast mode (every 10 tranches, 50% words, 3 runs)
python scripts/plot_neighbor_overlap.py --mode fast

# Early tranches (30-200) with full word coverage
python scripts/plot_neighbor_overlap.py --mode early

# Late tranches (950+) with 50% word sampling
python scripts/plot_neighbor_overlap.py --mode late
```

### Custom Analysis

```bash
# Analyze tranches 100-500 with 75% word sampling
python scripts/plot_neighbor_overlap.py \
    --start_tranche 100 \
    --end_tranche 500 \
    --word_sample_frac 0.75 \
    --num_runs 3

# Every 5th tranche, 60 nearest neighbors
python scripts/plot_neighbor_overlap.py \
    --sample_every 5 \
    --k 60 \
    --num_runs 2

# Use 5 runs for maximum statistical power
python scripts/plot_neighbor_overlap.py --num_runs 5 --mode fast
```

### Custom Paths

```bash
# Use custom embedding directories
python scripts/plot_neighbor_overlap.py \
    --aoa_run1 path/to/aoa/run1 \
    --aoa_run2 path/to/aoa/run2 \
    --shuffled_run1 path/to/shuffled/run1 \
    --shuffled_run2 path/to/shuffled/run2 \
    --num_runs 2

# Custom output path
python scripts/plot_neighbor_overlap.py \
    --mode fast \
    --output outputs/figures/my_analysis.png
```

## Output

The script generates:

1. **Plot** (`outputs/figures/aoa_vs_shuffled_overlap{suffix}.png`):
   - Two lines showing overlap over training tranches
   - Blue: AoA curriculum
   - Red: Shuffled curriculum
   - Dashed lines: Mean overlap for each curriculum

2. **Console Statistics**:
   - Mean, min, max overlap for each curriculum
   - Number of tranches processed

## Interpreting Results

- **Higher overlap** = More stable embeddings
- **AoA > Shuffled**: Age-of-acquisition ordering improves stability
- **Early vs Late**: Compare stability at different training stages
- **Effect size**: Typical differences are 1-5% in overlap

## Number of Pairwise Comparisons

| Runs | Pairs | Formula |
|------|-------|---------|
| 2 | 1 | C(2,2) = 1 |
| 3 | 3 | C(3,2) = 3 |
| 4 | 6 | C(4,2) = 6 |
| 5 | 10 | C(5,2) = 10 |

More runs provide more robust overlap estimates but require more embedding data.

## Requirements

- Python 3.8+
- numpy
- pyarrow (for parquet files)
- matplotlib
- scikit-learn
- tqdm

## File Structure Expected

```
outputs/embeddings/
├── aoa_50d_0/
│   ├── tranche_0.parquet
│   ├── tranche_1.parquet
│   └── ...
├── aoa_50d_1/
│   └── ...
├── shuffled_50d_0/
│   └── ...
└── shuffled_50d_1/
    └── ...
```

Each parquet file should contain columns:
- `word`: string - the word
- `embedding`: array - the embedding vector

