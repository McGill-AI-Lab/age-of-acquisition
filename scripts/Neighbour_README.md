# Nearest Neighbor Overlap Analysis

This script (`plot_neighbor_overlap.py`) measures **embedding stability** by comparing word embeddings across multiple training runs. It computes the k-nearest neighbor overlap between runs to quantify how consistently words cluster together in the embedding space.

## Recent Updates

### Dynamic Curriculum Labels
- **Auto-detection**: Curriculum names are automatically extracted from run directory paths
- **Manual override**: Use `--curriculum1_name` and `--curriculum2_name` to specify custom labels
- **Flexible comparisons**: Compare any curricula (not just AoA vs Shuffled)

### Tranche Type Display
- **New flags**: `--curriculum1_tranche_type` and `--curriculum2_tranche_type` specify tranche types
- **Plot titles**: Tranche type information is now included in plot titles
- **Supported types**: `word-based` (unique words), `sentence-based`, `word-count`, `matching`
- **Display format**: Shows as "Unique Word-Based", "Sentence-Based", etc. in titles

### Error Bars
- **Standard error bars**: Plots now include error bars showing the standard error of the mean overlap
- **Variability measurement**: Error bars represent variability in overlap across words within each tranche
- **Calculation**: Standard error = σ / √n, where σ is the standard deviation of word-level overlaps and n is the number of words

## How It Works

1. **Load Embeddings**: For each tranche, load word embeddings from multiple training runs
2. **Find Common Words**: Identify words present in all runs being compared
3. **Compute k-NN**: For each word, find its k nearest neighbors in each run's embedding space
4. **Calculate Overlap**: Measure how many neighbors are shared between runs (pairwise)
5. **Average**: Report the mean overlap across all word pairs and all run pairs
6. **Compute Error Bars**: Calculate standard error of the mean overlap across words

Higher overlap = more stable embeddings (words have consistent neighbors across runs).

### Error Bar Calculation

For each tranche, the script computes:
1. **Word-level overlaps**: For each word, compute the average pairwise overlap across all run pairs
2. **Mean overlap**: Average across all words: `μ = mean(word_overlaps)`
3. **Standard deviation**: Sample standard deviation: `σ = std(word_overlaps, ddof=1)`
4. **Standard error**: `SE = σ / √n`, where n is the number of words

The error bars represent the **standard error of the mean**, indicating the precision of the overlap estimate. Smaller error bars indicate more consistent overlap across words, while larger error bars indicate greater variability.

## Quick Start

```bash
# Basic usage with default settings (2 runs, all tranches)
python scripts/plot_neighbor_overlap.py

# Fast analysis (recommended for initial exploration)
python scripts/plot_neighbor_overlap.py --mode fast

# Early tranches only
python scripts/plot_neighbor_overlap.py --mode early

# Late tranches analysis
python scripts/plot_neighbor_overlap.py --mode late
```

## Analysis Modes

| Mode | Tranches | Word Sampling | Default Runs | Use Case |
|------|----------|---------------|--------------|----------|
| `full` | All (1+) | 100%        | 2            | Complete analysis |
| `fast` |Every 10th| 50%         |3             | Quick overview |
| `early`| 30-200   | 100%        | 3            | Early training dynamics |
| `late` | 950+     | 50%         | 3            | Late training stability |

## Command Line Arguments

### Mode Selection

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `full` | Preset mode: `full`, `fast`, `early`, `late` |

### Run Paths

The script compares two curricula, each with up to 5 training runs. The argument names (`--aoa_run*` and `--shuffled_run*`) are legacy but can be used for any curriculum types:

| Argument | Default | Description |
|----------|---------|-------------|
| `--aoa_run1` | `outputs/embeddings/aoa_50d_0` | First curriculum run 1 |
| `--aoa_run2` | `outputs/embeddings/aoa_50d_1` | First curriculum run 2 |
| `--aoa_run3` | `outputs/embeddings/aoa_50d_2` | First curriculum run 3 |
| `--aoa_run4` | `outputs/embeddings/aoa_50d_3` | First curriculum run 4 |
| `--aoa_run5` | `outputs/embeddings/aoa_50d_4` | First curriculum run 5 |
| `--shuffled_run1` | `outputs/embeddings/shuffled_50d_0` | Second curriculum run 1 |
| `--shuffled_run2` | `outputs/embeddings/shuffled_50d_1` | Second curriculum run 2 |
| `--shuffled_run3` | `outputs/embeddings/shuffled_50d_2` | Second curriculum run 3 |
| `--shuffled_run4` | `outputs/embeddings/shuffled_50d_3` | Second curriculum run 4 |
| `--shuffled_run5` | `outputs/embeddings/shuffled_50d_4` | Second curriculum run 5 |

### Curriculum Labels

The script automatically extracts curriculum names from run paths, but you can override them:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--curriculum1_name` | str | auto-detect | Display name for first curriculum (e.g., "AoA", "Frequency", "Concreteness") |
| `--curriculum2_name` | str | auto-detect | Display name for second curriculum (e.g., "Shuffled", "Random") |

**Auto-detection**: The script extracts curriculum names from directory paths:
- `outputs/embeddings/aoa_50d_0` → "AoA"
- `outputs/embeddings/freq_50d_0` → "Frequency"
- `outputs/embeddings/conc_50d_0` → "Concreteness"
- `outputs/embeddings/shuffled_50d_0` → "Shuffled"

### Tranche Type Specification

Specify the tranche type for each curriculum to display it in the plot title:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--curriculum1_tranche_type` | str | None | Tranche type: `word-based`, `sentence-based`, `word-count`, or `matching` |
| `--curriculum2_tranche_type` | str | None | Tranche type: `word-based`, `sentence-based`, `word-count`, or `matching` |

**Tranche Type Meanings**:
- `word-based`: Unique word-based tranches (each tranche introduces N new vocabulary words)
- `sentence-based`: Sentence-based tranches (each tranche contains N sentences)
- `word-count`: Word-count tranches (each tranche contains N total word tokens)
- `matching`: Matching tranches (matches word counts from a reference curriculum)

When specified, the plot title will include tranche type information (e.g., "Unique Word-Based Tranches").

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

### Custom Paths and Curriculum Names

```bash
# Use custom embedding directories (names auto-detected from paths)
python scripts/plot_neighbor_overlap.py \
    --aoa_run1 path/to/freq/run1 \
    --aoa_run2 path/to/freq/run2 \
    --shuffled_run1 path/to/shuffled/run1 \
    --shuffled_run2 path/to/shuffled/run2 \
    --num_runs 2

# Explicitly specify curriculum names
python scripts/plot_neighbor_overlap.py \
    --aoa_run1 outputs/embeddings/freq_50d_0 \
    --aoa_run2 outputs/embeddings/freq_50d_1 \
    --shuffled_run1 outputs/embeddings/shuffled_50d_0 \
    --shuffled_run2 outputs/embeddings/shuffled_50d_1 \
    --curriculum1_name Frequency \
    --curriculum2_name Shuffled \
    --num_runs 2

# Custom output path
python scripts/plot_neighbor_overlap.py \
    --mode fast \
    --output outputs/figures/my_analysis.png
```

### Comparing Different Curricula

```bash
# Frequency vs Shuffled (not AoA)
python scripts/plot_neighbor_overlap.py \
    --aoa_run1 outputs/embeddings/freq_50d_0 \
    --aoa_run2 outputs/embeddings/freq_50d_1 \
    --shuffled_run1 outputs/embeddings/shuffled_50d_0 \
    --shuffled_run2 outputs/embeddings/shuffled_50d_1 \
    --curriculum1_name Frequency \
    --curriculum2_name Shuffled \
    --word_sample_frac 0.1 \
    --sample_every 10 \
    --num_runs 2

# Concreteness vs Frequency
python scripts/plot_neighbor_overlap.py \
    --aoa_run1 outputs/embeddings/conc_50d_0 \
    --aoa_run2 outputs/embeddings/conc_50d_1 \
    --shuffled_run1 outputs/embeddings/freq_50d_0 \
    --shuffled_run2 outputs/embeddings/freq_50d_1 \
    --curriculum1_name Concreteness \
    --curriculum2_name Frequency \
    --num_runs 2
```

### Specifying Tranche Types

```bash
# Include tranche type in plot title (word-based)
python scripts/plot_neighbor_overlap.py \
    --aoa_run1 outputs/embeddings/freq_50d_0 \
    --aoa_run2 outputs/embeddings/freq_50d_1 \
    --shuffled_run1 outputs/embeddings/shuffled_50d_0 \
    --shuffled_run2 outputs/embeddings/shuffled_50d_1 \
    --curriculum1_name Frequency \
    --curriculum2_name Shuffled \
    --curriculum1_tranche_type word-based \
    --curriculum2_tranche_type word-based \
    --num_runs 2

# Different tranche types
python scripts/plot_neighbor_overlap.py \
    --aoa_run1 outputs/embeddings/aoa_50d_0 \
    --aoa_run2 outputs/embeddings/aoa_50d_1 \
    --shuffled_run1 outputs/embeddings/shuffled_50d_0 \
    --shuffled_run2 outputs/embeddings/shuffled_50d_1 \
    --curriculum1_tranche_type word-based \
    --curriculum2_tranche_type sentence-based \
    --num_runs 2
```

## Output

The script generates:

1. **Plot** (default: `outputs/figures/{curriculum1}_vs_{curriculum2}_overlap{suffix}.png`):
   - Two lines showing overlap over training tranches with error bars
   - Blue: First curriculum (default: AoA)
   - Red: Second curriculum (default: Shuffled)
   - **Error bars**: Standard error of the mean overlap (shows variability across words)
   - Dashed lines: Mean overlap for each curriculum
   - **Title includes tranche type** when specified (e.g., "Unique Word-Based Tranches")
   - Title format: `"Embedding Stability: {Curriculum1} vs {Curriculum2} Curriculum ({TrancheType})"`
   - Error bars are automatically spaced when there are many data points to avoid clutter

2. **Console Statistics**:
   - Mean, min, max overlap for each curriculum
   - Number of tranches processed
   - Tranche type information (if specified)

## Interpreting Results

- **Higher overlap** = More stable embeddings
- **Error bars**: Smaller error bars indicate more consistent overlap across words; larger error bars indicate greater variability
- **Non-overlapping error bars**: When error bars don't overlap between curricula, it suggests a statistically meaningful difference
- **Curriculum comparison**: Compare any two curricula (e.g., AoA vs Shuffled, Frequency vs Concreteness)
- **Tranche type matters**: Different tranche types may show different stability patterns
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


