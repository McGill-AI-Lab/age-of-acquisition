# Embedding Evaluation Scripts (AoA & Concreteness)

These scripts benchmark word embeddings by predicting lexical norms: **Age of Acquisition (AoA)** or **Concreteness**. They use the same pipeline (Ridge regression, LOO and 10-fold CV) and differ only in the norm table and reported units.

---

## Overview

| Script        | Norm predicted   | Lookup table                                      | Output plot                 |
|---------------|------------------|---------------------------------------------------|-----------------------------|
| `eval_aoa.py` | Age of Acquisition | `data/processed/lookup_tables/aoa_table.parquet`  | `outputs/plots/aoa_correlation.png`  |
| `eval_conc.py`| Concreteness     | `data/processed/lookup_tables/conc_table.parquet`  | `outputs/plots/conc_correlation.png` |

Both scripts:

1. Load embeddings from a directory of `tranche_*.parquet` files.
2. Load norm ratings (word + value) from the parquet lookup table.
3. Merge on `word` (lowercased) and keep only overlapping words.
4. Fit **Ridge regression** (embedding → norm).
5. Report **Leave-One-Out (LOO)** Mean Absolute Deviation.
6. Report **10-fold CV** Spearman rank correlation (mean ± std).
7. Save a **scatter plot** (predicted vs actual) for the best fold.

---

## Usage

Run from the **project root** (the `age-of-acquisition` directory).

### AoA evaluation

```bash
# Default embeddings: outputs/embeddings/shuffled_word_40k
python scripts/eval_aoa.py

# Custom embeddings (e.g. AoA-trained or concreteness-trained)
python scripts/eval_aoa.py --embeddings-dir outputs/embeddings/aoa_word_40k
python scripts/eval_aoa.py --embeddings-dir outputs/embeddings/conc_word_40k
```

### Concreteness evaluation

```bash
# Default embeddings
python scripts/eval_conc.py

# Custom embeddings
python scripts/eval_conc.py --embeddings-dir outputs/embeddings/conc_word_40k
python scripts/eval_conc.py --embeddings-dir outputs/embeddings/aoa_word_40k
```

### Shared option

- **`--embeddings-dir`**  
  Path to the folder containing `tranche_*.parquet` files (e.g. `outputs/embeddings/aoa_word_40k`).  
  Can be relative to the project root or absolute.  
  Default: `outputs/embeddings/shuffled_word_40k`.

---

## Inputs

### Embeddings

- **Location:** Directory given by `--embeddings-dir` (default: `outputs/embeddings/shuffled_word_40k`).
- **Format:** One or more Parquet files matching `tranche_*.parquet`, each with columns:
  - `word` (string)
  - `embedding` (list of floats)
- Words are deduplicated (first occurrence across tranches is kept).

### Norm tables

- **AoA:** `data/processed/lookup_tables/aoa_table.parquet` — columns `word`, `value` (AoA in **years**; Kuperman-style).
- **Concreteness:** `data/processed/lookup_tables/conc_table.parquet` — columns `word`, `value` (concreteness rating).

Only words present in **both** the embedding set and the norm table are used; the script prints the overlap count.

---

## Outputs

### Console

- Number of words in embeddings and in the norm table.
- **Overlapping words** used for evaluation.
- **Ridge (alpha=1.0): Mean Absolute Deviation (LOO)**  
  - **AoA:** in **months** (|predicted − actual| in years × 12).  
  - **Concreteness:** in the same units as the rating (no conversion).
- **10-fold CV Spearman rank correlation:** mean ± standard deviation.
- **Best fold** index and its Spearman ρ.
- Path where the plot was saved.

### Plot

- **AoA:** `outputs/plots/aoa_correlation.png` — predicted vs actual AoA (best fold).
- **Concreteness:** `outputs/plots/conc_correlation.png` — predicted vs actual concreteness (best fold).

Each plot includes a diagonal line (y = x) and the best-fold Spearman ρ in the title.

---

## Interpreting the metrics

- **Spearman correlation:** Measures how well the **rank order** of predicted norms matches the true order. Higher is better; relevant when you care about which words are earlier/later (AoA) or more/less concrete.
- **MAD (LOO):** Average absolute error in the same units as the norm (months for AoA, rating units for concreteness). Lower is better for raw accuracy.

Comparing the same metric across different `--embeddings-dir` (e.g. shuffled vs AoA vs concreteness curricula) shows how well each embedding set predicts that norm.
