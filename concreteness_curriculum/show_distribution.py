# show_distribution.py
from __future__ import annotations

"""
Build and save a histogram of concreteness scores across all scored shards.

- Reads *.parquet from INPUT_DIR
- Uses the first available column from SCORE_COLUMNS (compat with older files)
- Bins scores into N_BINS between SCORE_MIN..SCORE_MAX
- Plots % of sentences per bin (optionally adds an 'Unknown (-1)' bar)
"""

# --------------------------- Parameters --------------------------------------
INPUT_DIR = "concreteness_curriculum/data/outputs/ordered_data"  # folder containing per-shard parquet files
OUTPUT_PATH = "concreteness_curriculum/plots/score_distribution-MEAN-REMOVE_STOP.png"
PARAMETER_INFO = "method=mean, lemmatize=false, remove_stop=true, n_parquets=all (refined bookcorpus)"

# Histogram settings
N_BINS = 10
SCORE_MIN = 1.0
SCORE_MAX = 5.0

# Whether to append an extra bar for unknowns (-1 or NaN)
INCLUDE_UNKNOWN_BAR = True

# Column compatibility (new -> old)
SCORE_COLUMNS = ("concreteness_score", "concreteness_scores")

# Figure settings
FIGSIZE = (10, 5)
DPI = 180
TITLE = f"Concreteness score distribution - {PARAMETER_INFO}"
X_LABEL = "Concreteness bins"
Y_LABEL = "% of sentences"
# -----------------------------------------------------------------------------

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_scores_one(file: Path) -> tuple[np.ndarray, int]:
  """
  Read scores from one parquet.
  Returns (valid_scores_array, unknown_count).
  Unknowns are values < 0 or NaN.
  """
  s = None
  for col in SCORE_COLUMNS:
    try:
      s = pd.read_parquet(file, columns=[col], engine="pyarrow")[col].to_numpy()
      break
    except Exception:
      continue
  if s is None:
    raise RuntimeError(f"{file} does not contain any of {SCORE_COLUMNS}")

  is_unknown = (s < 0) | np.isnan(s)
  unknown_count = int(is_unknown.sum())
  valid = s[~is_unknown].astype(np.float64, copy=False)
  return valid, unknown_count


def accumulate_histogram(files: list[Path]) -> tuple[np.ndarray, int, int, np.ndarray]:
  """
  Accumulate histogram counts across files.
  Returns (hist_counts, total_valid, total_unknown, bin_edges)
  """
  edges = np.linspace(SCORE_MIN, SCORE_MAX, N_BINS + 1, dtype=np.float64)
  hist = np.zeros(N_BINS, dtype=np.int64)
  total_unknown = 0
  total_valid = 0

  for f in files:
    valid, unk = read_scores_one(f)
    total_unknown += unk
    if valid.size:
      h, _ = np.histogram(valid, bins=edges)
      hist += h
      total_valid += int(valid.size)

  return hist, total_valid, total_unknown, edges


def format_bin_labels(edges: np.ndarray) -> list[str]:
  return [f"{a:.1f}–{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]


def main() -> None:
  in_dir = Path(INPUT_DIR)
  if not in_dir.exists():
    print(f"Input directory not found: {in_dir}", file=sys.stderr)
    sys.exit(1)

  files = sorted(in_dir.glob("*.parquet"))
  if not files:
    print(f"No parquet files found in: {in_dir}", file=sys.stderr)
    sys.exit(1)

  hist, total_valid, total_unknown, edges = accumulate_histogram(files)
  if total_valid == 0:
    print("No valid (>=0) scores found.", file=sys.stderr)
    sys.exit(1)

  pct = (hist / total_valid) * 100.0
  labels = format_bin_labels(edges)
  x = np.arange(len(labels), dtype=np.int64)

  fig, ax = plt.subplots(figsize=FIGSIZE)
  ax.bar(x, pct)
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.set_ylabel(Y_LABEL)
  ax.set_xlabel(X_LABEL)
  ax.set_title(TITLE)
  ax.grid(True, axis="y", linestyle="--", alpha=0.5)

  if INCLUDE_UNKNOWN_BAR:
    unknown_pct = (total_unknown / (total_valid + total_unknown)) * 100.0
    ax.bar(len(x), unknown_pct)
    ax.set_xticks(np.append(x, len(x)))
    ax.set_xticklabels([*labels, "Unknown (-1)"])

  out_path = Path(OUTPUT_PATH)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  plt.tight_layout()
  fig.savefig(out_path, dpi=DPI)
  print(f"Saved plot to: {out_path}")

  coverage = 100.0 * total_valid / (total_valid + total_unknown)
  print(
    f"Files: {len(files)} | Valid sentences: {total_valid:,} | "
    f"Unknown: {total_unknown:,} ({100.0 - coverage:.2f}% of total)"
  )


if __name__ == "__main__":
  main()
