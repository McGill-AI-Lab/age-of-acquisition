# find_bin_examples_all_bins.py
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------- Parameters --------------------
SCORES_DIR = Path("concreteness_curriculum/data/outputs/ordered_data")
CORPORA_DIR = Path("concreteness_curriculum/data/corpora")  # {dataset}/{dataset}_tokens_part-PPP.parquet
TOKENS_COLUMN = "tokens"

SCORE_MIN = 1.0
SCORE_MAX = 5.0
N_BINS = 10
N_EXAMPLES_PER_BIN = 2

SCORE_COLUMNS = ("concreteness_score", "concreteness_scores")  # new -> old
# ----------------------------------------------------

def _find_score_col(file: Path) -> str:
    for col in SCORE_COLUMNS:
        try:
            pd.read_parquet(file, columns=[col], engine="pyarrow")
            return col
        except Exception:
            continue
    raise RuntimeError(f"{file} missing score column(s): {SCORE_COLUMNS}")

def _detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+'([sdmtrlve])\b", r"'\1", text)  # 's 'd 'm 't 'r 'l 've
    text = re.sub(r"\b(n)\s+'t\b", r"\1't", text)       # n't
    text = re.sub(r"\s+([.,!?;:%)\]\}])", r"\1", text)  # no space before punct
    text = re.sub(r"([(\[\{])\s+", r"\1", text)         # no space after open
    text = text.replace("`` ", "“").replace(" ''", "”")
    return text

def _load_sentence(dataset: str, part: str, row_idx: int) -> list[str] | None:
    src = CORPORA_DIR / dataset / f"{dataset}_tokens_part-{part}.parquet"
    if not src.exists():
        return None
    df = pd.read_parquet(src, columns=[TOKENS_COLUMN], engine="pyarrow")
    if not (0 <= row_idx < len(df)):
        return None
    tokens = df.iloc[row_idx][TOKENS_COLUMN]
    return list(tokens)

def _bin_labels(edges: np.ndarray) -> list[str]:
    return [f"{a:.1f}–{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]

def _collect_examples_all_bins() -> list[list[tuple]]:
    """
    Returns a list of length N_BINS; each item is a list of examples:
    (id, dataset, score, text)
    """
    edges = np.linspace(SCORE_MIN, SCORE_MAX, N_BINS + 1, dtype=np.float64)
    examples: list[list[tuple]] = [[] for _ in range(N_BINS)]

    files = sorted(SCORES_DIR.glob("*.parquet"))
    for f in files:
        # Stop early if all bins are filled
        if all(len(examples[b]) >= N_EXAMPLES_PER_BIN for b in range(N_BINS)):
            break

        score_col = _find_score_col(f)
        df = pd.read_parquet(f, columns=[score_col, "dataset"], engine="pyarrow")
        s = df[score_col].to_numpy()
        # ignore unknowns (-1 or NaN)
        valid_mask = (s >= SCORE_MIN) & (s <= SCORE_MAX) & np.isfinite(s)
        if not valid_mask.any():
            continue
        dfv = df.loc[valid_mask]

        # Try to fill each bin from this file
        for b in range(N_BINS):
            need = N_EXAMPLES_PER_BIN - len(examples[b])
            if need <= 0:
                continue
            lo, hi = edges[b], edges[b + 1]
            if b < N_BINS - 1:
                mask = (dfv[score_col] >= lo) & (dfv[score_col] < hi)
            else:
                mask = (dfv[score_col] >= lo) & (dfv[score_col] <= hi)  # include top edge
            if not mask.any():
                continue

            picks = dfv[mask].head(need)
            for idx, row in picks.iterrows():
                # id format: "PPP-<rowIndex>"
                try:
                    part, row_idx = str(idx).split("-", 1)
                    row_idx = int(row_idx)
                except Exception:
                    continue
                dataset = row["dataset"]
                tokens = _load_sentence(dataset, part, row_idx)
                if not tokens:
                    continue
                text = _detokenize(tokens)
                examples[b].append((str(idx), dataset, float(row[score_col]), text))

                if len(examples[b]) >= N_EXAMPLES_PER_BIN:
                    break

    return examples

def main():
    ex_by_bin = _collect_examples_all_bins()
    edges = np.linspace(SCORE_MIN, SCORE_MAX, N_BINS + 1, dtype=np.float64)
    labels = _bin_labels(edges)

    for b, label in enumerate(labels):
        print(f"\nBin {b+1:02d} [{label}]:")
        if not ex_by_bin[b]:
            print("  (none found)")
            continue
        for i, (idx, dataset, score, text) in enumerate(ex_by_bin[b], 1):
            print(f"{i}. [{dataset} | id={idx} | score={score:.3f}]")
            print(f"   {text}")

if __name__ == "__main__":
    main()
