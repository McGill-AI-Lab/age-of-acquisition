"""
Benchmark word embeddings by predicting Concreteness of words.

Uses Ridge regression with 10-fold CV; reports Spearman rank correlation
between predicted and actual concreteness (rank order matters).
Also reports Mean Absolute Deviation (MAD) via LOO cross-validation.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "outputs" / "embeddings" / "shuffled_word_40k"
CONC_PATH = PROJECT_ROOT / "data" / "processed" / "lookup_tables" / "conc_table.parquet"
OUTPUT_PLOT = PROJECT_ROOT / "outputs" / "plots" / "conc_correlation.png"


def load_embeddings(embeddings_dir: Path) -> pd.DataFrame:
    """Load all tranche_*.parquet files and concatenate into one dataframe."""
    files = sorted(embeddings_dir.glob("tranche_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No tranche_*.parquet files in {embeddings_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate by word (keep first occurrence across tranches)
    combined = combined.drop_duplicates(subset=["word"], keep="first").reset_index(drop=True)
    return combined


def load_conc(path: Path) -> pd.DataFrame:
    """Load concreteness norms; expect columns 'word' and 'value'."""
    df = pd.read_parquet(path, columns=["word", "value"])
    if "word" not in df.columns or "value" not in df.columns:
        raise KeyError(f"Concreteness parquet must have 'word' and 'value'; got {list(df.columns)}")
    df["word"] = df["word"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["word", "value"])
    return df


def loo_ridge_predict(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Leave-one-out predictions for Ridge regression using the hat-matrix shortcut.
    """
    X_c = X - np.mean(X, axis=0)
    y_mean = np.mean(y)
    y_c = y - y_mean
    n, d = X_c.shape
    M = np.linalg.solve(X_c.T @ X_c + alpha * np.eye(d), np.eye(d))
    # Hat diagonal for centered design: H_ii = (X_c @ M @ X_c.T)_ii
    A = X_c @ M  # (n, d)
    h_ii = np.sum(A * X_c, axis=1)  # (n,)
    # Fitted values (same as Ridge(alpha=alpha).fit(X,y).predict(X))
    y_pred_full = y_mean + (X_c @ M @ X_c.T) @ y_c
    # LOO: y_pred_LOO_i = (y_pred_i - h_ii * y_i) / (1 - h_ii)
    y_pred_loo = (y_pred_full - h_ii * y) / (1 - h_ii)
    return y_pred_loo


def main(embeddings_dir: Path) -> None:
    print("Loading embeddings...")
    emb_df = load_embeddings(embeddings_dir)
    print(f"  Embeddings: {len(emb_df)} words from {embeddings_dir.name}")

    print("Loading concreteness norms...")
    conc_df = load_conc(CONC_PATH)
    print(f"  Concreteness norms: {len(conc_df)} words")

    # Normalize embedding words for merge
    emb_df = emb_df.copy()
    emb_df["word"] = emb_df["word"].astype(str).str.strip().str.lower()

    # Data intersection: only words in BOTH
    merged = emb_df.merge(conc_df, on="word", how="inner")
    n_overlap = len(merged)
    print(f"\nOverlapping words (in both embeddings and concreteness): {n_overlap}")
    if n_overlap < 50:
        print("  WARNING: Low overlap may yield unstable correlations.")

    # Build X (embedding matrix) and y (concreteness rating)
    # Parquet stores embedding as list of floats; tolist() + np.array gives (n_words, dim)
    X = np.array(merged["embedding"].tolist(), dtype=np.float64)
    y = merged["value"].values.astype(np.float64)

    # LOO cross-validation: MAD in same units as concreteness rating
    alpha = 1.0
    y_pred_loo = loo_ridge_predict(X, y, alpha=alpha)
    mad = np.mean(np.abs(y_pred_loo - y))
    print(f"\nRidge (alpha={alpha}): Mean Absolute Deviation (LOO) = {mad:.4f}")

    # Ridge regression + 10-fold CV, Spearman per fold
    model = Ridge(alpha=1.0, random_state=42)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_spearman = []
    fold_results = []  # (fold_idx, y_true, y_pred) for best-fold plot

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rho, _ = spearmanr(y_test, y_pred)
        fold_spearman.append(rho)
        fold_results.append((fold, y_test, y_pred))

    mean_spearman = np.mean(fold_spearman)
    std_spearman = np.std(fold_spearman)
    print(f"\n10-fold CV Spearman rank correlation: {mean_spearman:.4f} +/- {std_spearman:.4f}")

    # Best fold = highest Spearman
    best_fold_idx = int(np.argmax(fold_spearman))
    best_fold, y_true_best, y_pred_best = fold_results[best_fold_idx]
    print(f"Best fold: {best_fold} (Spearman rho = {fold_spearman[best_fold_idx]:.4f})")

    # Scatter: predicted vs actual for best fold
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true_best, y_pred_best, alpha=0.6, s=20, c="steelblue", edgecolors="none")
    ax.set_xlabel("Actual concreteness")
    ax.set_ylabel("Predicted concreteness")
    ax.set_title(f"Ridge -> Concreteness (best fold {best_fold}, rho = {fold_spearman[best_fold_idx]:.3f})")
    min_val = min(y_true_best.min(), y_pred_best.min())
    max_val = max(y_true_best.max(), y_pred_best.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, label="y=x")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved scatter plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark word embeddings by predicting Concreteness.")
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default=None,
        help="Path to directory containing tranche_*.parquet embedding files (e.g. outputs/embeddings/conc_word_40k). Default: outputs/embeddings/shuffled_word_40k",
    )
    args = parser.parse_args()
    if args.embeddings_dir is None:
        emb_dir = DEFAULT_EMBEDDINGS_DIR
    else:
        p = Path(args.embeddings_dir)
        emb_dir = p if p.is_absolute() else (PROJECT_ROOT / p)
    main(emb_dir)
