from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from tqdm import tqdm

from curricula.path_utils import resolve_curriculum_path
from curricula.simple_tokenizer import tokenize_words_lower_already


def _find_tranche_dirs(curriculum_path: Path) -> List[Path]:
    """
    Accepts either:
      - curriculum root (contains 'tranches/')
      - or the tranches directory itself
    Returns tranche directories sorted by tranche index.
    """
    curriculum_path = resolve_curriculum_path(curriculum_path)

    # If user passed the curriculum root, look for tranches/
    if (curriculum_path / "tranches").is_dir():
        tranches_dir = curriculum_path / "tranches"
    else:
        tranches_dir = curriculum_path

    if not tranches_dir.is_dir():
        raise FileNotFoundError(f"Could not find tranches directory at: {tranches_dir}")

    tranche_dirs = [p for p in tranches_dir.iterdir() if p.is_dir() and p.name.startswith("tranche_")]
    tranche_dirs.sort(key=lambda p: _parse_tranche_index(p.name))
    return tranche_dirs


def _parse_tranche_index(name: str) -> int:
    # expects tranche_0001
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return 10**9


def _parquet_num_rows(parquet_path: Path) -> int:
    """
    Reads row count from parquet metadata (fast, no full load).
    """
    pf = pq.ParquetFile(parquet_path.as_posix())
    md = pf.metadata
    return int(md.num_rows) if md is not None else 0

def _parquet_num_words(
    parquet_path: Path,
    text_column: str = "sentence",
    batch_size: int = 4096,
) -> int:
    """
    Counts total words in a parquet by streaming over the text column and tokenizing.

    NOTE: This must read the data (slower than metadata row counts).
    """
    total_words = 0
    pf = pq.ParquetFile(parquet_path.as_posix())

    # Stream batches to avoid loading whole file
    for batch in pf.iter_batches(columns=[text_column], batch_size=batch_size):
        col = batch.column(0)
        # Convert to Python strings safely; handle nulls
        for v in col.to_pylist():
            if not v:
                continue
            total_words += len(tokenize_words_lower_already(v))

    return total_words

def plot_tranche_sizes(
    curriculum_path: str | Path,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    metric: Literal["sentence", "word"] = "sentence",
    text_column: str = "sentence",
    batch_size: int = 4096,
) -> Tuple[List[int], List[int]]:
    """
    Plots tranche index vs tranche size.

    metric:
      - "sentence": y-axis is number of rows (sentences) per tranche (fast; metadata)
      - "word": y-axis is number of words per tranche (streams parquet; tokenizes)

    Args:
      curriculum_path: path to curriculum root / tranches dir, OR an index like "7" (resolved under training/)
      save_path: optional image output path (e.g. ".../tranche_sizes.png")
      show: whether to display the plot window
      metric: "sentence" or "word"
      text_column: parquet column name containing the sentence text (used for metric="word")
      batch_size: batches for streaming word counts

    Returns:
      (tranche_indices, tranche_sizes)
    """
    if metric not in {"sentence", "word"}:
        raise ValueError(f"metric must be 'sentence' or 'word', got: {metric}")

    curriculum_path = Path(curriculum_path)
    tranche_dirs = _find_tranche_dirs(curriculum_path)

    xs: List[int] = []
    ys: List[int] = []

    missing = []
    for tdir in tqdm(tranche_dirs, desc=f"Processing tranches ({metric}) for plotting"):
        data_path = tdir / "data.parquet"
        if not data_path.exists():
            missing.append(data_path)
            continue

        idx = _parse_tranche_index(tdir.name)

        if metric == "sentence":
            n = _parquet_num_rows(data_path)
        else:
            n = _parquet_num_words(data_path, text_column=text_column, batch_size=batch_size)

        xs.append(idx)
        ys.append(n)

    if missing:
        print(f"Warning: {len(missing)} tranche(s) missing data.parquet (skipped). Example: {missing[0]}")

    if not xs:
        raise RuntimeError(f"No tranche data.parquet files found under: {curriculum_path}")

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Tranche")

    if metric == "sentence":
        plt.ylabel("Number of sentences (rows)")
        plt.title("Tranche size (sentences) over curriculum")
    else:
        plt.ylabel("Number of words")
        plt.title("Tranche size (words) over curriculum")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path.as_posix(), dpi=200)

    if show:
        plt.show()
    else:
        plt.close()

    return xs, ys


def main():
    plot_tranche_sizes("0", metric="sentence")
    plot_tranche_sizes("0", metric="word")


if __name__ == "__main__":
    main()
