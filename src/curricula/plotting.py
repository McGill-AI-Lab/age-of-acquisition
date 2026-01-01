# src/curricula/analyze.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from curricula.path_utils import resolve_curriculum_path


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


def plot_tranche_sizes(
    curriculum_path: str | Path,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Plots tranche index vs tranche size (number of rows in tranche parquet).

    Args:
      curriculum_path: path to curriculum root / tranches dir, OR an index like "7" (resolved under training/)
      save_path: optional image output path (e.g. ".../tranche_sizes.png")
      show: whether to display the plot window

    Returns:
      (tranche_indices, tranche_row_counts)
    """
    curriculum_path = Path(curriculum_path)
    tranche_dirs = _find_tranche_dirs(curriculum_path)

    xs: List[int] = []
    ys: List[int] = []

    missing = []
    for tdir in tranche_dirs:
        data_path = tdir / "data.parquet"
        if not data_path.exists():
            missing.append(data_path)
            continue
        idx = _parse_tranche_index(tdir.name)
        n = _parquet_num_rows(data_path)
        xs.append(idx)
        ys.append(n)

    if missing:
        # Not fatal, but useful warning
        print(f"Warning: {len(missing)} tranche(s) missing data.parquet (skipped). Example: {missing[0]}")

    if not xs:
        raise RuntimeError(f"No tranche data.parquet files found under: {curriculum_path}")

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Tranche")
    plt.ylabel("Number of rows")
    plt.title("Tranche size (rows) over curriculum")
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
    plot_tranche_sizes("0")


if __name__ == "__main__":
    main()
