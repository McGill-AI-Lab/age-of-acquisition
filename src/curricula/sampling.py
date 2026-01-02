from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import pyarrow.parquet as pq

from curricula.path_utils import resolve_curriculum_path


def _get_tranches_dir(curriculum_root_or_tranches: Path) -> Path:
  p = Path(curriculum_root_or_tranches)
  if (p / "tranches").is_dir():
    return p / "tranches"
  return p


def _parse_tranche_index(name: str) -> int:
  # tranche_0001 -> 1
  try:
    return int(name.split("_", 1)[1])
  except Exception:
    return 10**9


def _read_first_row_sentence_value(parquet_path: Path) -> Optional[Tuple[str, float]]:
  """
  Reads only the first row (sentence, value) from a tranche parquet.
  Returns None if file has 0 rows.
  """
  pf = pq.ParquetFile(parquet_path.as_posix())
  if pf.metadata is None or pf.metadata.num_rows == 0:
    return None

  # Read a tiny table: first row, only the needed columns
  tbl = pf.read(columns=["sentence", "value"], use_threads=True)
  if tbl.num_rows == 0:
    return None

  # Take first row
  sent = tbl["sentence"][0].as_py()
  val = tbl["value"][0].as_py()
  return ("" if sent is None else str(sent), float(val) if val is not None else float("nan"))


def write_samples(curriculum_path: str | Path) -> Path:
  """
  Creates <curriculum_root>/samples.txt.
  Each line contains: tranche_index, score, sentence (first row of tranche).

  Args:
    curriculum_path: curriculum root path, tranches path, or an index like "7"

  Returns:
    Path to the generated samples.txt
  """
  curriculum_root = resolve_curriculum_path(curriculum_path)
  tranches_dir = _get_tranches_dir(curriculum_root)

  if not tranches_dir.is_dir():
    raise FileNotFoundError(f"Could not find tranches directory at: {tranches_dir}")

  tranche_dirs = [p for p in tranches_dir.iterdir() if p.is_dir() and p.name.startswith("tranche_")]
  tranche_dirs.sort(key=lambda p: _parse_tranche_index(p.name))

  out_path = curriculum_root / "samples.txt"

  lines: List[str] = []
  for tdir in tranche_dirs:
    data_path = tdir / "data.parquet"
    if not data_path.exists():
      continue

    idx = _parse_tranche_index(tdir.name)
    first = _read_first_row_sentence_value(data_path)
    if first is None:
      continue

    sent, val = first
    # keep it single-line and readable
    sent_one_line = " ".join(sent.split())
    lines.append(f"{idx:04d}\t{val:.6f}\t{sent_one_line}")

  out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
  return out_path


def main():
  out = write_samples("0")
  print(f"Wrote {out}")


if __name__ == "__main__":
  main()
