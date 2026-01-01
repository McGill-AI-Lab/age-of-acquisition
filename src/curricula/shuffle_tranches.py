from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import os

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from curricula.path_utils import resolve_curriculum_path

# shuffles rows within each tranche for the curriculum at idx
# returns the seed (int) used to seed the shuffle
def shuffle_tranches(idx: str) -> int:
  curriculum_root = resolve_curriculum_path(idx)
  tranches_dir = _get_tranches_dir(curriculum_root)

  if not tranches_dir.is_dir():
    raise FileNotFoundError(f"Could not find tranches directory at: {tranches_dir}")

  tranche_dirs = _find_tranche_dirs(tranches_dir)

  # Random seed for the whole operation
  seed = int.from_bytes(os.urandom(8), "big", signed=False)

  rng = _make_rng(seed)

  pbar = tqdm(tranche_dirs, desc="Shuffling tranches", unit="tranche")
  for tdir in pbar:
    data_path = tdir / "data.parquet"
    if not data_path.exists():
      continue
    n = _shuffle_parquet_in_place(data_path, rng=rng)
    pbar.set_postfix(rows=n, refresh=False)

  return seed

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

def _find_tranche_dirs(tranches_dir: Path) -> List[Path]:
  tranche_dirs = [p for p in Path(tranches_dir).iterdir() if p.is_dir() and p.name.startswith("tranche_")]
  tranche_dirs.sort(key=lambda p: _parse_tranche_index(p.name))
  return tranche_dirs

def _make_rng(seed: int):
  """
  Returns an RNG object with a shuffle/permutation method.
  Prefers numpy Generator; falls back to random.Random.
  """
  try:
    import numpy as np
    return np.random.default_rng(seed)
  except Exception:
    import random
    r = random.Random(seed)
    return r

def _permutation(n: int, rng):
  """
  Create a permutation of [0..n-1] using either numpy or random.
  """
  try:
    # numpy Generator
    return rng.permutation(n)
  except Exception:
    # random.Random
    import random
    idxs = list(range(n))
    rng.shuffle(idxs)
    return idxs

def _shuffle_parquet_in_place(parquet_path: Path, rng) -> int:
  """
  Reads the parquet at `parquet_path`, shuffles its rows, and overwrites it atomically.
  """
  parquet_path = Path(parquet_path)

  # Read entire tranche table (tranches are expected to be manageable)
  table = pq.read_table(parquet_path.as_posix())

  n = table.num_rows
  if n <= 1:
    return n

  perm = _permutation(n, rng)

  # pyarrow take expects an Array of indices
  if not isinstance(perm, pa.Array):
    # numpy array or python list -> arrow array
    perm = pa.array(perm, type=pa.int64())

  shuffled = table.take(perm)

  # Atomic replace: write temp then replace
  tmp_path = parquet_path.with_suffix(".parquet.tmp")
  pq.write_table(shuffled, tmp_path.as_posix())
  os.replace(tmp_path.as_posix(), parquet_path.as_posix())
  return n

def main():
  idx = 0
  seed = shuffle_tranches(idx)
  print(f"Shuffled tranches for curriculum {idx} (seed={seed}).")

if __name__ == "__main__":
  main()