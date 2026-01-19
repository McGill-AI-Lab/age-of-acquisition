# curricula/word_counts.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Union, Any
import json

import pyarrow.dataset as ds
from tqdm import tqdm

from curricula.simple_tokenizer import tokenize_words_lower_already
from curricula.path_utils import resolve_curriculum_path, get_tranches_dir, find_tranche_dirs


def _iter_sentences_in_parquet(parquet_path: Path, text_col: str = "sentence", batch_size: int = 200_000):
  """
  Streams sentences from a parquet column in batches (memory-stable).
  """
  dataset = ds.dataset([parquet_path.as_posix()], format="parquet")
  scanner = dataset.scanner(columns=[text_col], batch_size=batch_size)

  for batch in scanner.to_batches():
    col = batch.column(0)
    for v in col.to_pylist():
      if v is None:
        continue
      s = str(v)
      if s:
        yield s


def write_unique_word_counts(curriculum_path: str | Path) -> Path:
  """
  Creates <curriculum_root>/unique_word_counts.json.

  The JSON records the number of *new* unique words introduced by each tranche,
  in tranche order, relative to all previous tranches.

  Args:
    curriculum_path: curriculum root path, tranches path, or an index like "7"

  Returns:
    Path to the generated unique_word_counts.json
  """
  curriculum_root = resolve_curriculum_path(curriculum_path)
  tranches_dir = get_tranches_dir(curriculum_root)

  if not tranches_dir.is_dir():
    raise FileNotFoundError(f"Could not find tranches directory at: {tranches_dir}")

  tranche_dirs = find_tranche_dirs(tranches_dir)
  if not tranche_dirs:
    raise RuntimeError(f"No tranche_* directories found under: {tranches_dir}")

  seen: Set[str] = set()
  per_tranche: List[Dict[str, Any]] = []

  pbar = tqdm(tranche_dirs, desc="Computing unique word introductions", unit="tranche")
  for tdir in pbar:
    dp = tdir / "data.parquet"
    if not dp.exists():
      raise FileNotFoundError(f"Missing tranche parquet: {dp}")

    introduced: Set[str] = set()

    # NOTE: Mirrors your tranche-building behavior: uses tokenize_words_lower_already directly.
    for sent in _iter_sentences_in_parquet(dp, text_col="sentence"):
      for w in tokenize_words_lower_already(sent):
        if w not in seen:
          introduced.add(w)

    # Update global seen after finishing tranche
    seen |= introduced

    # tranche_0001 -> 1
    try:
      tranche_index = int(tdir.name.split("_", 1)[1])
    except Exception:
      tranche_index = None

    per_tranche.append(
      {
        "tranche_dir": tdir.name,
        "tranche_index": tranche_index,
        "new_unique_words": len(introduced),
        "cumulative_unique_words": len(seen),
      }
    )

    pbar.set_postfix(new=len(introduced), cumulative=len(seen), refresh=False)

  out_path = curriculum_root / "unique_word_counts.json"
  payload = {
    "curriculum_root": str(curriculum_root),
    "num_tranches": len(per_tranche),
    "per_tranche": per_tranche,
  }

  out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
  return out_path


def main():
  out = write_unique_word_counts("0")
  print(f"Wrote {out}")


if __name__ == "__main__":
  main()
