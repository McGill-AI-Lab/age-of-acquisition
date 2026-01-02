from __future__ import annotations

from pathlib import Path
from typing import Iterator, Dict, Any, List, Set, Optional

import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from curricula.simple_tokenizer import tokenize_words_lower_already

def _iter_ordered_rows(ordered_parquet: Path) -> Iterator[Dict[str, Any]]:
  dataset = ds.dataset([Path(ordered_parquet).as_posix()], format="parquet")
  scanner = dataset.scanner(columns=["sentence", "value"], batch_size=200_000)
  for batch in scanner.to_batches():
    for r in batch.to_pylist():
      yield r

def write_sentence_based_tranches(
  ordered_parquet: Path,
  out_dir: Path,
  tranche_size: int,
) -> int:
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  tranche_idx = 1
  buffer: List[Dict[str, Any]] = []

  def flush_tranche(rows: List[Dict[str, Any]], idx: int) -> None:
    tdir = out_dir / f"tranche_{idx:04d}"
    tdir.mkdir(parents=True, exist_ok=False)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, (tdir / "data.parquet").as_posix())

  pbar = tqdm(unit="rows", desc="Tranching (sentence-based)")
  for row in _iter_ordered_rows(ordered_parquet):
    buffer.append(row)
    pbar.update(1)
    if len(buffer) >= tranche_size:
      flush_tranche(buffer, tranche_idx)
      tranche_idx += 1
      buffer = []
  
  if buffer:
    flush_tranche(buffer, tranche_idx)
    tranche_idx += 1
  
  pbar.close()
  return tranche_idx - 1

def write_word_based_tranches(
  ordered_parquet: Path,
  out_dir: Path,
  tranche_size: int,
) -> int:
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  seen_words: Set[str] = set()
  new_words_current: Set[str] = set()
  rows_current: List[Dict[str, Any]] = []

  tranche_idx = 1

  def flush_tranche(idx: int) -> None:
    nonlocal seen_words, new_words_current, rows_current

    tdir = out_dir / f"tranche_{idx:04d}"
    tdir.mkdir(parents=True, exist_ok=False)

    # write parquet
    table = pa.Table.from_pylist(rows_current)
    pq.write_table(table, (tdir / "data.parquet").as_posix())

    # write new words list
    words_path = tdir / "new_words.txt"
    with words_path.open("w", encoding="utf-8") as f:
      for w in sorted(new_words_current):
        f.write(w + "\n")

    seen_words |= new_words_current
    new_words_current = set()
    rows_current = []
    pbar.set_postfix(tranche=idx, refresh=False)

  it = _iter_ordered_rows(ordered_parquet)
  pbar = tqdm(unit="rows", desc="Tranching (word-based)")
  pending: Optional[Dict[str, Any]] = None

  while True:
    row = pending if pending is not None else next(it, None)
    pending = None
    if row is None:
      break
    pbar.update(1)

    sentence = row["sentence"]
    words = set(tokenize_words_lower_already(sentence))

    introduced = {w for w in words if (w not in seen_words and w not in new_words_current)}
    would_exceed = (len(new_words_current) + len(introduced) > tranche_size)

    if would_exceed and rows_current:
      # close current tranche before adding this sentence
      flush_tranche(tranche_idx)
      tranche_idx += 1

      # reprocess this sentence in the next tranche
      pending = row
      continue

    if would_exceed and not rows_current:
      # case that there is a single sentence in a tranche
      rows_current.append(row)
      new_words_current |= introduced
      flush_tranche(tranche_idx)
      tranche_idx += 1
      continue

    # normal add
    rows_current.append(row)
    new_words_current |= introduced

    # close tranche if we hit exactly tranche_size new words
    if len(new_words_current) >= tranche_size:
      flush_tranche(tranche_idx)
      tranche_idx += 1
    
  if rows_current:
    flush_tranche(tranche_idx)
    tranche_idx += 1

  pbar.close() 
  return tranche_idx - 1