from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Tuple
import heapq
import os
import shutil
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# external merge sort of a parquet file by 'key_column'
def external_sort_parquet(
    input_parquet: Path,
    output_parquet: Path,
    key_column: str,
    descending: bool,
    drop_columns: Optional[List[str]] = None,
    temp_dir: Optional[Path] = None,
    max_rows_in_memory: int = 500_000,
    read_batch_rows: int = 200_000,
) -> None:
  input_parquet = Path(input_parquet)
  output_parquet = Path(output_parquet)
  output_parquet.parent.mkdir(parents=True, exist_ok=True)

  if temp_dir is None:
    temp_dir = output_parquet.parent / "_sort_tmp"
  temp_dir = Path(temp_dir)
  if temp_dir.exists():
    shutil.rmtree(temp_dir)
  temp_dir.mkdir(parents=True, exist_ok=True)

  drop_columns = drop_columns or []

  # phase 1: write sorted rows
  run_files: List[Path] = []
  buffer: List[Dict[str, Any]] = []

  dataset = ds.dataset([input_parquet.as_posix()], format="parquet")
  scanner = dataset.scanner(batch_size=read_batch_rows)

  # progress bar
  run_pbar = tqdm(unit="rows", desc="External sort: building runs")

  def flush_run(run_idx: int):
    nonlocal buffer
    if not buffer:
      return
    buffer.sort(key=lambda r: r[key_column], reverse=descending)
    run_path = temp_dir / f"run_{run_idx:05d}.parquet"
    _write_table_from_rows(run_path, buffer)
    run_files.append(run_path)
    buffer = []

  run_idx = 0
  for batch in scanner.to_batches():
    rows = batch.to_pylist()
    run_pbar.update(len(rows))
    for r in rows:
      buffer.append(r)
      if len(buffer) >= max_rows_in_memory:
        flush_run(run_idx)
        run_idx += 1

  flush_run(run_idx)
  run_pbar.close()

  if not run_files:
    # empty input -> empty output with expected schema
    empty = pa.table({"sentence": pa.array([], pa.string()), "value": pa.array([], pa.float64())})
    pq.write_table(empty, output_parquet.as_posix())
    shutil.rmtree(temp_dir, ignore_errors=True)
    return

  # phase 2: k-way merge
  _merge_runs(
    run_files=run_files,
    output_parquet=output_parquet,
    key_column=key_column,
    descending=descending,
    drop_columns=drop_columns,
    batch_rows=100_000,
  )

  # remove temp
  shutil.rmtree(temp_dir, ignore_errors=True)

def _write_table_from_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
  table = pa.Table.from_pylist(rows)
  pq.write_table(table, path.as_posix())

class _RunCursor:
  """
  Streaming cursor over a parquet run file, yielding dictionaries row-by-row.
  """
  def __init__(self, path: Path, batch_size: int = 50_000):
    self.path = Path(path)
    self.pf = pq.ParquetFile(self.path.as_posix())
    self.batch_size = batch_size
    self._batch_iter = self.pf.iter_batches(batch_size=self.batch_size)
    self._current_rows = []
    self._pos = 0
    self._exhausted = False

  def _load_next_batch(self) -> None:
    try:
      batch = next(self._batch_iter)
    except StopIteration:
      self._current_rows = []
      self._pos = 0
      self._exhausted = True
      return
    self._current_rows = batch.to_pylist()
    self._pos = 0

  def peek(self) -> Optional[Dict[str, Any]]:
    if self._exhausted:
      return None
    if self._pos >= len(self._current_rows):
      self._load_next_batch()
      if self._exhausted:
        return None
    return self._current_rows[self._pos]

  def pop(self) -> Optional[Dict[str, Any]]:
    row = self.peek()
    if row is None:
      return None
    self._pos += 1
    return row

def _merge_runs(
  run_files: List[Path],
  output_parquet: Path,
  key_column: str,
  descending: bool,
  drop_columns: List[str],
  batch_rows: int = 100_000,
) -> None:
  cursors = [_RunCursor(p) for p in run_files]

  # heap items: (key, run_id, tie, row_dict)
  # For descending sort, we invert keys so heapq (min-heap) produces correct order.
  heap: List[Tuple[Any, int, int, Dict[str, Any]]] = []

  def heap_key(v):
    return -v if descending and isinstance(v, (int, float)) else (("~", v) if descending else v)

  # Initialize heap
  for i, c in enumerate(cursors):
    row = c.pop()
    if row is not None:
      k = row[key_column]
      heapq.heappush(heap, (heap_key(k), i, 0, row))

  writer = None
  buffer: List[Dict[str, Any]] = []
  tie_counter = 1  # monotonic tie breaker across pushes

  # progress bar
  merge_pbar = tqdm(unit="rows", desc="External sort: merging runs")

  def flush():
    nonlocal writer, buffer
    if not buffer:
      return
    # Drop columns if requested
    if drop_columns:
      out_rows = []
      for r in buffer:
        rr = {k: v for k, v in r.items() if k not in drop_columns}
        out_rows.append(rr)
    else:
      out_rows = buffer

    table = pa.Table.from_pylist(out_rows)
    if writer is None:
      writer = pq.ParquetWriter(output_parquet.as_posix(), table.schema)
    writer.write_table(table)
    buffer = []

  while heap:
    _, run_id, _, row = heapq.heappop(heap)
    buffer.append(row)
    merge_pbar.update(1)
    if len(buffer) >= batch_rows:
      flush()

    nxt = cursors[run_id].pop()
    if nxt is not None:
      k = nxt[key_column]
      heapq.heappush(heap, (heap_key(k), run_id, tie_counter, nxt))
      tie_counter += 1

  flush()
  if writer is not None:
    writer.close()
  merge_pbar.close()