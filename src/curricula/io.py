from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any, Iterator, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

def list_parquet_shards(shard_dir: Path) -> list[Path]:
  shard_dir = Path(shard_dir)
  files = sorted([p for p in shard_dir.glob("*.parquet") if p.is_file()])
  if not files:
    raise FileNotFoundError(f"No parquet shards found in {shard_dir}")
  return files


# stream sentences from many praquet shards without putting everything into memory at once
def iter_input_rows_with_source(shard_dir: Path, text_col: str = "text"):
  """
  Yields (sentence, source_file_path) streaming across shards.
  Iterates shard-by-shard so callers can show per-file progress.
  """
  files = list_parquet_shards(shard_dir)
  for fp in files:
    dataset = ds.dataset([fp.as_posix()], format="parquet")
    scanner = dataset.scanner(columns=[text_col], batch_size=200_000)
    for batch in scanner.to_batches():
      col = batch.column(0)
      for v in col.to_pylist():
        yield ("" if v is None else str(v), fp)

# write rows to parquet file in batches
def write_parquet_rows(
  out_path: Path,
  rows: Iterable[Dict[str, Any]],
  batch_rows: int = 100_000,
) -> None:
  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  writer: Optional[pq.ParquetWriter] = None
  buffer = []

  def flush():
    nonlocal writer, buffer
    if not buffer:
      return
    table = pa.Table.from_pylist(buffer)
    if writer is None:
      writer = pq.ParquetWriter(out_path.as_posix(), table.schema)
    writer.write_table(table)
    buffer = []

  for r in rows:
    buffer.append(r)
    if len(buffer) >= batch_rows:
      flush()

  flush()
  if writer is not None:
    writer.close()