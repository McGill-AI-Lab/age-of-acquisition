from __future__ import annotations
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
import spacy
from tqdm import tqdm
import hashlib

PKG_DIR = Path(__file__).resolve().parent
RAW_DIR = PKG_DIR.parent.parent / "data" / "raw" / "corpora"
BABYLM_DIR = RAW_DIR / "BabyLM"
REFINED_CSV = RAW_DIR / "RefinedBookCorpus.csv"

OUT_DIR = PKG_DIR.parent.parent / "data" / "processed" / "corpora" / "raw_shards"
NUM_SHARDS = 100

# how many rows to accumulate before writing to parquet batch
WRITE_BUFFER_ROWS = 200_000
# chunk size for pandas csv reading
CSV_CHUNKSIZE = 200_000

REFINED_SAMPLE_SEED = 1283
PERCENT_REFINED_KEEP = 10 # percent of refined book corpus to keep 

_NLP = spacy.blank("en")
_NLP.add_pipe("sentencizer")

# helper to decide whether to keep a refined paragraph
def _u64_hash(row_idx: int) -> int:
  return int.from_bytes(
    hashlib.blake2b(f"{REFINED_SAMPLE_SEED}:{row_idx}".encode("utf-8"), digest_size=8).digest(),
    "little",
    signed=False,
  )

# helper to stream lines from a .train file
def _iter_train_lines(train_path: Path) -> Iterator[str]:
  with train_path.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
      text = line.strip()
      if text:
        yield text

# streams lines from all babylm texts
def _iter_babylm_texts() -> Iterator[str]:
  train_files: List[Path] = sorted(BABYLM_DIR.glob("*.train"))
  if not train_files:
    raise FileNotFoundError(f"No files found in {BABYLM_DIR}")
  
  for fp in train_files:
    yield from _iter_train_lines(fp)


# streams sentences from refined book corpus
# uses spacy to split paragraphs into sentences
def _iter_refined_book_corpus_texts(percent_refined: int) -> Iterator[str]:
  if not (0 <= percent_refined <= 100):
    raise ValueError(f"percent_refined must be in [0, 100], got {percent_refined}")

  if not REFINED_CSV.exists():
    raise FileNotFoundError(f"Missing CSV: {REFINED_CSV}")

  header_df = pd.read_csv(REFINED_CSV, nrows=0)
  text_col = str(list(header_df.columns)[0])

  row_idx = 0
  # used to determine if paragraph should be kept or not
  threshold = int((percent_refined / 100.0) * (2**64 - 1))
  
  for chunk in pd.read_csv(REFINED_CSV, usecols=[text_col], chunksize=CSV_CHUNKSIZE):
    # list of paragraphs we keep from chunk
    kept: List[str] = []

    for val in chunk[text_col]:
      if pd.isna(val):
        row_idx += 1
        continue
      p = str(val).strip()
      if p and _u64_hash(row_idx) <= threshold:
        kept.append(p)
      row_idx += 1

    # run spacy only on kept paragraphs
    for doc in _NLP.pipe(kept, batch_size=2048):
      for sent in doc.sents:
        s = sent.text.strip()
        if s:
          yield s

# write shards from input text iterator
def _write_shards_streaming(text_iter: Iterable[str], n_shards: int) -> None:
  if n_shards <= 0:
    raise ValueError("n_shards must be positive")
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  schema = pa.schema([("text", pa.string())])
  writers: List[Optional[pq.ParquetWriter]] = [None] * n_shards 
  buffers: List[List[str]] = [[] for _ in range(n_shards)]

  def ensure_writer(i: int) -> pq.ParquetWriter:
    w = writers[i]
    if w is not None:
      return w
    shard_path = OUT_DIR / f"shard_{i:03d}.parquet"
    w = pq.ParquetWriter(shard_path, schema=schema, compression="zstd")
    writers[i] = w
    return w

  def flush(i: int) -> None:
    if not buffers[i]:
        return
    table = pa.table({"text": buffers[i]})
    ensure_writer(i).write_table(table)
    buffers[i].clear()

  shard_idx = 0
  for text in text_iter:
    buffers[shard_idx].append(text)
    if len(buffers[shard_idx]) >= WRITE_BUFFER_ROWS:
      flush(shard_idx)
    shard_idx = (shard_idx + 1) % n_shards 

  for i in range(n_shards):
    flush(i)

  for w in writers:
    if w is not None:
      w.close()

def load_corpus_shards(n_shards=NUM_SHARDS, percent_refined=PERCENT_REFINED_KEEP) -> None:
  def combined_iter() -> Iterator[str]:
    yield from _iter_babylm_texts()
    yield from _iter_refined_book_corpus_texts(percent_refined)

  # for progress bar
  progress_iter = tqdm(combined_iter(), desc="Writing shards", unit=" sentences")

  _write_shards_streaming(progress_iter, n_shards)
  print(f"Wrote {n_shards} shards to: {OUT_DIR.resolve()}")

def main():
  # TESTING
  # babylm = _iter_babylm_texts()
  # refined = _iter_refined_book_corpus_texts()
  # print([next(babylm) for _ in range(4)])
  # print([next(refined) for _ in range(10)])

  load_corpus_shards(n_shards=NUM_SHARDS, percent_refined=PERCENT_REFINED_KEEP)

if __name__ == "__main__":
  main()