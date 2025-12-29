from __future__ import annotations
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
import spacy
from tqdm import tqdm

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

_NLP = spacy.blank("en")
_NLP.add_pipe("sentencizer")

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
def _iter_refined_book_corpus_texts() -> Iterator[str]:
  if not REFINED_CSV.exists():
    raise FileNotFoundError(f"Missing CSV: {REFINED_CSV}")

  header_df = pd.read_csv(REFINED_CSV, nrows=0)
  text_col = str(list(header_df.columns)[0])
  
  for chunk in pd.read_csv(REFINED_CSV, usecols=[text_col], chunksize=CSV_CHUNKSIZE):
    # get paragraphs as string, normalized
    paragraphs: List[str] = (
      chunk[text_col]
      .dropna()
      .astype(str)
      .map(lambda s: s.strip())
      .tolist()
    )

    for doc in _NLP.pipe(paragraphs, batch_size=2048):
      for sent in doc.sents:
        s = sent.text.strip()
        if s:
          yield s

# write shards from input text iterator
def _write_shards_streaming(text_iter: Iterable[str]) -> None:
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  schema = pa.schema([("text", pa.string())])
  writers: List[Optional[pq.ParquetWriter]] = [None] * NUM_SHARDS
  buffers: List[List[str]] = [[] for _ in range(NUM_SHARDS)]

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
    shard_idx = (shard_idx + 1) % NUM_SHARDS

  for i in range(NUM_SHARDS):
    flush(i)

  for w in writers:
    if w is not None:
      w.close()

def load_corpus_shards() -> None:
  def combined_iter() -> Iterator[str]:
    yield from _iter_babylm_texts()
    yield from _iter_refined_book_corpus_texts()

  # for progress bar
  progress_iter = tqdm(combined_iter(), desc="Writing shards", unit=" sentences")

  _write_shards_streaming(progress_iter)
  print(f"Wrote {NUM_SHARDS} shards to: {OUT_DIR.resolve()}")

def main():
  # TESTING
  # babylm = _iter_babylm_texts()
  # refined = _iter_refined_book_corpus_texts()
  # print([next(babylm) for _ in range(4)])
  # print([next(refined) for _ in range(10)])

  load_corpus_shards()

if __name__ == "__main__":
  main()