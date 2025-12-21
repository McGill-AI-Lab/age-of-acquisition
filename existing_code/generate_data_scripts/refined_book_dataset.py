#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import gzip
import json
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from tqdm import tqdm

# batching & safety
CHUNK_DOCS = 100_000
BATCH_SIZE = 200
N_PROCESS = 2
MAX_CHARS_PER_DOC = 200_000  # hard chunk to avoid spaCy E088 on very long docs




def _kagglehub_download(handle: str) -> Path | None:
   """Fetch dataset via kagglehub and return the local cache path."""
   try:
       import kagglehub  # pip install -U kagglehub
   except Exception:
       print("kagglehub not installed—skipping (pip install -U kagglehub).")
       return None
   try:
       print(f"Downloading with kagglehub… ({handle})")
       cache_path = Path(kagglehub.dataset_download(handle))
       print("KaggleHub cache path:", cache_path)
       return cache_path
   except Exception as e:
       print("kagglehub download failed:", e)
       return None




def _extract_inner_zips(root: Path):
   """Some datasets include nested zips—extract for convenience."""
   for z in root.rglob("*.zip"):
       out_dir = z.parent / (z.stem + "_unzipped")
       if out_dir.exists():
           continue
       try:
           print(f"Extracting inner zip: {z} → {out_dir}")
           out_dir.mkdir(parents=True, exist_ok=True)
           with zipfile.ZipFile(z, "r") as zf:
               zf.extractall(out_dir)
       except Exception as e:
           print(f"Warning: failed to extract {z}: {e}")




# ---------------- readers ----------------
def _iter_txt(path: Path) -> Iterator[str]:
   """Yield safe-size doc chunks from text-like files (blank line boundary + hard chunk)."""
   opener = gzip.open if path.suffix == ".gz" else open
   with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
       buf: list[str] = []
       cur = 0
       for line in f:
           s = line.rstrip("\n")
           if s.strip():
               buf.append(s)
               cur += len(s) + 1
               if cur >= MAX_CHARS_PER_DOC:
                   yield " ".join(buf)
                   buf, cur = [], 0
           else:
               if buf:
                   yield " ".join(buf)
                   buf, cur = [], 0
       if buf:
           yield " ".join(buf)




def _iter_csv(path: Path) -> Iterator[str]:
   """Load CSV and yield a best-effort text field."""
   df = pd.read_csv(path)
   for col in ["text", "book_text", "content", "body", "paragraph", "article", "sentence"]:
       if col in df.columns:
           for t in df[col].astype(str):
               t = t.strip()
               if t:
                   yield t
           return
   # Fallback: concat string-like cells per row
   for _, row in df.iterrows():
       yield " ".join(str(v) for v in row.values if isinstance(v, str))




def _iter_json(path: Path) -> Iterator[str]:
   """Handle JSON arrays or JSONL; common text field names."""
   with open(path, "r", encoding="utf-8", errors="ignore") as f:
       first = f.read(1)
       f.seek(0)
       keys = ["text", "book_text", "content", "body", "paragraph", "article", "sentence"]
       if first == "[":  # JSON array
           arr = json.load(f)
           for obj in arr:
               if isinstance(obj, dict):
                   for k in keys:
                       if k in obj and isinstance(obj[k], str):
                           t = obj[k].strip()
                           if t:
                               yield t
                               break
       else:  # JSONL
           for line in f:
               line = line.strip()
               if not line:
                   continue
               try:
                   obj = json.loads(line)
               except Exception:
                   continue
               if isinstance(obj, dict):
                   for k in keys:
                       if k in obj and isinstance(obj[k], str):
                           t = obj[k].strip()
                           if t:
                               yield t
                               break




def _iter_parquet(path: Path) -> Iterator[str]:
   """Read Parquet and yield text from common columns."""
   df = pd.read_parquet(path)
   for col in ["text", "book_text", "content", "body", "paragraph", "article", "sentence"]:
       if col in df.columns:
           for t in df[col].astype(str):
               t = t.strip()
               if t:
                   yield t
           return
   # fallback: try the first string column
   for col in df.columns:
       if df[col].dtype == object:
           for t in df[col].astype(str):
               t = t.strip()
               if t:
                   yield t
           return




# ---------------- discovery ----------------
def _find_raw_files(raw_dir: Path, extra_root: Path | None) -> list[Path]:
   roots = [p for p in [raw_dir, extra_root] if p and p.exists()]
   pats = [
       "**/*.txt", "**/*.txt.gz",
       "**/*.csv", "**/*.json", "**/*.jsonl",
       "**/*.parquet",
       "**/*.tokens", "**/*.raw", "**/*.text",
   ]
   files: list[Path] = []
   for r in roots:
       for pat in pats:
           files.extend(Path(r).glob(pat))
   files = sorted(set(files))
   if files:
       print(f"Found {len(files)} files under {[str(r) for r in roots]} (showing up to 10):")
       for p in files[:10]:
           print("  -", p)
       if len(files) > 10:
           print(f"  … +{len(files)-10} more")
   return files




def _iter_docs(raw_dir: Path, extra_root: Path | None) -> Iterator[str]:
   files = _find_raw_files(raw_dir, extra_root)
   if not files:
       raise SystemExit(
           f"No raw files found.\n"
           f"- Place files under {raw_dir}\n"
           f"- Or ensure kagglehub is installed & authenticated."
       )
   for fp in files:
       if fp.suffix in {".txt", ".tokens", ".raw", ".text"} or fp.suffixes[-2:] == [".txt", ".gz"]:
           yield from _iter_txt(fp)
       elif fp.suffix == ".csv":
           yield from _iter_csv(fp)
       elif fp.suffix in {".json", ".jsonl"}:
           yield from _iter_json(fp)
       elif fp.suffix == ".parquet":
           yield from _iter_parquet(fp)




# ---------------- batching + nlp ----------------
def _batched(it: Iterable[str], n: int) -> Iterator[list[str]]:
   buf: list[str] = []
   for x in it:
       buf.append(x)
       if len(buf) >= n:
           yield buf
           buf = []
   if buf:
       yield buf




def _load_sentencizer():
   import spacy
   nlp = spacy.blank("en")
   if not nlp.has_pipe("sentencizer"):
       nlp.add_pipe("sentencizer")
   nlp.max_length = 10_000_000
   return nlp




def main():
   parser = argparse.ArgumentParser(
       description="Download and process a book corpus from KaggleHub."
   )
   parser.add_argument(
       "dataset_handle",
       nargs="?",
       default="nishantsingh96/refined-bookcorpus-dataset",
       help="KaggleHub dataset handle (e.g., 'user/dataset-name'). Defaults to 'nishantsingh96/refined-bookcorpus-dataset'.",
   )
   args = parser.parse_args()

   # --- Define project structure relative to this script's location ---
   project_root = Path(__file__).resolve().parent
   data_dir = project_root / "data"
   raw_base_dir = data_dir / "raw"
   processed_base_dir = data_dir / "processed"

   # Ensure base directories exist
   raw_base_dir.mkdir(parents=True, exist_ok=True)
   processed_base_dir.mkdir(parents=True, exist_ok=True)

   dataset_handle = args.dataset_handle
   dataset_name = dataset_handle.split("/")[-1]

   raw_dir = raw_base_dir / dataset_name
   out_dir = processed_base_dir / dataset_name

   out_dir.mkdir(parents=True, exist_ok=True)
   kh_root = _kagglehub_download(dataset_handle)
   if kh_root:
       _extract_inner_zips(kh_root)

   nlp = _load_sentencizer()

   shard_idx = 0
   total_docs = 0
   for batch in _batched(_iter_docs(raw_dir, kh_root), CHUNK_DOCS):
       tokenized: list[list[str]] = []
       for doc in tqdm(
           nlp.pipe(batch, batch_size=BATCH_SIZE, n_process=N_PROCESS),
           total=len(batch),
           desc=f"Tokenizing shard {shard_idx:03d}",
       ):
           for s in doc.sents:
               toks = [t.text.lower() for t in s if not t.is_space]
               if toks:
                   tokenized.append(toks)

       out_path = out_dir / f"{dataset_name}_tokens_part-{shard_idx:03d}.parquet"
       pd.DataFrame({"tokens": tokenized}).to_parquet(out_path, index=False)
       total_docs += len(batch)
       print(f"✅ wrote {len(tokenized):,} sentences from {len(batch):,} docs → {out_path}")

       del tokenized
       gc.collect()
       shard_idx += 1


   print(f"Done. Shards: {shard_idx}, docs processed: {total_docs:,}")




if __name__ == "__main__":
   main()