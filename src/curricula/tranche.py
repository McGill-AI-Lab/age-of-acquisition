from __future__ import annotations

from pathlib import Path
from typing import Iterator, Dict, Any, List, Set, Optional

import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from curricula.simple_tokenizer import tokenize_words_lower_already
from curricula.path_utils import resolve_curriculum_path, find_tranche_dirs, get_tranches_dir
from curricula.io import write_parquet_rows

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
  max_tranches: int = -1,
) -> int:
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  tranche_idx = 1
  buffer: List[Dict[str, Any]] = []

  def flush_tranche(rows: List[Dict[str, Any]], idx: int) -> bool:
    # return True to signal "stop" due to max_tranches cap
    if max_tranches != -1 and idx > max_tranches:
      return True
    tdir = out_dir / f"tranche_{idx:04d}"
    tdir.mkdir(parents=True, exist_ok=False)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, (tdir / "data.parquet").as_posix())
    return False

  pbar = tqdm(unit="rows", desc="Tranching (sentence-based)")
  stopped = False
  for row in _iter_ordered_rows(ordered_parquet):
    buffer.append(row)
    pbar.update(1)
    if len(buffer) >= tranche_size:
      if flush_tranche(buffer, tranche_idx):
        # this buffer would become tranche_idx (> cap); discard and stop
        buffer = []
        stopped = True
        break
      tranche_idx += 1
      buffer = []

  if (not stopped) and buffer:
    if not flush_tranche(buffer, tranche_idx):
      tranche_idx += 1
  
  pbar.close()
  return tranche_idx - 1

def write_word_based_tranches(
  ordered_parquet: Path,
  out_dir: Path,
  tranche_size: int,
  max_tranches: int = -1,
) -> int:
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  seen_words: Set[str] = set()
  new_words_current: Set[str] = set()
  rows_current: List[Dict[str, Any]] = []

  tranche_idx = 1

  def flush_tranche(idx: int) -> bool:
    nonlocal seen_words, new_words_current, rows_current
    # return True to signal "stop" due to max_tranches cap
    if max_tranches != -1 and idx > max_tranches:
      return True

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
    return False

  it = _iter_ordered_rows(ordered_parquet)
  pbar = tqdm(unit="rows", desc="Tranching (word-based)")
  pending: Optional[Dict[str, Any]] = None
  stopped = False

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
      if flush_tranche(tranche_idx):
        # Would exceed cap (trying to write tranche_idx > max_tranches)
        rows_current = []
        new_words_current = set()
        stopped = True
        break
      tranche_idx += 1

      # reprocess this sentence in the next tranche
      pending = row
      continue

    if would_exceed and not rows_current:
      # case that there is a single sentence in a tranche
      rows_current.append(row)
      new_words_current |= introduced
      if flush_tranche(tranche_idx):
        rows_current = []
        new_words_current = set()
        stopped = True
        break
      tranche_idx += 1
      continue

    # normal add
    rows_current.append(row)
    new_words_current |= introduced

    # close tranche if we hit exactly tranche_size new words
    if len(new_words_current) >= tranche_size:
      if flush_tranche(tranche_idx):
        rows_current = []
        new_words_current = set()
        stopped = True
        break
      tranche_idx += 1
    
  if (not stopped) and rows_current:
    if not flush_tranche(tranche_idx):
      tranche_idx += 1

  pbar.close() 
  return tranche_idx - 1

def write_matching_tranches(
  ordered_parquet: Path,
  out_dir: Path,
  matching_idx: str,
  max_tranches: int = -1,
) -> int:
  """
  Writes tranches to out_dir where tranche i's total word count matches (by word) the i-th tranche
  of the reference curriculum at matching_idx.
  """
  targets = _load_matching_tranche_word_targets(matching_idx)
  if not targets:
    raise RuntimeError("Reference curriculum produced no tranche word targets.")

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  dataset = ds.dataset([Path(ordered_parquet).as_posix()], format="parquet")
  scanner = dataset.scanner(columns=["sentence", "value"], batch_size=200_000)

  tranche_idx = 1
  current_rows: List[Dict[str, Any]] = []
  current_words = 0

  def current_target() -> int:
    # tranche_idx is 1-based
    if tranche_idx <= len(targets):
      return targets[tranche_idx - 1]
    return targets[-1]

  def flush_tranche() -> bool:
    nonlocal tranche_idx, current_rows, current_words
    if not current_rows:
      return False
    # cap: never write more than max_tranches
    if max_tranches != -1 and tranche_idx > max_tranches:
      return True
    tranche_dir = out_dir / f"tranche_{tranche_idx:04d}"
    tranche_dir.mkdir(parents=True, exist_ok=False)
    write_parquet_rows(tranche_dir / "data.parquet", current_rows, batch_rows=100_000)
    tranche_idx += 1
    current_rows = []
    current_words = 0
    return False

  pf = pq.ParquetFile(Path(ordered_parquet).as_posix())
  total_rows = int(pf.metadata.num_rows) if pf.metadata is not None else None
  pbar = tqdm(total=total_rows, desc="Building matching tranches", unit="rows")
  stopped = False

  for batch in scanner.to_batches():
    sent_col = batch.column(0).to_pylist()
    val_col = batch.column(1).to_pylist()

    for s, v in zip(sent_col, val_col):
      s = "" if s is None else str(s)
      if not s.strip():
        continue

      w = len(tokenize_words_lower_already(s))
      target = current_target()

      # If adding would exceed target and we already have something, flush first.
      if current_rows and (current_words + w) > target:
        if flush_tranche():
          # cap reached: stop without writing further tranches
          current_rows = []
          current_words = 0
          stopped = True
          break
        target = current_target()

      current_rows.append({"sentence": s, "value": float(v) if v is not None else 0.0})
      current_words += w

      # Optional: If we hit target exactly, flush immediately.
      if current_words == target:
        if flush_tranche():
          current_rows = []
          current_words = 0
          stopped = True
          break
  
    pbar.update(batch.num_rows)
    if stopped:
      break

  # flush remainder
  if not stopped:
    flush_tranche()
  pbar.close()

  # tranche_idx was incremented after last flush; num tranches = tranche_idx - 1
  return tranche_idx - 1

def _count_words_in_parquet(parquet_path: Path, text_col: str = "sentence", batch_size: int = 200_000) -> int:
  """
  Counts total words in a parquet column by streaming batches (keeps memory stable).
  """
  dataset = ds.dataset([parquet_path.as_posix()], format="parquet")
  scanner = dataset.scanner(columns=[text_col], batch_size=batch_size)

  total = 0
  for batch in scanner.to_batches():
    col = batch.column(0)
    for v in col.to_pylist():
      if not v:
        continue
      total += len(tokenize_words_lower_already(str(v)))
  return total

def _load_matching_tranche_word_targets(matching_idx: str) -> List[int]:
  """
  Returns a list of tranche word totals from the reference curriculum, ordered by tranche index.
  Expected structure:
    <curriculum_name>/tranches/tranche_0001/data.parquet, ...
  """
  ref_root = resolve_curriculum_path(matching_idx)
  ref_tranches_dir = get_tranches_dir(ref_root)

  if not ref_tranches_dir.is_dir():
    raise FileNotFoundError(f"Could not find tranches directory for matching_idx at: {ref_tranches_dir}")

  tranche_dirs = find_tranche_dirs(ref_tranches_dir)
  if not tranche_dirs:
    raise RuntimeError(f"No tranche_* directories found under: {ref_tranches_dir}")

  targets: List[int] = []

  pbar = tqdm(tranche_dirs, desc=f"Reading target tranche sizes ({matching_idx})")
  for tdir in pbar:
    dp = tdir / "data.parquet"
    if not dp.exists():
      raise FileNotFoundError(f"Missing reference tranche parquet: {dp}")
    
    wcount = _count_words_in_parquet(dp, text_col="sentence")
    targets.append(wcount)

    pbar.set_postfix(tranche=tdir.name, words=wcount, refresh=False)

  return targets

def write_word_count_tranches(
  ordered_parquet: Path,
  out_dir: Path,
  tranche_size: int,
  max_tranches: int = -1,
) -> int:
  """
  Word-count tranches with exact tranche_size word tokens per tranche (except possibly last),
  by splitting sentences across tranche boundaries.

  Word definition is aligned with simple_tokenizer: [a-z]+(?:'[a-z]+)?
  (expects lowercased input for consistent counting).
  """
  import re
  from typing import List, Dict, Any
  from pathlib import Path
  import pyarrow.parquet as pq
  from tqdm import tqdm

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  # Match the tokenizer used elsewhere (tokenize_words_lower_already)
  WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

  def count_words(s: str) -> int:
    return len(WORD_RE.findall(s))

  def split_at_k_words(s: str, k: int) -> tuple[str, str]:
    """
    Split string s into (left, right) where left contains exactly k word tokens
    per WORD_RE. Keeps punctuation attached to the left side when it directly
    follows the k-th word (e.g., "dog," stays with left).
    """
    if k <= 0:
      return ("", s)

    matches = list(WORD_RE.finditer(s))
    if len(matches) <= k:
      return (s, "")

    # End of the k-th word token
    end = matches[k - 1].end()

    # Extend cut forward to include any immediately-following punctuation
    # until whitespace (so "dog," stays in left not right).
    cut = end
    n = len(s)
    while cut < n and not s[cut].isspace():
      cut += 1

    left = s[:cut].rstrip()
    right = s[cut:].lstrip()
    return (left, right)

  tranche_idx = 1
  current_rows: List[Dict[str, Any]] = []
  current_words = 0  # word token count in current tranche

  def flush_tranche(idx: int) -> bool:
    nonlocal current_rows, current_words
    if not current_rows:
      return False
    # cap: never write more than max_tranches
    if max_tranches != -1 and idx > max_tranches:
      return True
    tdir = out_dir / f"tranche_{idx:04d}"
    tdir.mkdir(parents=True, exist_ok=False)
    write_parquet_rows(tdir / "data.parquet", current_rows, batch_rows=100_000)
    current_rows = []
    current_words = 0
    pbar.set_postfix(tranche=idx, refresh=False)
    return False

  pf = pq.ParquetFile(Path(ordered_parquet).as_posix())
  total_rows = int(pf.metadata.num_rows) if pf.metadata is not None else None
  pbar = tqdm(total=total_rows, desc="Tranching (word-count exact)", unit="rows")
  stopped = False

  for row in _iter_ordered_rows(ordered_parquet):
    pbar.update(1)

    s = "" if row["sentence"] is None else str(row["sentence"])
    s = s.strip()
    if not s:
      continue

    # Keep the same score/value for all split pieces of the sentence
    val = float(row["value"]) if row.get("value", None) is not None else 0.0

    # We may need to split this sentence across multiple tranches
    remaining = tranche_size - current_words
    piece = s

    while piece:
      w = count_words(piece)
      if w == 0:
        # no countable words -> ignore (or you could keep as 0-word row; current pipeline drops empties earlier)
        break

      remaining = tranche_size - current_words

      if w < remaining:
        # Fits without filling tranche exactly
        current_rows.append({"sentence": piece, "value": val})
        current_words += w
        break

      if w == remaining:
        # Perfectly fills tranche
        current_rows.append({"sentence": piece, "value": val})
        current_words += w  # now == tranche_size
        if flush_tranche(tranche_idx):
          # cap reached: discard remainder and stop
          current_rows = []
          current_words = 0
          stopped = True
          break
        tranche_idx += 1
        break

      # w > remaining: split so left has exactly 'remaining' words
      left, right = split_at_k_words(piece, remaining)

      # Safety: if splitting failed in an unexpected way, avoid infinite loops
      if not left:
        # Fallback: put entire piece in current tranche (will exceed) — but this should never happen
        current_rows.append({"sentence": piece, "value": val})
        current_words += w
        if flush_tranche(tranche_idx):
          current_rows = []
          current_words = 0
          stopped = True
          break
        tranche_idx += 1
        break

      # Add left piece to fill tranche exactly, flush, continue with right in next tranche
      current_rows.append({"sentence": left, "value": val})
      current_words += count_words(left)  # should equal remaining

      if flush_tranche(tranche_idx):
        current_rows = []
        current_words = 0
        stopped = True
        break
      tranche_idx += 1

      piece = right  # continue splitting remainder into subsequent tranches

      if stopped:
        break
    
    if stopped:
      break

  # Last tranche may be < tranche_size
  if (not stopped) and current_rows:
    if not flush_tranche(tranche_idx):
      tranche_idx += 1

  pbar.close()
  return tranche_idx - 1
