from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal, Optional, Dict, Any
from tqdm import tqdm

from curricula.io import iter_input_rows_with_source, list_parquet_shards, write_parquet_rows
from curricula.external_sort import external_sort_parquet
from curricula.tranche import write_sentence_based_tranches, write_word_based_tranches
from curricula.naming import make_curriculum_dir, make_run_dir
from curricula.config import write_config_json

from sentence_scoring import score_sentence

# Str parameters for building curricula
Curriculum = Literal["shuffled", "aoa", "conc", "freq", "phon"]
Method = Literal["mean", "min", "max", "add"]
Order = Literal["asc", "desc"]
TrancheType = Literal["word-based", "sentence-based"]

PKG_DIR = Path(__file__).resolve().parent
BASE_DIR = PKG_DIR.parent.parent / "data" / "processed" / "corpora"
IN_DIR = BASE_DIR / "raw_shards"
OUT_BASE = BASE_DIR / "training"

def build_curriculum(
  curriculum: Curriculum = "aoa",
  scoring_method: Method = "max",
  sort_order: Order = "asc",
  tranche_type: TrancheType = "word-based",
  tranche_size: int = 500,
  aoa_agnostic: bool = True,
  multiword: bool = False,
  skip_stopwords: bool = False,
  inflect: bool = False,
) -> Path: # output path where curriculum was built
  if tranche_size <= 0:
    raise ValueError("tranche_size must be > 0")

  if curriculum != "shuffled" and sort_order not in ("asc", "desc"):
    raise ValueError("sort_order must be 'asc' or 'desc'")

  if not IN_DIR.exists():
    FileNotFoundError(f"Input shard directory not found: {IN_DIR}")
  
  OUT_BASE.mkdir(parents=True, exist_ok=True)

  # create out directory with parameters in folder name
  out_dir = make_curriculum_dir(
    out_base=OUT_BASE,
    curriculum=curriculum,
    scoring_method=scoring_method,
    sort_order=sort_order,
    tranche_type=tranche_type,
    tranche_size=tranche_size,
    aoa_agnostic=aoa_agnostic,
    multiword=multiword,
    skip_stopwords=skip_stopwords,
    inflect=inflect,
  )
  out_dir.mkdir(parents=True, exist_ok=False)

  # temp dir next to output
  run_dir = make_run_dir(out_dir)
  run_dir.mkdir(parents=True, exist_ok=True)

  scored_path = run_dir / "scored.parquet"

  # meta data for building curriculum 
  meta: Dict[str, Any] = {
    "input_dir": str(IN_DIR),
    "output_dir": str(out_dir),
    "curriculum": curriculum,
    "scoring_method": scoring_method,
    "sort_order": sort_order,
    "tranche_type": tranche_type,
    "tranche_size": tranche_size,
    "aoa_agnostic": aoa_agnostic,
    "multiword": multiword,
    "skip_stopwords": skip_stopwords,
    "inflect": inflect,
  }

  # seed for shuffled curriculum
  seed: Optional[int] = None
  if curriculum == "shuffled":
    import secrets
    seed = secrets.randbits(64)
    meta["shuffled_seed"] = seed
  
  # stored in metadata later 
  total_in = 0
  total_kept = 0
  total_dropped = 0
  
  def scored_row_generator():
    nonlocal total_in, total_kept, total_dropped

    rng = None
    if curriculum == "shuffled":
      import numpy as np
      rng = np.random.default_rng(seed)
    
    shard_files = list_parquet_shards(IN_DIR)
    total_bytes = sum(p.stat().st_size for p in shard_files)
    # Progress by bytes (cheap, works even if row count is unknown)
    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Scoring shards")
    current_file = None
    last_file = None

    for sent, fp in iter_input_rows_with_source(IN_DIR, text_col="text"):
      # When file changes, advance pbar by that file's size exactly once
      if last_file is None:
        last_file = fp
      if fp != last_file:
        pbar.update(last_file.stat().st_size)
        last_file = fp
      
      total_in += 1
      s = sent.strip()
      if not s:
        total_dropped += 1
        continue
      
      # shuffled generates a random sort_key; value is a placeholder
      if curriculum == "shuffled":
        sort_key = int(rng.integers(0, 2**63 - 1, dtype="int64"))
        total_kept += 1
        yield {"sentence": s, "value": 0.0, "sort_key": sort_key}
        continue
      
      # otherwise score sentence normally and drop if ineligible
      score = _score_sentence_for_curriculum(
        sentence=s,
        curriculum=curriculum,
        scoring_method=scoring_method,
        aoa_agnostic=aoa_agnostic,
        multiword=multiword,
        skip_stopwords=skip_stopwords,
        inflect=inflect,
      )
      if score == -1:
        total_dropped += 1
        continue

      total_kept += 1
      yield {"sentence": s, "value": float(score), "sort_key": float(score)}
    
    # close out last file
    if last_file is not None:
      pbar.update(last_file.stat().st_size)
    pbar.close()
  
  write_parquet_rows(scored_path, scored_row_generator(), batch_rows=100_000)

  meta["total_input_rows"] = total_in
  meta["total_kept_rows"] = total_kept
  meta["total_dropped_rows"] = total_dropped

  # sort scored sentences
  ordered_path = run_dir / "ordered.parquet"
  descending = (curriculum != "shuffled" and sort_order == "desc")
  external_sort_parquet(
    input_parquet=scored_path,
    output_parquet=ordered_path,
    key_column="sort_key",
    descending=descending,
    drop_columns=["sort_key"],
    temp_dir=run_dir / "sort_tmp",
    max_rows_in_memory=500_000, # may need to tune as needed
    read_batch_rows=200_000,
  )

  # separate into tranches
  tranche_root = out_dir / "tranches"
  tranche_root.mkdir(parents=True, exist_ok=True)
  if tranche_type == "sentence-based":
    num_tranches = write_sentence_based_tranches(
      ordered_parquet=ordered_path,
      out_dir=tranche_root,
      tranche_size=tranche_size,
    )
  elif tranche_type == "word-based":
    num_tranches = write_word_based_tranches(
      ordered_parquet=ordered_path,
      out_dir=tranche_root,
      tranche_size=tranche_size,
    )
  else:
    raise ValueError(f"Unknown tranche_type: {tranche_type}")

  meta["num_tranches"] = num_tranches

  # write config
  write_config_json(out_dir / "config.json", meta)

  # UNCOMMENT TO DELETE TEMP DIR
  # shutil.rmtree(run_dir, ignore_errors=True)

  return out_dir


def _score_sentence_for_curriculum(
  sentence: str,
  curriculum: str,
  scoring_method: str,
  aoa_agnostic: bool,
  multiword: bool,
  skip_stopwords: bool,
  inflect: bool,
) -> float:
  if curriculum == "aoa":
    score, _ = score_sentence(
      sentence,
      metric="aoa",
      method=scoring_method,
      multiword=False,
      skip_stopwords=skip_stopwords,
      inflect=inflect,
    )
    return float(score)

  if curriculum in ("conc", "freq", "phon"):
    if aoa_agnostic:
      # normal scoring
      use_multiword = bool(multiword and curriculum == "conc")
      score, _ = score_sentence(
        sentence,
        metric=curriculum,
        method=scoring_method,
        multiword=use_multiword,
        skip_stopwords=skip_stopwords,
        inflect=inflect,
      )
      return float(score)

    # aoa_agnostic = False
    # take score of max aoa word
    aoa_score, witness = score_sentence(
      sentence,
      metric="aoa",
      method="max",
      multiword=False,
      skip_stopwords=skip_stopwords,
      inflect=inflect,
    )
    if aoa_score == -1 or not witness:
      return float(-1)
    
    from lexical_features import conc, freq, phon
    if curriculum == "conc":
      v = conc(witness, inflect=inflect)
    elif curriculum == "freq":
      v = freq(witness)
    else:
      v = phon(witness)
    
    return float(v) if v != -1 else float(-1)

  raise ValueError(f"Unknown curriculum: {curriculum}")