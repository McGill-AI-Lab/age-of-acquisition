"""
  Summary:
    Has one utility function add_inflections that adds inflections to an input DataFrame and returns a new one.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

EPS = 1e-5

PKG_DIR = Path(__file__).resolve().parent
TABLE_DIR = PKG_DIR.parent.parent / "data" / "processed" / "lookup_tables"

def add_inflections(df: pd.DataFrame, eps: float = EPS) -> pd.DataFrame:
  """
  For each base word w in df['word']:
    - generate inflections via lemminflect (verbs/nouns/adjectives)
    - for each inflected form not already in df, append it with value = base_value + eps

  Returns a NEW dataframe (does not mutate input).
  """
  try:
    from lemminflect import getAllInflections
  except ImportError as e:
    raise ImportError(
      "lemminflect is required for adding inflections. Install with:\n"
      "  pip install lemminflect\n"
    ) from e

  # make sure no duplicates
  out = df.copy()
  out["word"] = out["word"].astype(str).str.strip().str.lower()
  out = out.dropna(subset=["word", "value"])
  out = out[out["word"] != ""].copy()
  out = out.drop_duplicates(subset=["word"], keep="first")
  out = out.reset_index(drop=True)
  # helper column used to keep inflected forms after base forms
  out["_base_rank"] = out.index

  existing = set(out["word"])
  # set "word" to index for O(1) lookup
  base_to_value = out.set_index("word")["value"]
  base_to_rank = out.set_index("word")["_base_rank"]

  new_rows = []

  # iterate bases once, membership checks are O(1) via set
  for base, val in base_to_value.items():
    # getAllInflections returns dict like {"NNS": ("cats",), "VBD": ("walked", ...), ...}
    infl_dict = getAllInflections(base)

    # no inflections
    if not infl_dict:
      continue

    # flatten to a set of candidate strings
    infls = set()
    for forms in infl_dict.values():
      infls.update(forms)

    # normalize / filter / add
    for w in infls:
      if not w:
        continue
      w = str(w).strip().lower()
      if not w or w == base:
        continue
      if w in existing:
        continue
      existing.add(w)
      new_rows.append(
        {
          "word": w,
          "value": val + eps,
          "_base_rank": int(base_to_rank.loc[base]),
          "_within_rank": 1,
        }
      )

  # no inflections added
  if not new_rows:
    return out.drop(columns=["_base_rank"])

  # print(f"{len(new_rows)} inflected forms added to lookup table.")
  additions = pd.DataFrame(new_rows)
  out["_within_rank"] = 0
  combined = pd.concat([out, additions], ignore_index=True)

  combined["_word_sort"] = combined["word"]  # for stable inflection ordering
  combined = combined.sort_values(
    by=["_base_rank", "_within_rank", "_word_sort"],
    kind="mergesort",  # stable sort
    ignore_index=True,
  )
  return combined.drop(columns=["_base_rank", "_within_rank", "_word_sort"]) 

# tests add_inflections on existing aoa table; will fail if you don't have the table already
def main():
  df = pd.read_parquet(TABLE_DIR / "aoa_table.parquet", columns=["word", "value"]).copy()
  print(df.head(10))
  print(f"Length of df before: {len(df)}")

  df = add_inflections(df)
  print(df.head(10))
  print(f"Length of df after: {len(df)}")

  df = pd.read_parquet(TABLE_DIR / "conc_table.parquet", columns=["word", "value"]).copy()
  print(df.head(10))
  print(f"Length of df before: {len(df)}")

  df = add_inflections(df, eps=-1e-5)
  print(df.head(10))
  print(f"Length of df after: {len(df)}")


if __name__ == "__main__":
  main()