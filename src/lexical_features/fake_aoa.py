from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def make_fake_aoa_table(
  seed: int,
  in_path: str | Path,
  out_path: str | Path,
) -> Path:
  in_path = Path(in_path)
  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  df = pd.read_parquet(in_path, columns=["word", "value"]).copy()

  rng = np.random.default_rng(seed)
  shuffled_vals = df["value"].to_numpy().copy()
  rng.shuffle(shuffled_vals)

  fake = df.copy()
  fake["value"] = shuffled_vals
  fake.to_parquet(out_path, index=False)
  return out_path
