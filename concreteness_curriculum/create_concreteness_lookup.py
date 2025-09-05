"""
This file loads data/outputs/word_concreteness_means.parquet into a dict {word: mean_rating}
Scale ranges from 1 to 5.
1 ~ abstract
5 ~ concrete
"""

from typing import Dict, Union
import pandas as pd
from pathlib import Path
script_path = Path(__file__).resolve()
script_dir = script_path.parent

__all__ = ["load_concreteness_word_ratings"]

def load_concreteness_word_ratings (
    path: Union[str, Path] = Path(script_dir, "data/outputs/word_concreteness_mean.parquet")
) -> Dict[str, float]:

  # Confirm path
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"File not found: {path}")

  # Confirm parquet dependencies
  try:
    df = pd.read_parquet(path)[["Word", "rating_mean"]]
  except ImportError as e:
    raise ImportError("Reading Parquet requires 'pyarrow' or 'fastparquet'") from e

  if not df["Word"].is_unique:
    raise ValueError("Duplicate words found.")

  return (
    df.set_index("Word")["rating_mean"]
      .round(2)
      .astype(float)
      .to_dict()
  )

if __name__ == "__main__":
  word_to_rating = load_concreteness_word_ratings()
  for i, (key, value) in enumerate(word_to_rating.items()):
    if i < 10:
      print(f"{key}: {value}")
    else:
      break