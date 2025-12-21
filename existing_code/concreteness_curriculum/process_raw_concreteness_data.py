"""
This file creates a dataframe that scores the words based on concreteness values.
Stores this dataframe as a parquet into data/outputs/word_concreteness_mean.parquet
This should be done after running convert_raw_to_parquets.py.
"""

from pathlib import Path

import pandas as pd

script_path = Path(__file__).resolve()
script_dir = script_path.parent

def main():
  base = Path(script_dir, "data/parquet/concreteness.participant")
  files = sorted(base.glob("*.parquet"))

  all_dfs = []
  for file in files:
    df = pd.read_parquet(file, columns=["Word", "Rating"]).copy()
    df["Word"] = df["Word"].str.lower()
    df["is_unknown"] = df["Rating"].isin(["N", "n"])
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce") # N/n -> NaN
    
    agg = df.groupby("Word").agg(
      n_ratings=("Rating", "size"), # all rows, including N/n
      n_unknown=("is_unknown", "sum"), # count of N/n
      sum_num=("Rating", "sum"), # sum of 1..5
      n_numeric=("Rating", "count") # count of 1..5 (excludes N/n)
    )
    all_dfs.append(agg)
  
  # Merge dfs from all shards
  final = pd.concat(all_dfs).groupby(level=0).sum()

  # Compute mean and % of people who knew the word
  final["rating_mean"] = final["sum_num"] / final["n_numeric"]
  final["percent_knew_word"] = final["n_numeric"] / (final["n_unknown"] + final["n_numeric"]) * 100
  out = final[["rating_mean", "n_ratings", "n_unknown", "percent_knew_word"]].reset_index()
  # Drop words that had n_numeric = 0
  out.dropna(subset=["rating_mean"])
  # Drop words known by fewer than 85% of participants
  out = out[out["percent_knew_word"] >= 85]
  
  print(out.head())
  print(f"{len(out)} word concreteness scores found.")
  output_dir = Path(script_dir, "data/outputs")
  output_dir.mkdir(parents=True, exist_ok=True)
  out.to_parquet(Path(output_dir, "word_concreteness_mean.parquet"), index=False)
  print(f"Dataframe downloaded to data/outputs/word_concreteness_mean.parquet")


if __name__ == "__main__":
  main()