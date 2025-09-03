import glob
import pandas as pd
from pathlib import Path
script_path = Path(__file__).resolve()
script_dir = script_path.parent

def main():
  files = sorted(glob.glob(str(script_dir.as_posix()) + "/data/parquet/concreteness.participant/*.parquet"))

  dfs = []
  for f in files:
    df = pd.read_parquet(f, columns=["Word", "Rating"])
    # Normalize and coerce to numeric
    # 'N'/'n' -> NaN -> dropped
    df["Word"] = df["Word"].str.lower()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])
    dfs.append(df[["Word", "Rating"]])
  
  big = pd.concat(dfs, ignore_index=True)
  out = big.groupby("Word", as_index=False).agg(
      rating_mean=("Rating", "mean"),
      n_ratings=("Rating", "size"),
  )
  print(out.head(20))
  # out.to_parquet("data/outputs/word_concreteness_mean.parquet", index=False)


if __name__ == "__main__":
  main()