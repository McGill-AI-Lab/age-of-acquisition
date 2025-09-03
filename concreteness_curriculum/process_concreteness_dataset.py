"""
Download dataset from https://osf.io/7pqyg
Add concreteness.participant.rda to concreteness_curriculum/data/

This file loads the data into a pandas DataFrame and then downloads the data into parquet shards for easy testing & access
"""
import pyreadr # pip install pyreadr
import pandas as pd
import argparse, os, json, hashlib, math
from pathlib import Path
script_path = Path(__file__).resolve()
script_dir = script_path.parent

# Converts dataframe into parquest shards and downloads them to out_dir
def write_shards(df: pd.DataFrame, out_dir: Path, base_name: str, rows_per_shard: int):
  out_dir.mkdir(parents=True, exist_ok=True)
  n_rows = len(df)
  n_shards = max(1, math.ceil(n_rows / rows_per_shard))
  manifest = {
    "object": base_name,
    "total_rows": int(n_rows),
    "rows_per_shard": int(rows_per_shard),
    "shards": []
  }

  for i in range(n_shards):
    start = i * rows_per_shard
    stop = min((i + 1) * rows_per_shard, n_rows)
    shard = df.iloc[start:stop]
    shard_name = f"{base_name}.part-{i:05d}.parquet"
    shard_path = out_dir / shard_name
    shard.to_parquet(shard_path, index=False)
    manifest["shards"].append({
      "file": shard_name,
      "rows": int(len(shard))
    })

  with open(out_dir / f"{base_name}.manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
  return manifest

def main():
  """ Parameters """
  rda_path = script_dir / "data" / "concreteness.participant.rda"
  out_root = script_dir / "data" / "parquet"
  rows_per_shard = 50_000

  print("Starting to read in data...")
  result = pyreadr.read_r(rda_path)
  print("Finished readint data in...")

  for obj_name, obj in result.items():
    if not isinstance(obj, pd.DataFrame):
      continue
      
    df = obj.copy()
    object_out_dir = out_root / obj_name
    manifest = write_shards(
      df=df,
      out_dir=object_out_dir,
      base_name=obj_name,
      rows_per_shard=rows_per_shard,
    )
    print(f"Wrote {len(manifest['shards'])} shards for object '{obj_name}' to {'object_out_dir'}")


if __name__ == "__main__":
  main()