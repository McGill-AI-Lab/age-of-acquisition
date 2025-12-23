"""
Summary:
  Converts raw conc data from Brysbaert et al. and a multiword dataset into a clean parquet with columns "word" and "value".

Dependencies:
  Download the raw data from these links and rename the files.

  https://link.springer.com/article/10.3758/s13428-013-0403-5#Sec10 (Supplemental Material)
  ----> BrysbaertConc.xlsx
  https://osf.io/ksypa/files/he4dv
  ----> MultiwordConc.csv
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

PKG_DIR = Path(__file__).resolve().parent
TABLE_DIR = PKG_DIR.parent.parent / "data" / "processed" / "lookup_tables"
DATA_DIR = PKG_DIR.parent.parent / "data" / "raw" / "lexical_norms"

def load_d1(path: str | Path) -> pd.DataFrame:
  df = pd.read_excel(path, sheet_name=0)

  # drop bigrams
  bigram_num = pd.to_numeric(df.get("Bigram"), errors="coerce")
  df = df.loc[~(bigram_num==1)].copy()

  # drop percent known < 0.85
  percent_known_num = pd.to_numeric(df.get("Percent_known"), errors="coerce")
  df = df.loc[percent_known_num >= 0.85].copy()

  # keep only word and value columns
  df = df[["Word", "Conc.M"]].copy()

  df["Conc.M"] = pd.to_numeric(df["Conc.M"], errors="coerce")
  df = df.dropna()

  df = df.rename(columns={"Word": "word", "Conc.M": "value"})

  return df

def load_d2(path: str | Path) -> pd.DataFrame:
  df = pd.read_csv(path)

  df = df[["Expression", "Mean_C"]].copy()

  df["Mean_C"] = pd.to_numeric(df["Mean_C"], errors="coerce")
  df = df.dropna()

  df = df.rename(columns={"Expression": "word", "Mean_C": "value"})

  return df

def build_and_save_conc_table(d1: pd.DataFrame, d2: pd.DataFrame, output_parquet: str) -> pd.DataFrame:
  combined = pd.concat([d1, d2], ignore_index=True)
  combined.to_parquet(output_parquet, index=False)
  return combined

def main(d1_path: str = DATA_DIR / "BrysbaertConc.xlsx", d2_path: str = DATA_DIR / "MultiwordConc.csv"):
  d1 = load_d1(d1_path)
  d2 = load_d2(d2_path)

  # TESTING
  # print(f"""D1 range: {d1["value"].min()}, {d1["value"].max()}""")
  # print(f"""D2 range: {d2["value"].min()}, {d2["value"].max()}""")
  # print(f"Words in D1: {len(d1)}")
  # print(f"Words in D2: {len(d2)}")
  
  combined = build_and_save_conc_table(d1, d2, output_parquet = TABLE_DIR / "conc_table.parquet")

  # print(f"Words in combined: {len(combined)}")

  return combined

if __name__ == "__main__":
  main(
    DATA_DIR / "BrysbaertConc.xlsx",
    DATA_DIR / "MultiwordConc.csv"
  )