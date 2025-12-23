"""
Summary:
  Converts raw aoa data from Kuperman et al. into a clean parquet with columns "word" and "value".

Dependencies:
  Download the raw data from these links and rename the files.

  https://osf.io/d7x6q/files/vb9je
  ----> KupermanAoA.xlsx
  https://osf.io/ch48r/files/ma586?view_only=ca45900ffc264645a32394d256101e7d
  ----> AIGeneratedAoA.xlsx
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd 

PKG_DIR = Path(__file__).resolve().parent
TABLE_DIR = PKG_DIR.parent.parent / "data" / "processed" / "lookup_tables"
DATA_DIR = PKG_DIR.parent.parent / "data" / "raw" / "lexical_norms"

# D1 is Kuperman AoA
# D2 is Ai AoA
D1_WORD_COL = "Word"
D1_AOA_COL = "Rating.Mean"
D1_DUNNO_COL = "Dunno"

D2_WORD_COL = "ECP_Target_Word_Mandera_et_al_2020"
D2_AOA_COL = "AI_Kuperman_et_al_2012 _AoA_Finetuned"

def _norm_word(s: pd.Series) -> pd.Series:
  return s.astype(str).str.strip().str.lower()

def load_d1(path: str | Path, sheet_name=0, dunno_threshold: float = 0.05) -> pd.DataFrame:

  # load excel file as dataframe and check for columns
  path = Path(path)
  df = pd.read_excel(path, sheet_name=sheet_name)
  needed = {D1_WORD_COL, D1_AOA_COL, D1_DUNNO_COL}
  missing = needed - set(df.columns)
  if missing:
    raise KeyError(f"D1 is missing columns: {missing}")
  
  # filter Dunno and keep only word and aoa columns
  df = df[df[D1_DUNNO_COL] > dunno_threshold].copy()
  df = df[[D1_WORD_COL, D1_AOA_COL]].copy()

  # rename word and aoa columns
  df = df.rename(columns={D1_WORD_COL: "word", D1_AOA_COL: "value"})

  # process columns
  df["word"] = _norm_word(df["word"])
  df["value"] = pd.to_numeric(df["value"], errors="coerce").dropna()
  df = df.dropna(subset=["word"])
  df = df[df["word"] != ""]
  df = df.drop_duplicates(subset=["word"], keep="first")

  return df

def load_d2(path: str | Path, sheet_name=0) -> pd.DataFrame:

  # load excep file as dataframe and check for columns
  path = Path(path)
  df = pd.read_excel(path, sheet_name=sheet_name)
  needed = {D2_WORD_COL, D2_AOA_COL}
  missing = needed - set(df.columns)
  if missing:
    raise KeyError(f"D2 is missing columns: {missing}")

  # keep only word and aoa columns
  df = df[[D2_WORD_COL, D2_AOA_COL]].copy()
  df = df.rename(columns={D2_WORD_COL: "word", D2_AOA_COL: "value"})

  # process columns
  df["word"] = _norm_word(df["word"])
  df["value"] = pd.to_numeric(df["value"], errors="coerce").dropna()
  df = df.dropna(subset=["word"])
  df = df[df["word"] != ""]
  df = df.drop_duplicates(subset=["word"], keep="first")

  return df

def build_and_save_aoa_table(d1: pd.DataFrame, d2: pd.DataFrame, out_parquet: str | Path) -> pd.DataFrame:
  d1_words = set(d1["word"])
  d2_only = d2[~d2["word"].isin(d1_words)].copy()

  combined = pd.concat([d1, d2_only], ignore_index=True)
  combined = combined.drop_duplicates(subset=["word"], keep="first")

  out_parquet = Path(out_parquet)
  combined.to_parquet(out_parquet, index=False)

  return combined

def main(d1_path: str = DATA_DIR / "KupermanAoA.xlsx", d2_path: str = DATA_DIR / "AIGeneratedAoA.xlsx", d1_sheet=0, d2_sheet=0):
  d1 = load_d1(d1_path, sheet_name=d1_sheet, dunno_threshold=0.15)
  d2 = load_d2(d2_path, sheet_name=d2_sheet)

  # TESTING
  # range
  # print(f"""D1 range: {d1["value"].min()}, {d1["value"].max()}""")
  # print(f"""D2 range: {d2["value"].min()}, {d2["value"].max()}""")
  # d1_words = set(d1["word"])
  # d2_words = set(d2["word"])
  # missing_in_d1 = sorted(d2_words - d1_words)
  # print(f"Words in D1: {len(d1_words)}")
  # print(f"Words in D2: {len(d2_words)}")
  # print(f"Words in D2 but not in D1: {len(missing_in_d1)}")
  
  combined = build_and_save_aoa_table(d1, d2, out_parquet = TABLE_DIR / "aoa_table.parquet")

  # print(f"Words in combined: {len(combined)}")

  return combined

if __name__ == "__main__":
  main(
    d1_path = DATA_DIR / "KupermanAoA.xlsx",
    d2_path = DATA_DIR / "AIGeneratedAoA.xlsx"
  )