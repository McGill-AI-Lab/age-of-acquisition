from pathlib import Path

__all__ = ["aoa", "conc", "freq", "phon"]

PKG_DIR = Path(__file__).resolve().parent
TABLE_DIR = PKG_DIR.parent.parent / "data" / "processed" / "lookup_tables"
DATA_DIR = PKG_DIR.parent.parent / "data" / "raw" / "lexical_norms"

req_datasets = [
    DATA_DIR / "AIGeneratedAoA.xlsx",
    DATA_DIR / "BrysbaertConc.xlsx",
    DATA_DIR / "KupermanAoA.xlsx",
    DATA_DIR / "MultiwordConc.csv",
]

for p in req_datasets:
  if not p.is_file():
    raise FileNotFoundError(f"{p} is missing.")
  
aoa_parquet = TABLE_DIR / "aoa_table.parquet"
aoa_parquet_inflected = TABLE_DIR / "aoa_table_inflected.parquet"
conc_parquet = TABLE_DIR / "conc_table.parquet"
conc_parquet_inflected = TABLE_DIR / "conc_table_inflected.parquet"

if not aoa_parquet.is_file() or not aoa_parquet_inflected.is_file():
    from . import load_aoa
    load_aoa.main()

if not conc_parquet.is_file() or not conc_parquet_inflected.is_file():
    from . import load_conc
    load_conc.main()

from .scoring_functions import aoa, conc, freq, phon