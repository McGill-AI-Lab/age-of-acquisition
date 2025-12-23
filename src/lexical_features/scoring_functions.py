from pathlib import Path
import pandas as pd
from typing import Callable

from wordfreq import word_frequency, zipf_frequency

import re
import pronouncing

# Used to memoize phon
from functools import lru_cache

PKG_DIR = Path(__file__).resolve().parent
TABLE_DIR = PKG_DIR.parent.parent / "data" / "processed" / "lookup_tables"

# used to find ARPAbet vowels
_VOWEL_RE = re.compile(r"\d$")

# caches that hold tables
_AOA_LOOKUP: dict[str, float] | None = None
_CONC_LOOKUP: dict[str, float] | None = None

# loads a table into a dictionary
def _load_lookup(parquet_path: str | Path) -> dict[str, float]:
  df = pd.read_parquet(Path(parquet_path), columns=["word", "value"]).copy()
  if "word" not in df.columns or "value" not in df.columns:
    raise KeyError("Parquet must contain columns: 'word' and 'value'")

  return {
    str(w).strip().lower(): float(v)
    for w, v in zip(df["word"], df["value"])
    if pd.notna(w) and pd.notna(v)
  }

def aoa(word: str) -> float:
  global _AOA_LOOKUP
  if _AOA_LOOKUP is None:
    _AOA_LOOKUP = _load_lookup(TABLE_DIR / "aoa_table.parquet")
  
  if word is None:
    return -1.0
  key = str(word).strip().lower()
  return float(_AOA_LOOKUP.get(key, -1.0))

def conc(word: str) -> float:
  global _CONC_LOOKUP
  if _CONC_LOOKUP is None:
    _CONC_LOOKUP = _load_lookup(TABLE_DIR / "conc_table.parquet")
  
  if word is None:
    return -1.0
  key = str(word).strip().lower()
  return float(_CONC_LOOKUP.get(key, -1.0))

def freq(word: str) -> float:
  if not word:
    return -1.0
  
  w = str(word).strip().lower()

  f = word_frequency(w, "en", minimum=0.0)
  if f == 0.0:
    return -1.0
  
  # log10(per billion)
  return float(zipf_frequency(w, "en"))

def _is_vowel(phone: str) -> bool:
  return bool(_VOWEL_RE.search(phone))

# Converts a word to ARPAbet phonemes using CMUdict
def _arpabet_phones(text: str) -> list[str] | None:
  tokens = re.findall(r"[A-Za-z]+", str(text).lower())
  if not tokens:
    return None
  
  phones: list[str] = []
  for tok in tokens:
    opts = pronouncing.phones_for_word(tok)
    if not opts:
      return None

    # arbitrarily take first pronunciation
    phones.extend(opts[0].split())
  return phones

def _phon_features(text: str) -> dict[str, float]:
  phones = _arpabet_phones(text)
  if phones is None:
    return {
      "found": 0.0,
      "n_phonemes": -1.0,
      "n_syllables": -1.0,
      "onset_cluster": -1.0,
      "coda_cluster": -1.0,
      "max_cluster": -1.0
    }
  
  vowel_idxs = [i for i, p in enumerate(phones) if _is_vowel(p)]
  n_syllables = float(len(vowel_idxs))
  n_phonemes = float(len(phones))

  onset = 0
  if vowel_idxs:
    onset = vowel_idxs[0]
  coda = 0
  if vowel_idxs:
    coda = (len(phones) - 1) - vowel_idxs[-1]
  
  max_run = 0
  run = 0
  for p in phones:
    if _is_vowel(p):
      run = 0
    else:
      run += 1
      max_run = max(max_run, run)
  
  return {
    "found": 1.0,
    "n_phonemes": n_phonemes,
    "n_syllables": n_syllables,
    "onset_cluster": float(onset),
    "coda_cluster": float(coda),
    "max_cluster": float(max_run)
  }

@lru_cache(maxsize=100_000)
def phon(text: str) -> float:
  if not text:
    return -1.0

  text = str(text).strip().lower()

  feats = _phon_features(text)
  if feats["found"] == 0.0:
    return -1.0
  
  return (
    feats["n_phonemes"]
    + 0.5 * feats["n_syllables"]
    + 1.5 * feats["max_cluster"]
  )


def main():
  print(f'aoa("hello") -> {aoa("hello")}')
  print(f'aoa("dog") -> {aoa("dog")}')
  print(f'aoa("unforgiving") -> {aoa("unforgiving")}')

  print(f'conc("dog") -> {conc("dog")}')
  print(f'conc("bite the bullet") -> {conc("bite the bullet")}')

  print(f'freq("dog") -> {freq("dog")}') 
  print(f'freq("unforgiving") -> {freq("unforgiving")}')

  print(f'phon("dog") -> {phon("dog")}')
  print(f'phon("worcestershire") -> {phon("worcestershire")}')

if __name__ == "__main__":
  main()