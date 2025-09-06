"""
This file scores all parquet shards inside data/corpora/,
outputting {id, dataset, concreteness_score} where id = {dataset}-{sentenceIndex} as parquets into data/outputs/ordering
"""
from __future__ import annotations

from create_concreteness_lookup import load_concreteness_word_ratings
SCORE_MAP = load_concreteness_word_ratings()

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

import pandas as pd
import numpy as np

from typing import Iterable, List, Mapping, Optional, Tuple
import re
from pathlib import Path
script_path = Path(__file__).resolve()
script_dir = script_path.parent

# Words we've already tried and failed to map (even after lemmatization)
# Speeds up by ~ 40%
unknown_cache: set[str] = set()

def get_lemmas(tokens: List[str]) -> List[str]:
  """
  Lemmatizes an entire sentence
  """
  lemmatized_tokens = []
  tagged_tokens = pos_tag(tokens)
  for word, tag in tagged_tokens:
    lemmatized_tokens.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
  return lemmatized_tokens

def get_wordnet_pos(tag):
  if tag.startswith('J'):  
    return 'a'
  elif tag.startswith('V'):  
    return 'v'
  elif tag.startswith('N'):  
    return 'n'
  elif tag.startswith('R'):  
    return 'r'
  else:
    return 'n'


def score(sentence: List, method: str = "min", lemmatize: bool = False) -> float:
  """
  Scores a sentence by concreteness score
  method can be "min", "mean", or "max"

  Unknown words are skipped over

  If sentence is empty or none of the words are known, returns -1
  """
  scores = []
  if lemmatize:
    lemmatized_sentence = []
  for i, token in enumerate(sentence):
    word = token.lower().strip()
    if word not in SCORE_MAP:
      if lemmatize:
        if word in unknown_cache:
          continue
        if len(lemmatized_sentence) == 0:
          lemmatized_sentence = get_lemmas([w.lower().strip() for w in sentence])
        new_word = lemmatized_sentence[i]
        if new_word in SCORE_MAP:
          scores.append(SCORE_MAP[new_word])
        else:
          unknown_cache.add(word)
          continue
      continue
    scores.append(SCORE_MAP[word])
  if len(scores) == 0:
    return -1.0
  if method == "min":
    return min(scores)
  if method == "mean":
    return sum(scores) / len(scores)
  if method == "max":
    return max(scores)
  raise ValueError("Method should be one of 'min', 'mean', or 'max'")

def score_tokens_column(
  token_seqs: Iterable[List[str]],
) -> np.ndarray:
  """
  Compute concreteness scores
  Returns a NumPy array of float scores aligned to token_seqs
  """
  out = np.empty(len(token_seqs), dtype=np.float64)
  for i, tokens in enumerate(token_seqs):
    if len(tokens) == 0:
      out[i] = -1.0
    else:
      # score() already returns -1.0 if no words are known in the sentence
      out[i] = score(tokens, "min", True)
  return out

def parse_name(
  path: Path = Path(script_dir, "data/corpora/refined-bookcorpus-dataset")
):
  """
  Extract dataset name and zero-padded part number from filenames like:
  refined-bookcorpus-dataset_tokens_part-000.parquet
  Returns: (dataset_name, parquet_id)
  """
  m = re.search(r"(.+?)_tokens_part-(\d+)\.parquet$", path.name)
  if not m:
    raise ValueError(f"Unexpected filename format: {path.name}")
  dataset, part = m.group(1), m.group(2).zfill(3)
  return dataset, part

def process_one_parquet(
  parquet_path: Path,
  output_dir: Path = Path(script_dir, "data/outputs/ordered_data"),
  tokens_column: str = "tokens"
):
  """
  Read a single parquet shard, computer sentence scores, and write parquet with index "PPP-<rowIndex>" and columns: dataset, concreteness_score
  """
  dataset, part = parse_name(parquet_path)
  output_dir.mkdir(parents=True, exist_ok=True)

  # Read in parquet
  df = pd.read_parquet(parquet_path)
  n = len(df)
  positional_idx = np.arange(n, dtype=np.int64)

  # Computer scores
  scores = score_tokens_column(df[tokens_column].tolist())

  shard_id = part
  new_index = pd.Index([f"{shard_id}-{i}" for i in positional_idx], name="id")
  out_df = pd.DataFrame(
    {
      "dataset": dataset,
      "concreteness_score": scores,
    },
    index=new_index
  )

  out_name = f"{dataset}-{part}.parquet"
  out_path = output_dir / out_name
  out_df.to_parquet(out_path, index=True)
  return out_path

# Testing
if __name__ == "__main__":
  process_one_parquet(
    Path("C:/Users/igiff/OneDrive/Desktop/Age of Acquisition/age-of-acquisition/concreteness_curriculum/data/corpora/refined-bookcorpus-dataset/refined-bookcorpus-dataset_tokens_part-006.parquet"),
  )


  df = pd.read_parquet(Path(script_dir, "data/outputs/ordered_data/refined-bookcorpus-dataset-006.parquet"))
  print(df.head())

  df2 = pd.read_parquet(Path(script_dir, "data/corpora/refined-bookcorpus-dataset/refined-bookcorpus-dataset_tokens_part-006.parquet"))
  pd.set_option('display.max_colwidth', None)
  print(df2.head())