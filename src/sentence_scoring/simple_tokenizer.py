"""
  Extracts word-like tokens.
  Keeps apostrophes except for 's, in which case it is removed.
  Assumes input is already lowercase with punctuation present.
"""

import re
from typing import List

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)

def simple_tokenize(sentence: str) -> List[str]:
  """
  Tokenize a sentence into word-like tokens.
  Example: "this is a sentence." -> ["this", "is", "a", "sentence"]
  """
  tokens = _TOKEN_RE.findall(sentence)
  return [t[:-2] if t.endswith("'s") else t for t in tokens]