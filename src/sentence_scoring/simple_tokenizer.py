"""
  Extracts word-like tokens and keeps apostrophes inside words.
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
  return _TOKEN_RE.findall(sentence)