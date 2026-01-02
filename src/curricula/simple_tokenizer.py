"""
  Only used when outputting list of words in each tranche for word-based tranches.
"""

from __future__ import annotations
import re
from typing import List

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize_words_lower_already(text: str) -> List[str]:
    """
    Tokenize lowercase corpus text into word tokens.
    Regex: [a-z]+(?:'[a-z]+)?
    """
    return _WORD_RE.findall(text)
