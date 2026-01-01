from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

from sentence_scoring.stopwords import STOPWORDS
from sentence_scoring.simple_tokenizer import simple_tokenize 

from lexical_features import *

Metric = Literal["aoa", "conc", "phon", "freq"]
Method = Literal["mean", "max", "min", "add"]

@dataclass(frozen=True)
class ScoredUnit:
  unit: str
  score: float

# scores a token or expression
def _score_unit(unit: str, metric: Metric, inflect: bool) -> float:
  if metric == "aoa":
    return aoa(unit, inflect=True) if inflect else aoa(unit)
  if metric == "conc":
    return conc(unit, inflect=True) if inflect else conc(unit)
  if metric == "phon":
    return phon(unit)
  if metric == "freq":
    return freq(unit)
  raise ValueError(f"Unknown metric: {metric!r}")

# aggregates eligible scored units
# if no eligible units -> (-1, "")
# mean -> (mean_score, "")
# max/min -> (extreme_score, witness_unit)
def _aggregate(scored: List[ScoredUnit], method: Method) -> Tuple[float, str]:
  if not scored:
    return -1.0, ""
  
  if method in ("mean", "add"):
    total = 0.0
    for s in scored:
      total += s.score
    if method == "add":
      return total, ""
    return total / float(len(scored)), ""
  
  if method == "max":
    best = scored[0]
    for s in scored[1:]:
      if s.score > best.score:
        best = s
    return best.score, best.unit

  if method == "min":
    best = scored[0]
    for s in scored[1:]:
      if s.score < best.score:
        best = s
    return best.score, best.unit

  raise ValueError(f"Unknown method: {method!r}")

# score each token independently (no multiword matching)
def _score_tokens_simple(
  tokens: List[str],
  metric: Metric,
  method: Method,
  skip_stopwords: bool,
  inflect: bool,
) -> Tuple[float, str]:
  scored: List[ScoredUnit] = []
  for tok in tokens:
    if skip_stopwords and tok in STOPWORDS:
      continue
    s = _score_unit(tok, metric, inflect)
    if s != -1:
      scored.append(ScoredUnit(tok, float(s)))
  return _aggregate(scored, method)

# concreteness scoring with greedy left-to-right longest-first multiword matching
# stopwords are allowed in expressions and may begin an expression
# if skip_stopwords=True and no expression is found starting at a stopword, ignore the lone stopword
def _score_tokens_conc_multiword(
  tokens: List[str],
  method: Method,
  skip_stopwords: bool,
  inflect: bool,
) -> Tuple[float, str]:
  scored: List[ScoredUnit] = []
  n = len(tokens)
  i = 0

  while i < n:
    # Try longest-first multiword expression starting at i
    matched_unit: Optional[str] = None
    matched_score: Optional[float] = None
    matched_end: Optional[int] = None  # inclusive end index

    for j in range(n - 1, i - 1, -1):
      expr = " ".join(tokens[i : j + 1])
      s = _score_unit(expr, "conc", inflect)
      if s != -1:
        matched_unit = expr
        matched_score = float(s)
        matched_end = j
        break
    
    # don't score lone stopwords
    if skip_stopwords and matched_unit in STOPWORDS and " " not in matched_unit:
      matched_unit = None
      matched_score = None
      matched_end = None
    
    if matched_unit is not None and matched_score is not None and matched_end is not None:
      scored.append(ScoredUnit(matched_unit, matched_score))
      i = matched_end + 1
      continue
    
    # no expression found starting at i
    tok = tokens[i]

    # don't score lone stopwords
    if skip_stopwords and tok in STOPWORDS:
      i += 1
      continue

    s = _score_unit(tok, "conc", inflect)
    if s != -1:
      scored.append(ScoredUnit(tok, float(s)))
    i += 1
  
  return _aggregate(scored, method)

# sentence scoring API
def score_sentence(
  sentence: str,
  metric: Metric = "aoa",
  multiword: bool = False,
  method: Method = "mean",
  skip_stopwords: bool = False,
  inflect: bool = False,
) -> Tuple[float, str]:
  if metric not in ("aoa", "conc", "phon", "freq"):
    raise ValueError(f"metric must be one of 'aoa','conc','phon','freq'; got {metric!r}")
  if method not in ("mean", "max", "min", "add"):
    raise ValueError(f"method must be one of 'mean','max','min','add'; got {method!r}")

  tokens = simple_tokenize(sentence)
  if not tokens:
    return -1.0, ""

  if metric == "conc" and multiword:
    return _score_tokens_conc_multiword(tokens, method, skip_stopwords, inflect)
  
  return _score_tokens_simple(tokens, metric, method, skip_stopwords, inflect)