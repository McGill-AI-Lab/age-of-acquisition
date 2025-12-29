# Sentence Scoring

## Step-by-step Guide
1. Follow instructions in `src/lexical_features/README.md` to set up lexical norms

2. Make sure all packages in `pyproject.toml` are installed (including lexical_features)
    ```cmd
    .../age-of-acquisition> pip install -e .
    ```

## Parameters
- `sentence: str`
  - Lowercased input string; punctuation may be present.
  - Can be lines directly from corpus.
- `metric: str`
  - One of: "aoa", "conc", "phon", "freq".
- `multiword: bool`
  - If True and metric == "conc", attempt greedy multiword matching (longest-first, left-to-right). Otherwise score token-by-token.
- `method: str`:
  - Aggregation method:
    - "mean": mean of eligible units, witness = ""
    - "max": max eligible unit score, witness = unit
    - "min": min eligible unit score, witness = unit
    - "add": sum of eligible units, witness = ""
- `skip_stopwords: bool`
  - If True, stopwords are ignored for scoring except:
    - In (metric="conc", multiword=True), stopwords are still allowed *inside* expressions and may begin an expression; but if no expression begins at a stopword, that stopword is ignored alone.
- `inflect: bool`:
  - If True, pass inflect=True into aoa() and conc(). Ignored for phon() and freq().

## Returns
  - (score, witness):
    - score = -1 if no eligible units were found, else aggregated score.
    - witness = unit for max/min, else "".

## Examples:
  ```python
  from sentence_scoring import *
  from lexical_features import *

  # scoring sentence based on mean concreteness while attempting multiwords, skipping stopwords, and using the inflected lookup table
  output = score_sentence(
    "this is a sentence.",
    metric="conc",
    multiword=True,
    method="mean",
    skip_stopwords=True,
    inflect=True
  )
  s = output[0] # get just the score
  
  # scoring sentence based on concreteness of max-aoa word
  output = score_sentence(
    "this is another sentence",
    metric="aoa",
    method="max",
    skip_stopwords=True,
    inflect=True
  )
  word = output[1] # get just max-aoa word
  s = conc(word)
  ```

## Tests:
- There are 2 basic testing scripts that test the basic functionality of sentence scoring.
- `test_sentence_scoring.py` and `test_sentence_scoring2.py`