
# Building Curricula

## Step-by-step Guide

1. Set up dependencies by following the instructions in:
    - `src/lexical_features/README.md`
    - `src/preprocessing/README.md`
    - `src/sentence_scoring/README.md`

2. Confirm that there is a corpus present in `data/processed/corpora/raw_shards` as outputted by running `scripts/build_corpus.py`
    - There should only be one corpus present in this location

3. Usage

```python
from curricula import build_curriculum

build_curriculum(
  curriculum: str = "shuffled" | "aoa" | "conc" | "freq" | "phon",
  scoring_method: str = "mean" | "min" | "max" | "add",
  sort_order: str = "asc" | "desc",
  tranche_type: str = "word-based" | "sentence-based",
  tranche_size: int = 500,  # either num words (word-based) or num sentences (sentence-based)
  aoa_agnostic: bool = True,
  multiword: bool = False,
  skip_stopwords: bool = False,
  inflect: bool = True
) -> pathlib.Path
```
- The curriculum is outputted into data/processed/corpora/training/NAME/, where NAME has an index and a list of parameters.
- This folder contains a `config.json` file with run configuration and basic statistics, a `tranches/` folder with `tranche_0001/`... folders containing `data.parquet` with columns `sentence` and `value`. For word-based tranches, `new_words.txt` within each tranche folder contains the 500 new words added with each tranche.

4. Details
- `aoa_agnostic` only applies when `curriculum` is one of `conc|freq|phon`. When True, score sentence directly using selected metric. When False, score based on max-AoA word in the sentence.

5. Shuffling tranches
- To shuffle tranches...

6. To get curriculum analytics like binning sizes...