
# Building Curricula

## Step-by-step Guide

1. Set up dependencies by following the instructions in:
    - `src/lexical_features/README.md`
    - `src/preprocessing/README.md`

2. Confirm that there is a corpus present in `data/processed/corpora/raw_shards` as outputted by running `scripts/build_corpus.py`
    - There should only be one corpus present in this location

3. Usage
    - The curriculum is outputted into `data/processed/corpora/training/NAME/`, where NAME has an index and a list of parameters.
    - This folder contains a `config.json` file with run configuration and basic statistics, a `tranches/` folder with `tranche_0001/`... folders containing `data.parquet` with columns `sentence` and `value`. For word-based tranches, `new_words.txt` within each tranche folder contains the 500 new words added with each tranche.
    - For more details about certain parameters, see `src/sentence_scoring/README.md`

    ```python
    from curricula import build_curriculum
    from curricula import shuffle_tranches

    build_curriculum(
      curriculum: str = "shuffled" | "aoa" | "conc" | "freq" | "phon",
      scoring_method: str = "mean" | "min" | "max" | "add",
      sort_order: str = "asc" | "desc",
      tranche_type: str = "word-based" | "sentence-based",
      tranche_size: int = 500,  # either num words (word-based) or num sentences (sentence-based)
      aoa_agnostic: bool = True, # when false, use max-aoa word to score
      multiword: bool = False,
      skip_stopwords: bool = False,
      inflect: bool = True
    ) -> pathlib.Path

    # before training, shuffle within tranches in-place
    shuffle_tranches("0") # input string is curriculum index
    ```

4. Analytics
    - There are two useful functions for validating the curriculum.
    - `plot_tranche_sizes` displays a graph with tranche number against number of sentences
    - `write_samples` writes `samples.txt` into the curriculum folder. Each line is the first sentence from each tranche and its score.
    ```python
    from curricula import plot_tranche_sizes
    from curricula import write_samples

    # input string is curriculum index
    plot_tranche_sizes("1") # output on screen
    plot_write_samples("1") # output in data/processed/corpora/training/NAME
    ```

4. Extra Details
    - `aoa_agnostic` only applies when `curriculum` is one of `conc|freq|phon`. When True, score sentence directly using selected metric. When False, score based on max-AoA word in the sentence.
    - Do not rename curriculum folder names! They are used by plotting, sampling, and shuffling.

To do:
- preprocess babylm
- apostrophes
- check over all files and readme's