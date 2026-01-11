
# Building Curricula

## Step-by-step Guide

1. Set up dependencies by following the instructions in:
    - `src/lexical_features/README.md`
    - `src/preprocessing/README.md`

2. Confirm that there is a corpus present in `data/processed/corpora/raw_shards` as outputted by running `scripts/build_corpus.py`
    - There should only be one corpus present in this location

3. Usage
    - The curriculum is outputted into `data/processed/corpora/training/NAME/`, where NAME has an index and a list of parameters.
    - May look something like: `000_c=aoa_m=max_o=asc_t=sb_s=20000_ag=1_mw=0_ss=1_in=1`
    - This folder contains a `config.json` file with run configuration and basic statistics, a `tranches/` folder with `tranche_0001/`... folders containing `data.parquet` with columns `sentence` and `value`. For word-based tranches, `new_words.txt` within each tranche folder contains the 500 new words added with each tranche.
    - For more details about certain parameters, see `src/sentence_scoring/README.md`

    ```python
    # gives build_curriculum, plot_tranche_sizes, write_stamples, and shuffle_tranches
    from curricula import * 

    idx: int = build_curriculum(
      curriculum: str = "shuffled" | "aoa" | "conc" | "freq" | "phon",
      scoring_method: str = "mean" | "min" | "max" | "add",
      sort_order: str = "asc" | "desc",
      tranche_type: str = "word-based" | "sentence-based" | "matching",
      tranche_size: int = 500,  # either num words (word-based) or num sentences (sentence-based)
      aoa_agnostic: bool = True, # when false, use max-aoa word to score
      multiword: bool = False,
      skip_stopwords: bool = False,
      inflect: bool = True,
      duplication_cap: int = -1
    )

    # builds a shuffled curriculum with matching number of words as another curriculum
    idx2: int = build_curriculum(
        curriculum="shuffled",
        tranche_type="matching",
        matching_idx=idx,
        duplication_cap=5,
    )

    # before training, shuffle within tranches in-place
    shuffle_tranches(idx) # input string is curriculum index
    shuffle_tranches(idx2) 
    ```

4. Analytics
    - There are two useful functions for validating the curriculum.
    - `plot_tranche_sizes` displays a graph with tranche number against number of sentences or words
    - `write_samples` writes `samples.txt` into the curriculum folder. Each line is the first sentence from each tranche and its score.
    ```python
    from curricula import plot_tranche_sizes
    from curricula import write_samples

    # input string is curriculum index
    # all output automatically saved in data/processed/corpora/training/<curriculum_name>
    plot_tranche_sizes("1", metric="sentence") 
    plot_tranche_sizes("1", metric="word", show=True) # automatically show graph on screen (still saves)
    plot_write_samples("1") 
    ```

4. Extra Details
    - `aoa_agnostic` only applies when `curriculum` is one of `conc|freq|phon`. When True, score sentence directly using selected metric. When False, score based on max-AoA word in the sentence.
    - `duplication_cap` caps the number of duplicated, normalized sentences. Its default value is -1, in which case no duplicates are removed.
    - `tranche_type="matching"` builds tranches based on the number of words in each tranche in a specified curriculum, given by `matching_idx`.
    - Do not rename curriculum folder names! They are used by plotting, sampling, and shuffling.