# Preprocessing the Corpus

## Step-by-step guide

1. Download the following, rename, and place into `data/raw/corpora`:
    <table><tr>
    <th> Link to download </th> <th> Rename file to </th>
    </tr><tr><td>

    * https://osf.io/ryjfm/files/ywea7
    * https://www.kaggle.com/datasets/nishantsingh96/refined-bookcorpus-dataset/data

    </td><td>

    * BabyLM/* (extract train_100M.zip and place all *.train files into BabyLM folder)
    * RefinedBookCorpus.csv

    </td></tr></table>

    File structure:
    ```
    data/raw/corpora/
    ├── BabyLM/
    │   ├── bnc_spoken.train 
    │   └── ... 
    └── RefinedBookCorpus.csv
    ```

2. Make sure all packages in pyproject.toml are installed (including lexical_features)
    ```cmd
    .../age-of-acquisition> pip install -e .
    ```

2. Usage
    ```python
    from preprocessing import *

    # Convert BabyLM and % of Refined Book Corpus into shards (default: n_shards=100, percent_refined=10)
    # Output in data/processed/corpora/raw_shards
    # May take a while; refined book corpus is being split by sentence using spacy
    load_corpus_shards(n_shards=100, percent_refined=10)
    ```

3. Notes
    - All words are lowercase
    - Each line is a sentence
    - There is punctuation