# `intrinsic_metrics.py`

## Overview

Implements standard intrinsic evaluation benchmarks for word embeddings. It is designed to evaluate embedding quality across training runs, curricula, and incremental training stages.

The evaluator supports similarity, relatedness, and analogy-based benchmarks, and can be used either for single embedding snapshots or longitudinal evaluation over training tranches.

## Implemented Benchmarks

### Simlex-999

* Evaluates semantic similarity.
* Human judgement reflect similarity rather than general association (e.g. car-gasoline are associated but not similar, car-bicycle are both vehicles, so they show some similarity, and finally dog-wolf are very similar).
* Metric: Spearman rank correlation between cosine similarity and human ratings.

#### Requirements

* File: `SimLex-999.txt`
* Format: TSV
* Required columns:
  * `word1`
  * `word2`
  * `SimLex999`
* Source: https://fh295.github.io/simlex.html

### WordSim-353

* Evaluates semantic relatedness.
* Includes both similarity and associative relations.
* Metric: Spearman rank correlation between cosine similarity and human ratings.

#### Requirements

* File: `wordsim353.csv`
* Format: CSV
* Required columns:
  * `Word1`
  * `Word2`
  * `Human (mean)`
* Source: https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd

### Google Analogy Task

* Evaluates syntactic and semantic regularities.
* Uses vector arithmetic with the 3CosAdd method.
* Metric: accuracy (exact match).

#### Requirements

* File: `questions-words.txt`
* Format: 
  * Category headers prefixed with `:`
  * Analogy format: `word1 word2 word3 word4`
* Source: https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt

## Naming Guidelines

Embedding files must follow this format: `{curriculum}_tranche{N}_run{R}_dim{D}.txt`, where 
* `curriculum` is the training regime or the ordering strategy (e.g. `aoa`, `random`, `frequency`, ...)
* `N` is the training stage/corpus subset index (integer)
* `R` is the random seed/training repetition index (integer)
* `D` is the embedding dimentionality (integer)
* Example: `aoa_tranche20_run1_dim50.txt`