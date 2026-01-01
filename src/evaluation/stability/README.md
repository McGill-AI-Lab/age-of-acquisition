# `stability_metrics.py`

## Overview

Measures the stability of word embeddings across multiple training runs and over time. The core idea of stability here is that if you train the same embedding model multiple times with different random seeds, do words end up in similar semantic neighborhoods?

Stability is measured with nearest-neighbor overlap:
* For a given word, look at its `k` nearest neighbors in two different runs.
* Compute how many neighbors are shared.
* Average this overlap across words and across run pairs.

This code also supports:
* Tracking stability across incremental training tranches.
* Comparing different curricula, such as shuffled AoA.
* Measuring how much words move in vector space after they are introduced.

## Packages

* `numpy` is used for numerical aggregation such as mean and standard deviation.
* `pandas` is used to return results as DataFrames, which are convenient for analysis and plotting.
* `gensim.models.KeyedVectors` is Gensim's structure for storing word vectors, containing a vocabulary, vector representations, and methods like `most_similar`.
* `scipy.spatial.distance.cosine` is used to compute cosine distance between two vectors and to measure how much a word vector moves over time.

## Metrics

### 1. Nearest-Neighbor Overlap (Pairwise Stability)

Nearest-neighbor overlap measures how similar a word’s local semantic neighborhood is across independent training runs.

For a given word $w$:
* Retrieve its $k$ nearest neighbors in run $i$.
* Retrieve its $k$ nearest neighbors in run $j$.
* Compute the proportion of shared neighbors.

Formally, for neighbors sets $N_i(w)$ and $N_j(w)$:
$$\text{overlap}(w) = \frac{|N_i(w) \cap N_j(w)|}{k}$$
This value ranges from 0 to 1.

This metric captures local representational stability:
* High overlap means that the word consistently occupies a similar semantic region across runs.
* Low overlap means that the word's semantic neighborhood is sensitive to stochastic factors, such as initialization or data order.
* This metric is invariant to global rotations of the embedding space because it evaluates relative semantic structure.

### 2. Pairwise Stability Across Runs

Pairwise stability aggregates nearest-neighbor overlap:
* Across all words in a shared vocabulary.
* Across all unique pairs of training runs.

For each pair of runs:
$$\text{stability}_{i, j} = \frac{1}{|V|} \Sigma_{w \in V} \text{overlap}_{i, j}(w)$$

This metric captures global stability of the embedding space:
* It reflects whether the model converges to a consistent solution when trained multiple times under identical conditions.
* It quantifies robustness to randomness in training.
* If a curriculum indices stronger structural constraints on learning, then independent runs should converge to similar semantic organizations.

### 3. Stability Over Training Time (Tranche-wise Stability)

Stability over time tracks how pairwise stability evolves across incremental training tranches.

For each tranche:
* Embeddings from all runs at that tranche are compared.
* Pairwise stability is computed as described above.

The result is a trajectory of stability values across training time.

This metric captures when semantic structure stabilizes during training:
* Early tranches may show low stability due to sparse data.
* Later tranches may show increasing stability as representations consolidate.

Different curricula may not only differ in final stability, but also in the rate of convergence and the timing of representational stabilization. This is especially relevant for AoA hypotheses, where early exposure may shape later semantic organization.

### 4. Vocabulary-Restricted Stability

Stability can optionally be computed on a restricted vocabulary subset, such as:
* Words from a psycholinguistic benchmark
* Words introduced before a certain tranche
* Frequency or AoA-controlled subsets.
* etc.

The same overlap metrics are applied, but only for the selected words.

This allows testing whether stability effects are driven by a small subset of frequent or early-acquired words, or generalized across the lexicon. Embedding stability may not be uniform across words, so restricting the vocabulary enables targeted hypothesis training, for example:
* Are early-acquired words more stable than late-acquired words?
* Do rare words exhibit greater instability?

### 5. Word Cohort Movement (Semantic Drift)

Word cohort movement measures how much the embedding vector of a word moves over time after it is introduced.

For each word:
$$\text{movement}_t = 1 - \cos(v_{t - 1}, v_t)$$
This is computed between consecutive tranches.

This metric captures semantic drift:
* Large movement indicates that a word's representation is still being reshaped by later input.
* Small movement indicates that the word has stabilized.
By grouping words by their introduction tranche, the metric tracks cohort-level dynamics, which directly tests a core AoA hypothesis: early-acquired words may become anchors in the semantic space.