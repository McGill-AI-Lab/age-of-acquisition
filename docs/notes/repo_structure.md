# Repo Structure

## Root-level files

### `README.md`
- Project motivation and research question
- High-level pipeline overview
- Installation instructions
- Minimal examples
- Citation information

### `environment.yml` (conda env)
- Python version
- Core scientific and ML dependencies
- System-level libs 

### `pyproject.toml`
- Python packaging and tooling configuration
- Project metadata
- Optional dependency groups like dev, experiments, ...

### `.gitignore`
- Put `data/raw/`, `outputs/`
- Logs, caches, temporary files
- IDE-specific files

## Documents - `docs/` 

### `docs/paper/`
- Materials for the paper
- `draft.md`: Main text
- `figures/`: Figures used in the paper
- `tables/`: Tables for results or methods

### `docs/diagrams/`
- Conceptual explanations of the system
- `tranche_diagram.py`: Diagram of how we split the tranches
- `curriculum_overview.md`: Explanation of curriculum logic

### `docs/notes/`
- `notes.md`: Notes, assumptions, etc.
- `open_questions.md`: Known uncertainties and future directions

## Data - `data/`

### `data/raw/`
- Original, untouched data

#### `corpora/`
- Raw text sources

#### `lexical_norms/`
- External lexical resources like Kuperman 

### `data/processed/`
- Outputs of preprocessing

#### `corpora/`
- `cleaned/`: Normalized text
- `deduplicated/`: Filtered corpora
- `shards/`: Chunked corpora for streaming or training

#### `lookup_tables/`
- Standardized lexical features
- Used by scoring and curriculum modules

### `data/samples/`
- Debugging and demos

## Importable code -  `src/`

### `src/preprocessing/`
- Corpus cleaning and preparation, including:
- Text normalization
- Deduplication
- Sentence filtering
- Sharding

### `src/lexical_features/`
- One file per feature (AoA, frequency, phonology, etc.)
- Functions mapping word → value
- Feature standardization utilities

### `src/sentence_scoring/`
- Token scoring
- Sentence aggregation
- Coherence measures
- (Input: tokens, Output: sentence-level scores)

### `src/curricula/`
- Define sentence ordering strategies
- One file per curriculum

### `src/training/`
- Train embeddings on curricula
- `tranche_builder.py`: Split corpora into tranches
- `word2vec_train.py`: Word2Vec training logic
- `fasttext_train.py`: FastText training logic
- `contextual_embeddings.py`: Contextual models

### `src/evaluation/`
- Measure embedding quality

#### `stability/`
- Stability

#### `intrinsic/`
- Similarity, analogy benchmarks

#### `extrinsic/`
- Downstream NLP tasks

### `src/statistics/`
- Statistical analysis of results, including:
- Mixed-effects models
- Multicollinearity diagnostics
- Residual analysis

### `src/utils/`
- Shared helpers, such as:
- IO utilities
- Streaming utilities
- Logging
- Constants

### `src/curriculum_optimization`
- Aaron's Bayesian optimization files

## Running the experiments - `experiments/` 

### `experiments/configs/`
- Configurations, like model hyperparameters and curriculum defs

### `experiments/runs/`
- Metadata and logs for completed runs

## Results of experimentations - `outputs/` 

### `outputs/embeddings/`
- Trained embedding files

### `outputs/metrics/`
- Evaluation outputs

### `outputs/figures/`
- Generated plots

### `outputs/tables/`
- Tables with results

## Actual entry point of the experiments - `scripts/` 

- Download data
- Build lookup tables
- Score sentences
- Build curricula
- Train all models
- Evaluate all models