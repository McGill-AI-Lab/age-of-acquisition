"""
Curriculum Optimizer for Age of Acquisition (AoA) based sentence ranking.

This script implements a two-tier curriculum learning pipeline using Bayesian Optimization
for outer-layer feature weighting and mean-direction ranking for inner subfeature matrices.
"""

import os
import shutil
import tempfile
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
from functools import lru_cache
import spacy
from spacy.tokens import Doc
import gc
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Real
from tqdm import tqdm
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")
TRAINING_DATA_PATH = DATA_DIR / "training_data" / "combined_training_data.parquet"
AOA_DATA_PATH = DATA_DIR / "AoA_words.xlsx"
OUTPUT_PATH = "curriculum_output.csv"
EMBEDDING_CACHE = CACHE_DIR / "sentence_embeddings.pt"

# Command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Curriculum Optimizer')
    parser.add_argument('--objective', type=str, default='cosinedrift',
                      choices=['cosinedrift'],
                      help='Objective function for optimization')
    parser.add_argument('--n-calls', type=int, default=30,
                      help='Number of Bayesian Optimization iterations')
    parser.add_argument('--top-k', type=int, default=500,
                      help='Number of top/bottom frequent words to use for drift computation')
    return parser.parse_args()

# Feature groups and their subfeatures
FEATURE_GROUPS = {
    "AoA": ["max_aoa", "avg_concreteness", "avg_word_frequency_external"],
    "Simplicity": [
        "lm_score", "char_lm_score", "sent_length", "verb_ratio", 
        "noun_ratio", "parse_depth", "avg_word_frequency_corpus", "num_noun_phrases", 
        "num_verb_phrases", "num_prep_phrases"
    ],
    "Diversity": [
        "num_types", "type_token_ratio", "entropy", 
        "simpsons_index", "quadratic_entropy"
    ]
}

@dataclass
class FeatureCache:
    """Manages caching of feature computations."""
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    
    def get_cache_path(self, feature_name: str) -> Path:
        """Get the cache file path for a feature."""
        return self.cache_dir / f"{feature_name}.csv"
    
    def is_cached(self, feature_name: str) -> bool:
        """Check if a feature is already cached."""
        return self.get_cache_path(feature_name).exists()
    
    def load_feature(self, feature_name: str) -> pd.Series:
        """Load a feature from cache."""
        cache_path = self.get_cache_path(feature_name)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found for feature: {feature_name}")
        df = pd.read_csv(cache_path)
        return df.set_index('sentence_id')['score']
    
    def save_feature(self, feature_name: str, scores: pd.Series):
        """Save a feature to cache."""
        cache_path = self.get_cache_path(feature_name)
        scores.rename('score').reset_index().to_csv(cache_path, index=False)

class CurriculumOptimizer:
    """Main class for curriculum optimization."""
    
    def __init__(self, use_gpu: bool = torch.cuda.is_available()):
        self.args = parse_args()
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.cache = FeatureCache()
        self.scaler = MinMaxScaler()
        
        # Initialize models
        logger.info("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Downloading...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        logger.info("Loading language models...")
        self.lm_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        self.lm_model = AutoModelForMaskedLM.from_pretrained("distilroberta-base").to(self.device)
        self.char_lm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        self.char_lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Load AoA and concreteness data
        logger.info("Loading AoA and concreteness data...")
        self.aoa_data = self._load_aoa_data()
        
        # Initialize data
        logger.info("Streaming training data to temporary file...")
        self.data = self._load_data()
        
        # Initialize word frequencies
        self._compute_word_frequencies()
    
    def _load_aoa_data(self) -> Dict[str, Dict[str, float]]:
        """
        Load AoA and concreteness data from Excel file.
        
        Returns:
            Dict mapping words to their AoA, concreteness, and frequency data.
            
        Raises:
            FileNotFoundError: If the AoA data file is not found.
            ValueError: If required columns are missing from the data.
        """
        if not AOA_DATA_PATH.exists():
            raise FileNotFoundError(f"AoA data file not found at {AOA_DATA_PATH}. "
                                 f"Please ensure the file exists and contains the required data.")
        
        try:
            # Read the Excel file
            aoa_df = pd.read_excel(AOA_DATA_PATH)
            
            # Map column names to expected format
            # Maps the expected internal name to the name found in the Excel file.
            column_map = {
                'word': 'Word',
                'aoa': 'AoA_Kup',
                'concreteness': 'conc.M',
                'frequency': 'freq'
            }
            
            # Rename columns to standard names
            aoa_df = aoa_df.rename(columns={v: k for k, v in column_map.items() if v in aoa_df.columns})
            
            # Process the data
            aoa_data = {}
            for _, row in aoa_df.iterrows():
                word = str(row['word']).strip().lower()
                if not word or pd.isna(row.get('word')):
                    continue
                    
                aoa_data[word] = {
                    'aoa': float(row['aoa']) if not pd.isna(row.get('aoa')) else 0.0,
                    'concreteness': float(row['concreteness']) if not pd.isna(row.get('concreteness')) else 0.0,
                    'frequency': float(row['frequency']) if not pd.isna(row.get('frequency')) else 0.0
                }
            
            logger.info(f"Loaded AoA data for {len(aoa_data)} words")
            return aoa_data
            
        except Exception as e:
            raise ValueError(f"Error loading AoA data: {str(e)}\n"
                           f"Available columns: {aoa_df.columns.tolist()}")
    
    def _load_data(self) -> Path:
        """
        Stream data from Parquet to a temporary text file to avoid loading into RAM.

        Returns:
            Path to the temporary file containing one sentence per line.
        """
        logger.info(f"Loading training data from {TRAINING_DATA_PATH}...")
        if not TRAINING_DATA_PATH.exists():
            raise FileNotFoundError(f"Source data file not found at {TRAINING_DATA_PATH}")

        # Create a temporary file to store sentences
        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False, suffix=".txt") as tmp_file:
            temp_path = Path(tmp_file.name)
            logger.info(f"Streaming sentences to temporary file: {temp_path}")

            try:
                parquet_file = pq.ParquetFile(TRAINING_DATA_PATH)
                num_rows = parquet_file.metadata.num_rows
                logger.info(f"Found {num_rows:,} records in the dataset.")

                # Load entire parquet file into pandas DataFrame
                # Use pyarrow to read in batches with a progress bar
                batch_size = 100_000
                num_batches = (parquet_file.metadata.num_rows + batch_size - 1) // batch_size

                batches = [
                    batch
                    for batch in tqdm(
                        parquet_file.iter_batches(batch_size=batch_size),
                        total=num_batches,
                        desc="Loading training data",
                    )
                ]
                table = pa.Table.from_batches(batches)
                df = table.to_pandas()
                logger.info(f"Loaded {len(df):,} sentences into DataFrame.")

                # Identify the text column
                schema_names = df.columns.tolist()
                text_columns = ['sentence', 'text', 'content', 'tokens']  # Added 'tokens' as it might contain text data
                text_col = next((col for col in text_columns if col in schema_names), None)

                if text_col is None and schema_names:
                    text_col = schema_names[0]
                    logger.info(f"Using column '{text_col}' as the text column.")

                if text_col is None:
                    raise ValueError("No suitable text column found in the input data.")

                # Extract sentences from the text column
                sentence_count = 0
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing sentences to temp file"):
                    sentence = row[text_col]

                    # Handle different data types
                    if isinstance(sentence, list):
                        # If it's a list of tokens, join them into a sentence
                        sentence = ' '.join(str(token) for token in sentence if token)
                    elif pd.isna(sentence):
                        continue  # Skip NaN values

                    # Ensure sentence is a string and write it
                    tmp_file.write(str(sentence).strip() + '\n')
                    sentence_count += 1

            except Exception as e:
                logger.error(f"Failed to load the parquet file: {e}")
                raise

        logger.info(f"Successfully streamed {sentence_count:,} sentences to {temp_path}")
        return temp_path
        
    def _count_lines(self, filepath: Path) -> int:
        """Efficiently count the number of lines in a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _compute_word_frequencies(self):
        """Compute word frequencies across the corpus."""
        from collections import defaultdict
        
        word_counts = defaultdict(int)
        # Get total number of sentences for a proper progress bar
        num_sentences = self._count_lines(self.data)

        with open(self.data, 'r', encoding='utf-8') as f:
            # Use nlp.pipe for efficient processing of a stream of texts
            for doc in tqdm(self.nlp.pipe(f, batch_size=1000), total=num_sentences, desc="Computing word frequencies"):
                for token in doc:
                    if token.is_alpha:
                        word_counts[token.text.lower()] += 1
        
        if not word_counts:
            raise ValueError("No valid words found in the input data")
                    
        self.word_freq = pd.Series(word_counts).sort_values(ascending=False)
        
        # Select top and bottom k words as drift words
        k = min(self.args.top_k, len(self.word_freq) // 2) if self.word_freq.size > 0 else 0
        if k > 0:
            self.drift_words = set(self.word_freq.head(k).index) | set(self.word_freq.tail(k).index)
        else:
            self.drift_words = set()
    
    def _get_sentence_embeddings(self, docs: List[Doc]) -> torch.Tensor:
        """Compute sentence embeddings using the language model."""
        if EMBEDDING_CACHE.exists():
            logger.info("Loading cached sentence embeddings...")
            return torch.load(EMBEDDING_CACHE)
            
        logger.info("Computing sentence embeddings (this may take a while)...")
        embeddings = []
        
        with torch.no_grad():
            for doc in tqdm(docs, desc="Computing embeddings"):
                inputs = self.lm_tokenizer(
                    doc.text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_attention_mask=True
                ).to(self.device)
                
                # Get hidden states
                outputs = self.lm_model(**inputs, output_hidden_states=True)
                
                # Use mean of last hidden state as sentence embedding
                last_hidden = outputs.hidden_states[-1]
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                mean_embedding = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
                embeddings.append(mean_embedding.cpu())
        
        # Stack all embeddings
        all_embeddings = torch.cat(embeddings, dim=0)
        
        # Save to cache
        torch.save(all_embeddings, EMBEDDING_CACHE)
        return all_embeddings
    
    def _compute_cosine_drift(self, weights: np.ndarray, features: pd.DataFrame) -> float:
        """Compute cosine drift metric for the given weights."""
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Compute weighted scores for each sentence
        scores = (
            weights[0] * features[FEATURE_GROUPS['AoA']].mean(axis=1) +
            weights[1] * features[FEATURE_GROUPS['Simplicity']].mean(axis=1) +
            weights[2] * features[FEATURE_GROUPS['Diversity']].mean(axis=1)
        )
        
        # Get indices of top and bottom scored sentences
        k = min(self.args.top_k, len(scores) // 2)
        top_indices = scores.nlargest(k).index
        bottom_indices = scores.nsmallest(k).index
        
        # Get embeddings for selected sentences
        with open(self.data, 'r', encoding='utf-8') as f:
            all_sentences = f.readlines()
        
        # This part still requires loading sentences for spacy, but it's more controlled.
        all_docs = list(self.nlp.pipe(all_sentences, batch_size=1000))
        embeddings = self._get_sentence_embeddings(all_docs) # This uses its own cache
        
        # Select embeddings for drift computation
        top_embeddings = embeddings[top_indices]
        bottom_embeddings = embeddings[bottom_indices]
        
        # Compute cosine drift
        sim_matrix = cosine_similarity(
            torch.cat([top_embeddings, bottom_embeddings]).numpy()
        )
        
        # Compute mean cosine similarity within and between groups
        n = len(top_embeddings)
        within_top = np.mean(sim_matrix[:n, :n][np.triu_indices(n, k=1)])
        within_bottom = np.mean(sim_matrix[n:, n:][np.triu_indices(n, k=1)])
        between = np.mean(sim_matrix[:n, n:])
        
        # Drift is the difference between within-group and between-group similarity
        drift = (within_top + within_bottom) / 2 - between
        return -drift  # Negative because we want to minimize drift
    
    def compute_all_features(self):
        """Compute all features for the dataset."""
        logger.info("Computing features...")

        # Get total number of sentences for a proper progress bar
        num_sentences = self._count_lines(self.data)

        # Process sentences with spaCy
        with open(self.data, 'r', encoding='utf-8') as f:
            # nlp.pipe can take an iterator, which is memory-efficient
            docs = list(self.nlp.pipe(tqdm(f, total=num_sentences, desc="Processing sentences"), batch_size=1000))
        
        # Compute features
        features = {}
        
        # AoA features
        features.update(self._compute_aoa_features(docs))
        
        # Simplicity features
        features.update(self._compute_simplicity_features(docs))
        
        # Diversity features
        features.update(self._compute_diversity_features(docs))
        
        # Combine all features
        feature_df = pd.DataFrame(features)
        
        # Normalize features
        normalized_features = pd.DataFrame(
            self.scaler.fit_transform(feature_df),
            columns=feature_df.columns,
            index=feature_df.index
        )
        
        return normalized_features
    
    def _compute_aoa_features(self, docs: List[Doc]) -> Dict[str, List[float]]:
        """Compute Age of Acquisition related features."""
        features = {
            'max_aoa': [],
            'avg_concreteness': [],
            'avg_word_frequency_external': []
        }
        
        for doc in docs:
            words = [token.text.lower() for token in doc if token.is_alpha]
            aoa_scores = []
            concreteness_scores = []
            freq_scores = []
            
            for word in words:
                if word in self.aoa_data:
                    aoa_scores.append(self.aoa_data[word]['aoa'])
                    concreteness_scores.append(self.aoa_data[word]['concreteness'])
                    freq_scores.append(self.aoa_data[word]['frequency'])
            
            features['max_aoa'].append(float(max(aoa_scores)) if aoa_scores else 0.0)
            features['avg_concreteness'].append(float(np.mean(concreteness_scores)) if concreteness_scores else 0.0)
            features['avg_word_frequency_external'].append(float(np.mean(freq_scores)) if freq_scores else 0.0)
        
        return features
    
    def _compute_simplicity_features(self, docs: List[Doc]) -> Dict[str, List[float]]:
        """Compute sentence simplicity features."""
        features = {
            'sent_length': [],
            'verb_ratio': [],
            'noun_ratio': [],
            'avg_word_frequency_corpus': [],
            'parse_depth': [],
            'num_noun_phrases': [],
            'num_verb_phrases': [],
            'num_prep_phrases': []
        }
        
        for doc in docs:
            # Basic sentence features
            features['sent_length'].append(len(doc))
            
            # POS-based features
            verbs = [token for token in doc if token.pos_ == 'VERB']
            nouns = [token for token in doc if token.pos_ == 'NOUN']
            features['verb_ratio'].append(len(verbs) / len(doc) if len(doc) > 0 else 0.0)
            features['noun_ratio'].append(len(nouns) / len(doc) if len(doc) > 0 else 0.0)

            # Corpus-based word frequency
            words = [token.text.lower() for token in doc if token.is_alpha]
            corpus_freqs = [self.word_freq.get(w, 0) for w in words]
            features['avg_word_frequency_corpus'].append(np.mean(corpus_freqs) if corpus_freqs else 0.0)
            
            # Parse tree depth
            if len(doc) > 0:
                features['parse_depth'].append(max([len(list(token.ancestors)) for token in doc]))
            else:
                features['parse_depth'].append(0)
            
            # Phrase counts
            features['num_noun_phrases'].append(len([chunk for chunk in doc.noun_chunks]))
            features['num_verb_phrases'].append(len([token for token in doc if token.pos_ == 'VERB']))
            features['num_prep_phrases'].append(len([token for token in doc if token.dep_ == 'prep']))
        
        # Add LM scores (computed separately as they're more expensive)
        features.update(self._compute_lm_scores(docs))
        
        return features
    
    def _compute_lm_scores(self, docs: List[Doc]) -> Dict[str, List[float]]:
        """Compute language model based scores with caching."""
        # Check cache first
        lm_scores = self._load_or_compute_lm_scores('lm_score', docs, self._compute_word_level_lm_scores)
        char_lm_scores = self._load_or_compute_lm_scores('char_lm_score', docs, self._compute_char_level_lm_scores)
        
        return {
            'lm_score': lm_scores,
            'char_lm_score': char_lm_scores
        }
    
    def _load_or_compute_lm_scores(self, score_type: str, docs: List[Doc], compute_fn) -> List[float]:
        """Load scores from cache or compute them if not available."""
        cache_file = f"{score_type}s.csv"
        cache_path = CACHE_DIR / cache_file
        
        # Try to load from cache
        if cache_path.exists():
            try:
                cached_scores = pd.read_csv(cache_path)
                if len(cached_scores) == len(docs):
                    logger.info(f"Loaded {score_type} from cache")
                    return cached_scores['score'].tolist()
            except Exception as e:
                logger.warning(f"Failed to load {score_type} cache: {e}")
        
        # Compute scores if not in cache
        logger.info(f"Computing {score_type} (this may take a while)...")
        scores = compute_fn(docs)
        
        # Save to cache
        try:
            pd.DataFrame({
                'sentence_id': range(len(scores)),
                'score': scores
            }).to_csv(cache_path, index=False)
        except Exception as e:
            logger.warning(f"Failed to cache {score_type} scores: {e}")
        
        return scores
    
    def _compute_word_level_lm_scores(self, docs: List[Doc]) -> List[float]:
        """Compute word-level language model scores using distilroberta."""
        scores = []
        
        with torch.no_grad():
            for doc in tqdm(docs, desc="Computing word-level LM scores"):
                try:
                    # Tokenize with special tokens and return PyTorch tensors
                    inputs = self.lm_tokenizer(doc.text, return_tensors="pt").to(self.device)
                    
                    # Get model outputs
                    outputs = self.lm_model(**inputs, labels=inputs["input_ids"])
                    
                    # Calculate average log probability per token
                    logits = outputs.logits
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    # Get the log probability of each token given its context
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()
                    
                    # Calculate loss manually to get per-token log probs
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                 shift_labels.view(-1))
                    
                    # Convert to probabilities and average (handling padding tokens)
                    valid_tokens = (shift_labels != self.lm_tokenizer.pad_token_id).float()
                    if valid_tokens.sum().item() > 0:
                        avg_log_prob = (-loss * valid_tokens).sum().item() / valid_tokens.sum().item()
                        scores.append(avg_log_prob)
                    else:
                        scores.append(0.0)
                        
                except Exception as e:
                    logger.warning(f"Error processing sentence: {e}")
                    scores.append(0.0)
        
        # Normalize scores to [0, 1] range
        if scores:
            min_score, max_score = min(scores), max(scores)
            if max_score > min_score:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        return scores
    
    def _compute_char_level_lm_scores(self, docs: List[Doc]) -> List[float]:
        """Compute character-level language model scores using GPT-2."""
        scores = []
        
        with torch.no_grad():
            for doc in tqdm(docs, desc="Computing char-level LM scores"):
                try:
                    # Convert text to character tokens
                    text = doc.text
                    if not text.strip():
                        scores.append(0.0)
                        continue
                        
                    # Tokenize at character level (using GPT-2's tokenizer for simplicity)
                    tokens = self.char_lm_tokenizer.encode(text, add_special_tokens=True)
                    if len(tokens) <= 1:  # Need at least 1 character to predict
                        scores.append(0.0)
                        continue
                        
                    # Prepare input and target
                    input_ids = torch.tensor([tokens[:-1]], device=self.device)
                    target_ids = torch.tensor([tokens[1:]], device=self.device)
                    
                    # Get model outputs
                    outputs = self.char_lm_model(input_ids, labels=target_ids)
                    
                    # Calculate average log probability per character
                    log_probs = -torch.nn.functional.cross_entropy(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        target_ids.view(-1),
                        reduction='none'
                    )
                    
                    # Average log prob per character, if any valid log probs exist
                    if log_probs.numel() > 0:
                        avg_log_prob = log_probs.mean().item()
                        scores.append(avg_log_prob)
                    else:
                        scores.append(0.0)
                    
                except Exception as e:
                    logger.warning(f"Error processing sentence (char-level): {e}")
                    scores.append(0.0)
        
        # Normalize scores to [0, 1] range
        if scores:
            min_score, max_score = min(scores), max(scores)
            if max_score > min_score:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        return scores
    
    def _compute_diversity_features(self, docs: List[Doc]) -> Dict[str, List[float]]:
        """Compute lexical diversity features."""
        features = {
            'num_types': [],
            'type_token_ratio': [],
            'entropy': [],
            'simpsons_index': [],
            'quadratic_entropy': []
        }
        
        for doc in docs:
            words = [token.text.lower() for token in doc if token.is_alpha]
            if not words:
                for feat in features:
                    features[feat].append(0.0)
                continue
                
            # Basic type/token counts
            types = set(words)
            type_count = len(types)
            token_count = len(words)
            
            features['num_types'].append(type_count)
            features['type_token_ratio'].append(type_count / token_count if token_count > 0 else 0)
            
            # Frequency distribution
            freq_dist = {}
            for word in words:
                freq_dist[word] = freq_dist.get(word, 0) + 1
            
            # Entropy
            probs = np.array(list(freq_dist.values())) / token_count
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features['entropy'].append(entropy)
            
            # Simpson's Index
            simpsons = np.sum((np.array(list(freq_dist.values())) / token_count) ** 2)
            features['simpsons_index'].append(simpsons)
            
            # Quadratic Entropy
            quadratic_entropy = 1 - simpsons
            features['quadratic_entropy'].append(quadratic_entropy)
        
        return features
    
    def optimize_weights(self, features: pd.DataFrame, n_calls: int = 30):
        """Optimize feature weights using Bayesian Optimization."""
        logger.info("Optimizing feature weights using cosine drift objective...")
        
        # Define search space for the three main feature groups
        space = [
            Real(0, 1, name='weight_aoa'),
            Real(0, 1, name='weight_simplicity'),
            Real(0, 1, name='weight_diversity')
        ]
        
        # Track optimization progress
        self.current_iteration = 0
        self.total_iterations = n_calls
        self.progress_bar = tqdm(total=n_calls, desc="Bayesian Optimization", 
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        # Define callback for progress updates
        def callback(res):
            self.current_iteration += 1
            self.progress_bar.update(1)
            self.progress_bar.set_postfix({
                'Best Drift': f"{-res.fun:.4f}",
                'Weights': f"AoA:{res.x[0]:.2f}, Simp:{res.x[1]:.2f}, Div:{res.x[2]:.2f}"
            })
        
        # Define objective function based on cosine drift
        def objective(params):
            try:
                # Compute cosine drift (minimize negative drift = maximize separation)
                drift = self._compute_cosine_drift(np.array(params), features)
                return drift
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return 0.0  # Return neutral value on error
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=42,
            n_initial_points=min(10, n_calls // 3),
            callback=callback,
            verbose=True
        )
        
        # Get best weights and normalize
        best_weights = np.array(result.x)
        best_weights = np.maximum(best_weights, 0)  # Ensure non-negative weights
        best_weights = best_weights / (np.sum(best_weights) + 1e-10)  # Normalize
        
        logger.info(f"Optimization complete. Best weights: {best_weights}")
        
        return {
            'AoA': float(best_weights[0]),
            'Simplicity': float(best_weights[1]),
            'Diversity': float(best_weights[2])
        }, result
    
    def run(self):
        """Run the full curriculum optimization pipeline."""
        # Compute or load features
        features = self.compute_all_features()
        
        # Optimize weights
        best_weights, _ = self.optimize_weights(features)
        
        # Print results
        print("\nOptimized Weights:")
        for feature, weight in best_weights.items():
            print(f"{feature}: {weight:.4f}")
        
        # Compute final scores
        aoa_score = features[FEATURE_GROUPS['AoA']].mean(axis=1)
        simplicity_score = features[FEATURE_GROUPS['Simplicity']].mean(axis=1)
        diversity_score = features[FEATURE_GROUPS['Diversity']].mean(axis=1)
        
        final_scores = (
            best_weights['AoA'] * aoa_score +
            best_weights['Simplicity'] * simplicity_score +
            best_weights['Diversity'] * diversity_score
        )
        
        # Add scores to data and sort
        with open(self.data, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f]
        
        # Create a DataFrame for sorting and saving
        ranked_df = pd.DataFrame({
            'sentence': sentences,
            'score': final_scores
        })
        ranked_df = ranked_df.sort_values('score', ascending=False)
        
        # Save results
        ranked_df.to_csv(OUTPUT_PATH, index=False)
        
        # Print top 10 sentences
        print("\nTop 10 Ranked Sentences:")
        for i, (_, row) in enumerate(ranked_df.head(10).iterrows(), 1):
            print(f"{i}. [{row['score']:.4f}] {row['sentence']}")
        
        print(f"\nFull curriculum saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    optimizer = CurriculumOptimizer()
    optimizer.run()
