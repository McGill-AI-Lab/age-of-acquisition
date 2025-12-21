"""
scoring.py - Feature scoring functions for AoA, frequency, concreteness, and phonetic complexity.

This module computes linguistic features for each **sentence* in the corpus.

Note: All scoring functions should handle errors gracefully and return np.nan
for sentences that cannot be scored.
"""
from typing import Dict

import numpy as np


def compute_aoa(text: str) -> float:
    """
    Compute Age of Acquisition score for a sentence.
    
    Age of Acquisition (AoA) represents the average age at which words in a
    sentence are typically learned. Lower values indicate earlier acquisition.
    
    Must decide if mean or max AoA per sentence.
    """
    
    # Tokenize by splitting on whitespace (simple approach)
    # In production, use a proper tokenizer (e.g., spaCy, NLTK)
    words = text.lower().split()
    
    # Handle empty sentences
    if not words:
        return np.nan

    # PLACEHOLDER
    word_aoas = np.random.uniform(3, 12, len(words))
    
    # Aggregate: return mean AoA across all words
    return float(np.mean(word_aoas))


def compute_frequency(text: str) -> float:
    """
    Compute frequency score for a sentence.
    
    Frequency measures how common words are in the language. Higher values
    indicate more frequent (common) words.
    """
    # Tokenize by splitting on whitespace
    words = text.lower().split()
    
    # Handle empty sentences
    if not words:
        return np.nan
    
    # PLACEHOLDER
    word_freqs = np.random.uniform(2, 10, len(words))
    
    # Aggregate: return mean log frequency
    return float(np.mean(word_freqs))


def compute_concreteness(text: str) -> float:
    """
    Compute concreteness score for a sentence.
    
    Concreteness measures how much a word refers to physical, tangible things
    versus abstract concepts. Higher values indicate more concrete meanings.
    """
    # Tokenize by splitting on whitespace
    words = text.lower().split()
    
    # Handle empty sentences
    if not words:
        return np.nan
    
    # PLACEHOLDER
    word_concs = np.random.uniform(1, 5, len(words))
    
    # Aggregate: return mean concreteness
    return float(np.mean(word_concs))


def compute_phonetic_complexity(text: str) -> float:
    """
    Compute phonetic complexity score for a sentence.
    
    Phonetic complexity measures how difficult a sentence is to pronounce.
    This can be based on number of phonemes, syllables, or articulatory features.
    """
    # Tokenize by splitting on whitespace
    words = text.lower().split()
    
    # Handle empty sentences
    if not words:
        return np.nan
    
    # PLACEHOLDER
    word_complexities = np.random.uniform(5, 20, len(words))
    
    # Aggregate: return sum of complexities across sentence
    # Other options: np.mean (average per word), np.max (hardest word)
    return float(np.sum(word_complexities))


def score_batch(text_batch: list[str]) -> Dict[str, np.ndarray]:
    """
    Apply all four scorers to a batch of sentences.
    
    This is the main entry point for scoring. It processes an entire batch
    of sentences and returns all four feature scores for each sentence.
    
    Error handling: If any scorer fails for a sentence, that specific score
    is set to np.nan, but other scores are still computed. This ensures one
    failing scorer doesn't break the entire pipeline.
    
    Args:
        text_batch: List of sentences to score
                   Can be any length (typically 100K-200K for efficiency)
    
    Returns:
        Dictionary with keys "aoa", "freq", "conc", "phon"
        Each value is a numpy array of shape (n,) where n = len(text_batch)
        Arrays contain float64 values (np.nan for scoring failures)
    
    Note:
        Does not filter out bad scores - keeps array length equal to input length.
        This maintains alignment with sentence IDs. Filtering happens later
        during regression (rows with NaN are excluded).
    
    Example:
        >>> texts = ["The cat sat", "Dogs bark loudly", ""]
        >>> scores = score_batch(texts)
        >>> scores['aoa']
        array([4.2, 5.1, nan])  # Empty sentence gets NaN
        >>> scores['freq']
        array([8.3, 7.9, nan])
        >>> len(scores['aoa']) == len(texts)  # Always True
        True
    """
    # Get batch size
    n = len(text_batch)
    
    # Pre-allocate numpy arrays for all scores
    # Using float64 (double precision) for numerical stability in regression
    # NaN is a valid float64 value that represents missing data
    aoa_scores = np.empty(n, dtype=np.float64)
    freq_scores = np.empty(n, dtype=np.float64)
    conc_scores = np.empty(n, dtype=np.float64)
    phon_scores = np.empty(n, dtype=np.float64)
    
    # Process each sentence in the batch
    for i, text in enumerate(text_batch):
        # Try to compute AoA score
        # Catch any exceptions to prevent one bad sentence from crashing entire batch
        try:
            aoa_scores[i] = compute_aoa(text)
        except Exception as e:
            # If scoring fails, record NaN
            # In production, consider logging the error: logging.warning(f"AoA failed: {e}")
            aoa_scores[i] = np.nan
        
        # Try to compute frequency score
        try:
            freq_scores[i] = compute_frequency(text)
        except Exception as e:
            freq_scores[i] = np.nan
        
        # Try to compute concreteness score
        try:
            conc_scores[i] = compute_concreteness(text)
        except Exception as e:
            conc_scores[i] = np.nan
        
        # Try to compute phonetic complexity score
        try:
            phon_scores[i] = compute_phonetic_complexity(text)
        except Exception as e:
            phon_scores[i] = np.nan
    
    # Return all scores as a dictionary
    # This format is convenient for the append_features_parquet function
    return {
        'aoa': aoa_scores,    # Target variable for regression
        'freq': freq_scores,   # Predictor: word frequency
        'conc': conc_scores,   # Predictor: concreteness
        'phon': phon_scores    # Predictor: phonetic complexity
    }