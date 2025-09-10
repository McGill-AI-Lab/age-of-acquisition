#!/usr/bin/env python
from __future__ import annotations

"""
Trains a Word2Vec model on a large text corpus with memory optimization.

This script is designed to train a gensim Word2Vec model on a pre-processed
corpus with extensive memory optimization for handling very large datasets.
It can handle two input formats:
- A text file where each line contains a single sentence with tokens separated
  by spaces.
- A Parquet file containing a 'tokens' column, where each entry is a list of
  strings (tokens).

Memory optimization features:
- Streaming corpus loading with configurable batch sizes
- Chunked text file reading for very large files
- Memory usage monitoring and logging throughout training
- Explicit garbage collection at key points
- Optimized callback classes with reduced memory footprint
- Memory cleanup after plotting
- Streaming vocabulary building to prevent MemoryError during vocab scanning
- Vocabulary size limiting to control memory usage
- Pre-filtered sentence streams to reduce gensim's memory footprint

The script includes tracking capabilities:
- Vector movement tracking across epochs by frequency cohorts
- Overlap tracking of nearest neighbors across epochs

Prerequisites:
- A pre-processed corpus file (either .txt or .parquet).
- Install required packages: `pip install gensim scikit-learn matplotlib pyarrow scipy psutil`

Example usage (from the project root directory):
    # Using a text file with chunked reading and streaming vocab (prevents MemoryError)
    python train_word2vec.py \
        --input_file data/training_data/shuffled_bookcorpus_childes.txt \
        --output_dir models/word2vec \
        --use_chunked_text \
        --text_chunk_size 5000 \
        --use_streaming_vocab \
        --max_vocab_size 1000000

    # Using a Parquet file with small batch size and streaming vocab
    python train_word2vec.py \
        --input_file data/training_data/aoa_sorted_training_data_mean.parquet \
        --output_dir models/word2vec \
        --parquet_batch_size 500 \
        --use_streaming_vocab \
        --max_vocab_size 500000

The script will save the trained model and generate plots showing training dynamics.
"""

import argparse
import gc
import logging
import multiprocessing
import psutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors


def log_memory_usage(stage: str):
    """Log current memory usage for monitoring."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logging.info(f"Memory usage at {stage}: {memory_mb:.1f} MB")


def force_garbage_collection():
    """Force garbage collection to free up memory."""
    collected = gc.collect()
    logging.debug(f"Garbage collection freed {collected} objects")


def build_vocabulary_streaming(sentences, min_count=10, max_vocab_size=None):
    """
    Build vocabulary in a memory-efficient way by streaming through the corpus
    and maintaining only the most frequent words.
    """
    logging.info("Building vocabulary in streaming mode to reduce memory usage...")
    vocab_counts = {}
    total_sentences = 0
    
    # First pass: count words with memory management
    for sentence in sentences:
        total_sentences += 1
        if total_sentences % 100000 == 0:
            log_memory_usage(f"vocab_building_sentence_{total_sentences}")
            force_garbage_collection()
        
        for word in sentence:
            if word in vocab_counts:
                vocab_counts[word] += 1
            else:
                vocab_counts[word] = 1
        
        # Periodically clean up low-frequency words to save memory
        if total_sentences % 500000 == 0:
            # Remove words below min_count to free memory
            words_to_remove = [word for word, count in vocab_counts.items() if count < min_count]
            for word in words_to_remove:
                del vocab_counts[word]
            logging.info(f"Cleaned up {len(words_to_remove)} low-frequency words. Vocab size: {len(vocab_counts)}")
    
    # Filter by min_count
    filtered_vocab = {word: count for word, count in vocab_counts.items() if count >= min_count}
    logging.info(f"After min_count filtering: {len(filtered_vocab)} words")
    
    # If max_vocab_size is specified, keep only the most frequent words
    if max_vocab_size and len(filtered_vocab) > max_vocab_size:
        sorted_words = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
        filtered_vocab = dict(sorted_words[:max_vocab_size])
        logging.info(f"Limited to top {max_vocab_size} most frequent words")
    
    # Clear original vocab_counts to free memory
    del vocab_counts
    force_garbage_collection()
    
    return filtered_vocab


class MemoryEfficientSentenceStream:
    """
    A wrapper that filters sentences to only include words from a pre-built vocabulary.
    This prevents gensim from building a huge vocabulary in memory.
    """
    
    def __init__(self, sentences, vocab_words):
        self.sentences = sentences
        self.vocab_words = vocab_words
    
    def __iter__(self):
        for sentence in self.sentences:
            # Filter sentence to only include words in our vocabulary
            filtered_sentence = [word for word in sentence if word in self.vocab_words]
            if len(filtered_sentence) > 0:  # Only yield non-empty sentences
                yield filtered_sentence


class OverlapTracker(CallbackAny2Vec):
    """
    Gensim callback to track the average overlap (shared nearest neighbors)
    of word vectors across training epochs.

    This implementation is optimized to compute nearest neighbors only once per
    epoch (at the end) to reduce CPU overhead. It compares the neighbors at the
    end of the current epoch with the neighbors from the end of the previous one.
    """

    def __init__(self, sample_size=100, k_neighbors=10):
        self.epoch = 0
        self.sample_size = sample_size
        self.k_neighbors = k_neighbors
        self.sample_words = []
        self.valid_sample_words = []
        self.previous_neighbors = {}
        self.overlap_history = []

    def on_train_begin(self, model):
        """Called at the beginning of training. Sets up sample words for tracking."""
        logging.info("Callback: Setting up sample words for overlap tracking.")
        word_counts = {
            word: model.wv.get_vecattr(word, "count")
            for word in model.wv.index_to_key
        }
        sorted_words = sorted(
            word_counts.keys(), key=lambda w: word_counts[w], reverse=True
        )
        vocab_size = len(sorted_words)

        if vocab_size < self.sample_size:
            logging.warning("Vocabulary too small for overlap tracking. Skipping.")
            return

        # Sample words from different frequency ranges
        high_freq_end = min(1000, vocab_size // 3)
        mid_freq_start = vocab_size // 3
        mid_freq_end = 2 * vocab_size // 3
        low_freq_start = 2 * vocab_size // 3

        # Sample from different frequency ranges
        high_freq_words = sorted_words[100:high_freq_end]
        mid_freq_words = sorted_words[mid_freq_start:mid_freq_end]
        low_freq_words = sorted_words[low_freq_start:]

        # Combine and sample
        all_candidates = high_freq_words + mid_freq_words + low_freq_words
        if len(all_candidates) >= self.sample_size:
            # Randomly sample from candidates
            np.random.seed(42)  # For reproducibility
            self.sample_words = np.random.choice(
                all_candidates, self.sample_size, replace=False
            ).tolist()
        else:
            self.sample_words = all_candidates

        # Pre-filter sample words to ensure they are in the final vocabulary
        self.valid_sample_words = [w for w in self.sample_words if w in model.wv]
        if len(self.valid_sample_words) < self.k_neighbors + 1:
            logging.warning(
                f"Not enough valid sample words ({len(self.valid_sample_words)}) in vocabulary for overlap tracking. "
                f"Need at least {self.k_neighbors + 1}. Disabling."
            )
            self.valid_sample_words = []  # Disable tracking
            return

        logging.info(
            f"  - Selected {len(self.sample_words)} words for overlap tracking, {len(self.valid_sample_words)} are in vocab."
        )

    def _get_current_neighbors(self, model: Word2Vec) -> dict[str, set[str]]:
        """Helper to compute nearest neighbors for the current model state."""
        # Get vectors for the valid sample words
        sample_vectors = np.array([model.wv[word] for word in self.valid_sample_words])

        # The number of neighbors to query must be less than the number of samples.
        # We query for k+1 because the word itself is its own nearest neighbor.
        n_neighbors_query = min(self.k_neighbors + 1, len(sample_vectors))
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors_query, algorithm="auto", metric="cosine"
        )
        nbrs.fit(sample_vectors)

        # Find the neighbors
        _, indices = nbrs.kneighbors(sample_vectors)

        # Store neighbors for this state
        current_neighbors = {}
        for i, word in enumerate(self.valid_sample_words):
            # Get k nearest neighbors (excluding self at index 0)
            neighbor_indices = indices[i][1:n_neighbors_query]
            current_neighbors[word] = {
                self.valid_sample_words[j] for j in neighbor_indices
            }

        return current_neighbors

    def on_epoch_end(self, model):
        """
        Called at the end of each epoch. Computes neighbors, calculates overlap
        with the previous epoch's state, and stores the new state.
        """
        self.epoch += 1
        if not self.valid_sample_words:
            return

        logging.info(
            f"Callback: End of epoch {self.epoch}. Calculating nearest neighbor overlap."
        )

        # Get neighbors for the model state at the end of this epoch
        current_neighbors = self._get_current_neighbors(model)

        # If we have neighbors from a previous epoch, calculate the overlap
        if self.previous_neighbors:
            overlaps = []
            # Iterate over words from the previous state to ensure consistent comparison
            for word, prev_neighbors_set in self.previous_neighbors.items():
                if word in current_neighbors:
                    curr_neighbors_set = current_neighbors[word]
                    intersection_size = len(
                        curr_neighbors_set.intersection(prev_neighbors_set)
                    )

                    # Avoid division by zero if k_neighbors is 0
                    if self.k_neighbors > 0:
                        overlap = intersection_size / self.k_neighbors
                        overlaps.append(overlap)

            if overlaps:
                avg_overlap = np.mean(overlaps)
                self.overlap_history.append(avg_overlap)
                logging.info(f"  - Average overlap with previous epoch: {avg_overlap:.4f}")
        else:
            logging.info(
                "  - First epoch complete. Storing initial neighbor sets for next epoch."
            )

        # Clear previous neighbors to free memory before storing new ones
        if hasattr(self, 'previous_neighbors'):
            del self.previous_neighbors
            force_garbage_collection()
        
        # Update previous_neighbors for the next epoch's comparison
        self.previous_neighbors = current_neighbors


class VectorMovementTracker(CallbackAny2Vec):
    """
    Gensim callback to track the average cosine distance movement of word vectors
    for different frequency-based cohorts across training epochs.
    """

    def __init__(self, sample_size=100):
        self.epoch = 0
        self.sample_size = sample_size
        self.cohorts = {}
        self.previous_vectors = {}
        self.movement_history = {}  # e.g., {'high_freq': [0.1, 0.05, ...]}

    def on_train_begin(self, model):
        """Called at the beginning of training. Sets up the word cohorts."""
        logging.info("Callback: Setting up word cohorts for movement tracking.")
        word_counts = {word: model.wv.get_vecattr(word, "count") for word in model.wv.index_to_key}
        sorted_words = sorted(word_counts.keys(), key=lambda w: word_counts[w], reverse=True)
        vocab_size = len(sorted_words)

        if vocab_size < self.sample_size * 3:
            logging.warning("Vocabulary too small to create distinct cohorts. Skipping vector movement tracking.")
            return

        # Define cohorts by sampling from different parts of the frequency-sorted vocab
        cohort_definitions = {
            "high_freq": sorted_words[100 : 100 + self.sample_size],
            "mid_freq": sorted_words[vocab_size // 2 : vocab_size // 2 + self.sample_size],
            "low_freq": sorted_words[-self.sample_size - 100 : -100],
        }

        for name, words in cohort_definitions.items():
            if words:
                self.cohorts[name] = words
                self.movement_history[name] = []
                logging.info(f"  - Cohort '{name}' example: '{words[0]}' (count: {word_counts[words[0]]})")

    def on_epoch_begin(self, model):
        """Called at the start of each epoch. Stores current vectors."""
        if not self.cohorts: return
        self.epoch += 1
        logging.info(f"Callback: Beginning epoch {self.epoch}. Storing current vectors for tracking.")
        
        # Clear previous vectors to free memory
        if hasattr(self, 'previous_vectors'):
            del self.previous_vectors
            force_garbage_collection()
        
        # Store only the vectors we need for tracking
        self.previous_vectors = {}
        for cohort_words in self.cohorts.values():
            for word in cohort_words:
                if word in model.wv:
                    self.previous_vectors[word] = model.wv[word].copy()

    def on_epoch_end(self, model):
        """Called at the end of each epoch. Calculates and records vector movement."""
        if not self.cohorts or not self.previous_vectors: return
        logging.info(f"Callback: End of epoch {self.epoch}. Calculating vector movement.")
        for name, words in self.cohorts.items():
            distances = [cosine(self.previous_vectors[word], model.wv[word]) for word in words if word in self.previous_vectors and word in model.wv]
            if distances:
                avg_distance = np.mean(distances)
                self.movement_history[name].append(avg_distance)
                logging.info(f"  - Avg. movement for '{name}' cohort: {avg_distance:.6f}")


class ParquetSentenceStream:
    """
    An iterator that reads sentences (lists of tokens) from a Parquet file,
    one batch at a time, to keep memory usage low.
    Expects a column named 'tokens' containing lists of strings.
    """

    def __init__(self, file_path: Path, tokens_column: str = "tokens", batch_size: int = 1000):
        self.file_path = file_path
        self.tokens_column = tokens_column
        self.batch_size = batch_size

    def __iter__(self):
        parquet_file = pq.ParquetFile(self.file_path)
        batch_count = 0
        for batch in parquet_file.iter_batches(
            columns=[self.tokens_column], 
            batch_size=self.batch_size
        ):
            batch_count += 1
            if batch_count % 100 == 0:  # Log every 100 batches
                log_memory_usage(f"parquet_batch_{batch_count}")
                force_garbage_collection()
            
            # Process batch and yield sentences
            sentences = batch.column(self.tokens_column).to_pylist()
            for sentence in sentences:
                if sentence:  # Skip empty sentences
                    yield sentence
            
            # Clear the batch from memory
            del sentences
            del batch


class ChunkedTextStream:
    """
    An iterator that reads text files in chunks to reduce memory usage for very large files.
    """
    
    def __init__(self, file_path: Path, chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
    
    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            chunk = []
            line_count = 0
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    chunk.append(line.split())
                    line_count += 1
                    
                    if len(chunk) >= self.chunk_size:
                        for sentence in chunk:
                            yield sentence
                        chunk = []
                        
                        if line_count % (self.chunk_size * 10) == 0:  # Log every 10 chunks
                            log_memory_usage(f"text_chunk_{line_count}")
                            force_garbage_collection()
            
            # Yield remaining sentences
            for sentence in chunk:
                yield sentence


def main():
    """
    Main function to parse arguments, train the model, and save it.
    """
    parser = argparse.ArgumentParser(
        description="Train a Word2Vec model on a text corpus."
    )
    # Assuming the script is run from the project root `age-of-acquisition/`
    project_root = Path.cwd()

    parser.add_argument(
        "--input_file",
        type=str,
        default=str(project_root / "data" / "training_data" / "aoa_sorted_training_data_mean.parquet"),
        help="Path to the input corpus file. Can be a .txt file (one sentence per line) or a .parquet file (with a 'tokens' column).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "models" / "word2vec"),
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--vector_size",
        type=int,
        default=300,
        help="Dimensionality of the word vectors.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=8,
        help="Maximum distance between the current and predicted word within a sentence.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=10,
        help="Ignores all words with total frequency lower than this.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of iterations (epochs) over the corpus.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker threads to use for training.",
    )
    parser.add_argument(
        "--overlap_sample_size",
        type=int,
        default=100,
        help="Number of sample words to use for overlap tracking.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=30,
        help="Number of nearest neighbors to consider for overlap calculation.",
    )
    parser.add_argument(
        "--parquet_batch_size",
        type=int,
        default=1000,
        help="Batch size for reading Parquet files (smaller = less memory).",
    )
    parser.add_argument(
        "--text_chunk_size",
        type=int,
        default=10000,
        help="Chunk size for reading text files (smaller = less memory).",
    )
    parser.add_argument(
        "--use_chunked_text",
        action="store_true",
        help="Use chunked reading for text files (recommended for very large files).",
    )
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=None,
        help="Maximum vocabulary size to prevent memory errors (e.g., 1000000).",
    )
    parser.add_argument(
        "--use_streaming_vocab",
        action="store_true",
        help="Use streaming vocabulary building to prevent MemoryError during vocab scanning.",
    )
    args = parser.parse_args()

    # --- Setup ---
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    
    # Log initial memory usage
    log_memory_usage("startup")

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_path = output_dir / "word2vec.model"

    if not input_path.exists():
        logging.error(f"Input file not found at: {input_path}")
        logging.error(
            "Please ensure the training data exists. You may need to run a "
            "preprocessing script first."
        )
        return

    # --- Corpus Preparation ---
    logging.info(f"Loading corpus from {input_path}...")
    log_memory_usage("before_corpus_loading")
    
    if input_path.suffix == ".parquet":
        logging.info(f"Detected Parquet file. Using ParquetSentenceStream with batch_size={args.parquet_batch_size}.")
        raw_sentences = ParquetSentenceStream(input_path, batch_size=args.parquet_batch_size)
    else:
        if args.use_chunked_text:
            logging.info(f"Detected text file. Using ChunkedTextStream with chunk_size={args.text_chunk_size}.")
            raw_sentences = ChunkedTextStream(input_path, chunk_size=args.text_chunk_size)
        else:
            logging.info("Detected text file. Using LineSentence.")
            # LineSentence expects a string path
            raw_sentences = LineSentence(str(input_path))
    
    log_memory_usage("after_corpus_loading")
    
    # --- Vocabulary Preprocessing (if enabled) ---
    if args.use_streaming_vocab:
        logging.info("Using streaming vocabulary building to prevent MemoryError...")
        log_memory_usage("before_vocab_building")
        
        # Build vocabulary in streaming mode
        vocab_dict = build_vocabulary_streaming(
            raw_sentences, 
            min_count=args.min_count,
            max_vocab_size=args.max_vocab_size
        )
        
        log_memory_usage("after_vocab_building")
        
        # Create filtered sentence stream - we need to recreate the sentence iterator
        vocab_words = set(vocab_dict.keys())
        if input_path.suffix == ".parquet":
            filtered_sentences = ParquetSentenceStream(input_path, batch_size=args.parquet_batch_size)
        else:
            if args.use_chunked_text:
                filtered_sentences = ChunkedTextStream(input_path, chunk_size=args.text_chunk_size)
            else:
                filtered_sentences = LineSentence(str(input_path))
        
        sentences = MemoryEfficientSentenceStream(filtered_sentences, vocab_words)
        
        logging.info(f"Vocabulary pre-built with {len(vocab_words)} words. Filtering sentences...")
        log_memory_usage("after_vocab_filtering")
    else:
        sentences = raw_sentences

    # --- Callbacks for Tracking ---
    movement_tracker = VectorMovementTracker()
    overlap_tracker = OverlapTracker(sample_size=args.overlap_sample_size, k_neighbors=args.k_neighbors)

    # --- Model Training ---
    logging.info("Initializing and training Word2Vec model...")
    log_memory_usage("before_training")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=1,  # Use Skip-Gram
        hs=0,  # Use Negative Sampling
        negative=10,
        epochs=args.epochs,
        sample=1e-4, # Subsampling for frequent words
        callbacks=[movement_tracker, overlap_tracker],
    )
    
    log_memory_usage("after_training")
    force_garbage_collection()

    # --- Save Model ---
    logging.info(f"Saving trained model to {model_output_path}...")
    log_memory_usage("before_saving_model")
    model.save(str(model_output_path))
    log_memory_usage("after_saving_model")

    logging.info("✅ Training complete.")
    logging.info(f"Model saved. Vocabulary size: {len(model.wv.index_to_key):,}")

    # --- Plot Vector Movement ---
    if movement_tracker.cohorts:
        logging.info("Plotting vector movement across epochs...")
        log_memory_usage("before_plotting")
        plt.figure(figsize=(12, 7))

        for name, movements in movement_tracker.movement_history.items():
            if movements:
                epochs_ran = range(1, len(movements) + 1)
                plt.plot(epochs_ran, movements, marker="o", linestyle="-", label=f"{name} cohort")

        plt.title("Average Word Vector Movement per Epoch by Frequency Cohort")
        plt.xlabel("Epoch")
        plt.ylabel("Average Cosine Distance (from previous epoch)")
        if args.epochs > 0:
            plt.xticks(range(1, args.epochs + 1))
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plot_output_path = output_dir / "word2vec_movement.png"
        plt.savefig(plot_output_path)
        plt.close()  # Close figure to free memory
        logging.info(f"Vector movement plot saved to {plot_output_path}")

    # --- Plot Overlap ---
    if overlap_tracker.overlap_history:
        logging.info("Plotting average overlap across epochs...")
        plt.figure(figsize=(12, 7))
        
        epochs_ran = range(1, len(overlap_tracker.overlap_history) + 1)
        plt.plot(epochs_ran, overlap_tracker.overlap_history, marker="o", linestyle="-", 
                color="red", linewidth=2, markersize=6, label="Average Overlap")
        
        plt.title("Average Overlap of Nearest Neighbors Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Average Overlap (shared nearest neighbors / k)")
        if args.epochs > 0:
            plt.xticks(range(1, args.epochs + 1))
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.ylim(0, 1)  # Overlap is between 0 and 1
        
        plot_output_path = output_dir / "word2vec_overlap.png"
        plt.savefig(plot_output_path)
        plt.close()  # Close figure to free memory
        logging.info(f"Overlap plot saved to {plot_output_path}")
    
    # Final cleanup
    log_memory_usage("final")
    force_garbage_collection()
    logging.info("Memory optimization complete.")


if __name__ == "__main__":
    main()