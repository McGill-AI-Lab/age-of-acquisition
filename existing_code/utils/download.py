"""
download.py - Functions for downloading and streaming corpus data from HuggingFace.

This module handles:
1. Downloading corpus files from HuggingFace datasets.
2. Streaming large corpus files in memory-efficient batches.

The streaming approach ensures we can process corpora larger than available RAM.
"""
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download


def download_corpus_from_huggingface(
        dataset_id: str,
        filename: str,
        cache_dir: str = "./data"
) -> str:
    """
    Download corpus from HuggingFace and save it locally.

    Args:
        dataset_id: HuggingFace dataset identifier
                   (e.g., "mcgillailab/aoa_sorted_curriculums")
        filename: Name of the file to download
                 (e.g., "refined_corpus_shuffled.txt")
        cache_dir: Local directory to cache the downloaded file.

    Returns:
        Local file path to the downloaded text file.
    """
    # Create cache directory if it doesn't exist already
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Download the file from HuggingFace
    local_path = hf_hub_download(
        repo_id=dataset_id,  # The HuggingFace dataset ID
        filename=filename,  # The specific file to download
        cache_dir=cache_dir,  # Where to cache the file
        repo_type="dataset"  # Specifies this is a dataset, not a model
    )

    return local_path


def stream_corpus_lines(
        path: str,
        chunk_size: int = 200_000
) -> Iterator[tuple[list[int], list[str]]]:
    """
    Stream corpus lines in batches for memory-efficient processing.

    Each line in the corpus should contain one sentence. Empty lines are
    automatically skipped.

    Args:
        path: Path to corpus file
        chunk_size: Number of sentences per yielded batch. Default 200,000.

    Yields:
        Tuples of (sentence_ids, sentences) where:
        - sentence_ids: List of integer IDs for each sentence
        - sentences: List of corresponding sentence strings
    """
    # Initialize batch accumulators
    sid_batch = []  # Sentence IDs for current batch
    text_batch = []  # Sentence texts for current batch

    # Open file with UTF-8 encoding (handles international characters)
    with open(path, 'r', encoding='utf-8') as f:
        # Enumerate gives us both the line number (sid) and content (line)
        # sid starts at 0 and increments for each line (including empty lines)
        for sid, line in enumerate(f):
            # Strip whitespace from both ends of the line
            text = line.strip()

            # Skip empty lines - they don't contain useful sentences
            if text:
                # Add this sentence to the current batch
                sid_batch.append(sid)
                text_batch.append(text)

                # Check if we've reached the batch size limit
                if len(sid_batch) >= chunk_size:
                    # Yield the complete batch
                    yield sid_batch, text_batch

                    # Reset accumulators for the next batch
                    # This frees memory from processed sentences
                    sid_batch = []
                    text_batch = []

        # After processing all lines, check if there are remaining items
        # The last batch is usually smaller than chunk_size
        if sid_batch:
            yield sid_batch, text_batch
