"""
parquet_io.py - Parquet I/O operations for sentences, features, and residuals.

This module provides functions to write and read partitioned Parquet datasets.
Partitioned storage allows:
1. Parallel processing of batches
2. Incremental writes without loading entire dataset
3. Efficient columnar storage with compression

All write functions create "part-*.parquet" files in a directory, which
together form a single logical dataset.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_sentences_parquet(
        sid_batch: list[int],
        text_batch: list[str],
        out_dir: str
) -> None:
    """
    Append a batch of sentences to sentences.parquet as a new part.

    This function writes sentence data (IDs and text) to a partitioned
    Parquet dataset. Each call creates a new "part" file, allowing
    incremental writing as we stream through the corpus.

    The resulting directory structure:
        out_dir/sentences.parquet/part-00000.parquet
        out_dir/sentences.parquet/part-00001.parquet
        ...

    Args:
        sid_batch: List of sentence IDs
        text_batch: List of corresponding sentences
        out_dir: Directory to store 'sentences.parquet'

    Output:
        Creates out_dir/sentences.parquet/part-*.parquet
        Part number is auto-incremented based on existing files
    """
    # Construct the output directory path
    out_path = Path(out_dir) / "sentences.parquet"

    # Create directory structure if the it does not exist already
    out_path.mkdir(parents=True, exist_ok=True)

    # Create an Apache Arrow table from the data
    table = pa.table({
        'sid': pa.array(sid_batch, type=pa.int64()),  # 64-bit integers for IDs
        'text': pa.array(text_batch, type=pa.string())
    })

    # Generate unique part filename
    # Count existing part files to determine next part number
    existing_parts = list(out_path.glob("part-*.parquet"))
    part_num = len(existing_parts)

    # Format with 5 digits (supports up to 99,999 parts)
    part_path = out_path / f"part-{part_num:05d}.parquet"

    # Write the Arrow table to a Parquet file
    pq.write_table(table, part_path)


def append_features_parquet(
        sid_batch: list[int],
        scores: dict[str, np.ndarray],
        out_dir: str
) -> None:
    """
    Append a batch of feature scores to features.parquet as a new part.

    This function writes linguistic feature scores (AoA, frequency,
    concreteness, phonetic complexity) to a partitioned Parquet dataset.

    The resulting directory structure:
        out_dir/features.parquet/part-00000.parquet
        out_dir/features.parquet/part-00001.parquet
        ...

    Args:
        sid_batch: List of sentence IDs (must match the order in scores)
        scores: Dictionary with keys "aoa", "freq", "conc", "phon"
               Each value is a numpy array of shape (n,) where n = len(sid_batch)
               Arrays should contain float64 values (NaN for missing scores)
        out_dir: Directory to store 'features.parquet'

    Output:
        Creates out_dir/features.parquet/part-*.parquet
        Each part contains columns: sid, aoa, freq, conc, phon
    """
    # Construct the output directory path
    out_path = Path(out_dir) / "features.parquet"

    # Create directory structure if needed
    out_path.mkdir(parents=True, exist_ok=True)

    # Create Arrow table with all feature columns
    # All scores are stored as float64 (double precision)
    # This maintains precision for downstream regression analysis
    table = pa.table({
        'sid': pa.array(sid_batch, type=pa.int64()),  # Sentence ID
        'aoa': pa.array(scores['aoa'], type=pa.float64()),  # Age of Acquisition
        'freq': pa.array(scores['freq'], type=pa.float64()),  # Frequency
        'conc': pa.array(scores['conc'], type=pa.float64()),  # Concreteness
        'phon': pa.array(scores['phon'], type=pa.float64())  # Phonetic complexity
    })

    # Generate unique part filename (same pattern as write_sentences_parquet)
    existing_parts = list(out_path.glob("part-*.parquet"))
    part_num = len(existing_parts)
    part_path = out_path / f"part-{part_num:05d}.parquet"

    # Write to Parquet file
    pq.write_table(table, part_path)


def write_residuals_parquet(
        sid_batch: list[int],
        residual_raw_batch: np.ndarray,
        out_dir: str,
        residual_z_batch: Optional[np.ndarray] = None
) -> None:
    """
    Append a batch of residuals to residuals parquet as a new part.

    This function writes residual data in two possible formats:
    1. Raw residuals only (during initial computation)
    2. Both raw and standardized residuals (after standardization)

    The output directory name changes based on which residuals are included:
    - residuals_raw.parquet: Contains only raw residuals
    - residuals.parquet: Contains both raw and standardized (z-scored) residuals

    Args:
        sid_batch: List of sentence IDs
        residual_raw_batch: Numpy array of raw residuals (before standardization)
                           Shape: (n,) where n = len(sid_batch)
        out_dir: Directory to store parquet files
        residual_z_batch: Optional numpy array of standardized residuals
                         If None, writes to residuals_raw.parquet
                         If provided, writes to residuals.parquet

    Output:
        Creates either:
        - out_dir/residuals_raw.parquet/part-*.parquet (if residual_z_batch is None)
        - out_dir/residuals.parquet/part-*.parquet (if residual_z_batch provided)
    """
    # Determine output path and table structure based on what data we have
    if residual_z_batch is None:
        # First pass: only raw residuals computed
        out_path = Path(out_dir) / "residuals_raw.parquet"
        table_dict = {
            'sid': pa.array(sid_batch, type=pa.int64()),
            'residual_raw': pa.array(residual_raw_batch, type=pa.float64())
        }
    else:
        # Second pass: standardized residuals available
        # This is the final output with curriculum scores
        out_path = Path(out_dir) / "residuals.parquet"
        table_dict = {
            'sid': pa.array(sid_batch, type=pa.int64()),
            'residual_raw': pa.array(residual_raw_batch, type=pa.float64()),
            'residual_z': pa.array(residual_z_batch, type=pa.float64())  # Curriculum score
        }

    # Create directory structure
    out_path.mkdir(parents=True, exist_ok=True)

    # Create Arrow table from the dictionary
    table = pa.table(table_dict)

    # Generate unique part filename (same pattern as other write functions)
    existing_parts = list(out_path.glob("part-*.parquet"))
    part_num = len(existing_parts)
    part_path = out_path / f"part-{part_num:05d}.parquet"

    # Write to Parquet file
    pq.write_table(table, part_path)


def read_parquet_batches(parquet_dir: str, batch_size: int = 100_000):
    """
    Read a partitioned parquet dataset in batches for memory efficiency.

    This generator function reads a partitioned Parquet dataset and yields
    batches of records. This is more memory-efficient than loading the
    entire dataset into a single DataFrame.

    Args:
        parquet_dir: Directory containing part-*.parquet files
                    (e.g., "./out/sentences.parquet")
        batch_size: Number of rows to read at a time
                   Larger batches are faster but use more memory

    Yields:
        PyArrow RecordBatch objects
        Each batch contains up to batch_size rows
        Can be converted to pandas with: batch.to_pandas()
    """
    # Create a dataset object that represents the partitioned dataset
    # This doesn't load all data into memory
    dataset = pq.ParquetDataset(parquet_dir)

    # Iterate through the dataset in batches
    # read_batches() is a generator that yields RecordBatch objects
    for batch in dataset.read_batches(batch_size=batch_size):
        yield batch