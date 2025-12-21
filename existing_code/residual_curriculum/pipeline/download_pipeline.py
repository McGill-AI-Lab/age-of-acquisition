from data.residual_curriculum.parquet_io import write_sentences_parquet
from data.utils.download import download_corpus_from_huggingface, stream_corpus_lines


def download_pipeline(
        dataset_id: str = "mcgillailab/aoa_sorted_curriculums",
        filename: str = "refined_corpus_shuffled.txt",
        out_dir: str = "./out",
        cache_dir: str = "./data"
) -> str:
    """
    Download corpus and create sentences.parquet.

    This phase:
    1. Downloads the corpus from HuggingFace (if not already cached)
    2. Streams through the corpus line by line
    3. Writes sentences and IDs to partitioned parquet format

    The parquet format allows:
    - Efficient columnar storage with compression
    - Parallel processing of batches
    - Fast selective column reading

    Args:
        dataset_id: HuggingFace dataset identifier
                   Example: "mcgillailab/aoa_sorted_curriculums"
        filename: Corpus filename in the dataset
                 Example: "refined_corpus_shuffled.txt"
        out_dir: Output directory for sentences.parquet
                Will create out_dir/sentences.parquet/ directory
        cache_dir: Cache directory for downloads
                  HuggingFace will manage subdirectories here

    Returns:
        Path to downloaded corpus file (for use in next phase)

    Output:
        Creates out_dir/sentences.parquet/ with part-*.parquet files
        Each part contains a batch of sentences with columns: sid, text
    """
    print("Downloading corpus from HuggingFace...")

    # Download corpus (or use cached version)
    path = download_corpus_from_huggingface(dataset_id, filename, cache_dir)
    print(f"Downloaded to: {path}")

    print("\nStreaming corpus and writing sentences.parquet...")

    # Track progress
    batch_count = 0
    total_sentences = 0

    # Stream corpus in chunks and write to parquet
    # Default chunk_size=200,000 balances memory and I/O efficiency
    for sid_batch, text_batch in stream_corpus_lines(path):
        # Write this batch as a new parquet part
        write_sentences_parquet(sid_batch, text_batch, out_dir)

        # Update progress counters
        batch_count += 1
        total_sentences += len(sid_batch)

        # Print progress every 10 batches (every ~2M sentences)
        if batch_count % 10 == 0:
            print(f"  Processed {total_sentences:,} sentences ({batch_count} batches)")

    print(f"Complete! Total sentences: {total_sentences:,}")
    return path

if "__main__" == __name__:
    download_pipeline(dataset_id="mcgillailab/aoa_sorted_curriculums",
                      filename="refined_corpus_shuffled.txt",
                      out_dir="./out",
                      cache_dir="./data")