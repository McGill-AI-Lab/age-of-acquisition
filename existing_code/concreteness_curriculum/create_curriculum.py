import os
import re
import sqlite3
from typing import Union, Iterable, Tuple, List

import nltk
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

from create_concreteness_lookup import load_concreteness_word_ratings
SCORE_MAP = load_concreteness_word_ratings()

# ============================ Config ============================ #
OUTPUT_PATH = "concreteness_curriculum/data/outputs/entire_concreteness_curriculum.txt"
DB_PATH = "concreteness_curriculum/data/tmp/score_index.sqlite"   # on-disk DB for scores (no dedupe)
TEXT_COL_DEFAULT = "text"
BATCH_COMMIT = 10_000                     # commit every N inserted rows
METHOD = "mean"
LEMMATIZE = True
SKIP_STOPWORDS = True
# ================================================================ #

# ----------------------- Lemmatizing ------------------------- #

def get_lemmas(tokens: List[str]) -> List[str]:
  """
  Lemmatizes an entire sentence
  """
  lemmatized_tokens = []
  tagged_tokens = pos_tag(tokens)
  for word, tag in tagged_tokens:
    lemmatized_tokens.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))
  return lemmatized_tokens

def get_wordnet_pos(tag):
  if tag.startswith('J'):  
    return 'a'
  elif tag.startswith('V'):  
    return 'v'
  elif tag.startswith('N'):  
    return 'n'
  elif tag.startswith('R'):  
    return 'r'
  else:
    return 'n'


# ----------------------- Scoring and Tokenizing ------------------------- #

def score(
  sentence: List,
  lemmatize: bool = LEMMATIZE,
  method: str = METHOD,
  skip_stopwords: bool = SKIP_STOPWORDS
) -> float:
  """
  Scores a sentence by concreteness score
  method can be "min", "mean", or "max"

  Unknown words are skipped over

  If sentence is empty or none of the words are known, returns -1
  """
  scores = []
  if lemmatize:
    lemmatized_sentence = []
  for i, token in enumerate(sentence):
    word = token.lower().strip()
    if skip_stopwords:
      if word in english_stopwords:
        continue
    if word not in SCORE_MAP:
      if lemmatize:
        if len(lemmatized_sentence) == 0:
          lemmatized_sentence = get_lemmas([w.lower().strip() for w in sentence])
        new_word = lemmatized_sentence[i]
        if new_word in SCORE_MAP:
          scores.append(SCORE_MAP[new_word])
        else:
          continue
      continue
    scores.append(SCORE_MAP[word])
  if len(scores) == 0:
    return -1.0
  if method == "min":
    return min(scores)
  if method == "mean":
    return sum(scores) / len(scores)
  if method == "max":
    return max(scores)
  raise ValueError("Method should be one of 'min', 'mean', or 'max'")

def tokenize(line: str) -> list[str]:
    """
    Tokenize by splitting on spaces, then:
      - remove trailing punctuation (.,?!)
      - remove possessive/apostrophe endings ('s or ’s)
    """
    tokens = line.strip().split(" ")
    cleaned = []
    for tok in tokens:
        # strip .,?! from the end
        tok = re.sub(r"[.?,!]+$", "", tok)
        # remove possessive/apostrophe endings
        tok = re.sub(r"(\'s|’s)$", "", tok)
        if tok:  # keep non-empty
            cleaned.append(tok)
    return cleaned

# ----------------------- Dataset iteration ------------------------- #
def select_split(ds: Union[Dataset, DatasetDict], split: str = "train") -> Dataset:
    if isinstance(ds, DatasetDict):
        if split in ds:
            return ds[split]
        # fallback to first available split
        first = next(iter(ds.keys()))
        return ds[first]
    return ds

def iter_sentences(ds: Dataset, text_col: str) -> Iterable[Tuple[int, str]]:
    """Yield (idx, sentence) for each example with a non-empty sentence."""
    for i, ex in enumerate(ds):
        s = ex.get(text_col, "")
        if isinstance(s, str):
            s = s.strip()
            if s:
                yield (i, s)

# ------------------------- SQLite helpers -------------------------- #
def _ensure_dirs():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def _open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    # Pragmas tuned for safety + reasonable speed
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # ~200MB page cache (negative = KB)
    return conn

def _init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    # Only one table now; idx is the primary key so we can resume safely.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            idx INTEGER PRIMARY KEY,
            score REAL NOT NULL
        );
    """)
    # Keep the index for write-time sorting
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scores_score ON scores(score DESC);")
    conn.commit()

def _count_lines(path: str) -> int:
    """Count lines in a text file efficiently."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "rb") as f:
        # read in chunks
        for block in iter(lambda: f.read(1 << 20), b""):
            count += block.count(b"\n")
    return count

# ----------------------- Pass 1: score (resumable) --------------------- #
def build_scores_on_disk(
    ds: Union[Dataset, DatasetDict],
    scorer,
    text_col: str = TEXT_COL_DEFAULT,
    split: str = "train",
) -> int:
    """
    Stream the dataset once (resumable), and store (idx, score) rows.
    Resumes from MAX(idx) already present in the 'scores' table.
    Returns the count of rows inserted in this run.
    """
    print("Building scores on disk (resumable)...")
    _ensure_dirs()
    ds_split = select_split(ds, split=split)
    conn = _open_db(DB_PATH)
    _init_db(conn)
    cur = conn.cursor()

    # Determine resume point
    row = cur.execute("SELECT MAX(idx) FROM scores;").fetchone()
    last_done = row[0] if row and row[0] is not None else -1
    start_from = last_done + 1
    if start_from > 0:
        print(f"Resuming scoring from dataset idx {start_from}.")

    inserted_count = 0
    pending_since_commit = 0

    for i, s in iter_sentences(ds_split, text_col):
        # Skip anything we've already scored
        if i <= last_done:
            continue

        sc = float(scorer(tokenize(s)))  # pass tokens to your score()
        cur.execute("INSERT OR IGNORE INTO scores(idx, score) VALUES (?, ?);", (i, sc))
        if cur.rowcount == 1:
            inserted_count += 1
            pending_since_commit += 1

            if pending_since_commit >= BATCH_COMMIT:
                conn.commit()
                pending_since_commit = 0

            if inserted_count % 1_000_000 == 0:
                print(f"{inserted_count} new lines scored in this run.")

    conn.commit()
    conn.close()
    return inserted_count

# -------------------- Pass 2: sort + stream write (resumable) ----------- #
def write_sorted_tokenized_output(
    ds: Union[Dataset, DatasetDict],
    text_col: str = TEXT_COL_DEFAULT,
    split: str = "train",
    output_path: str = OUTPUT_PATH,
) -> int:
    """
    Streams sorted indices from SQLite, looks up sentences by idx, tokenizes,
    and writes one line per sentence (tokens space-separated).
    Resumable: counts existing lines in the output file and skips that many
    rows from the sorted SELECT.
    Returns number of lines written in this run.
    """
    print("Writing sorted output (resumable)...")
    ds_split = select_split(ds, split=split)

    already_written = _count_lines(output_path)
    if already_written > 0:
        print(f"Resuming writing at line {already_written} (appending).")
        mode = "a"
    else:
        mode = "w"

    written = 0
    with sqlite3.connect(DB_PATH) as conn, open(output_path, mode, encoding="utf-8") as f:
        cur = conn.cursor()
        # Stream rows in score-desc order; skip rows we've already written.
        for (idx,) in cur.execute(
            """
            SELECT idx
            FROM scores
            WHERE score > 0
            ORDER BY score DESC, idx ASC
            LIMIT -1 OFFSET ?;
            """,
            (already_written,)
        ):
            s = ds_split[int(idx)][text_col].strip() + '\n'
            f.write(s)
            written += 1
            if (already_written + written) % 1_000_000 == 0:
                print(f"{already_written + written} total lines present in output.")

    return written

# -------------------------- End-to-end API -------------------------- #
def build_and_write_curriculum(
    ds: Union[Dataset, DatasetDict],
    scorer,
    text_col: str = TEXT_COL_DEFAULT,
    split: str = "train",
    output_path: str = OUTPUT_PATH,
) -> int:
    """
    Complete pipeline (both resumable):
      1) Pass 1: score into SQLite
      2) Pass 2: sort in SQLite and stream-write tokenized lines to `output_path`
    Returns number of lines written in this run.
    """
    inserted = build_scores_on_disk(ds, scorer, text_col=text_col, split=split)
    print("Finished building scores on disk.")
    written = write_sorted_tokenized_output(ds, text_col=text_col, split=split, output_path=output_path)
    print("Finished writing sorted output.")
    return written


if __name__ == "__main__":
    # Your dataset (each row has "text" = one sentence)
    data_files = {"train": "refined_corpus_shuffled.txt"}
    dataset = load_dataset("mcgillailab/aoa_sorted_curriculums", data_files=data_files)

    count = build_and_write_curriculum(
        dataset,
        scorer=score,
        text_col="text",
        split="train",
        output_path="concreteness_curriculum/data/outputs/entire_concreteness_curriculum.txt",
    )

    print(f"Wrote {count} tokenized lines this run to concreteness_curriculum/data/outputs/entire_concreteness_curriculum.txt")
