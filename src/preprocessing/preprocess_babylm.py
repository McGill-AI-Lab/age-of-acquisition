from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import spacy
from tqdm import tqdm

PKG_DIR = Path(__file__).resolve().parent

def preprocess_babylm() -> None:
    """
    Preprocess BabyLM .train files from:
      data/raw/corpora/BabyLM/
    and write outputs to a mirroring structure under:
      data/processed/BabyLM/

    Rules:
      - childes.train:
          keep only lines starting with "*"
          remove everything before and including first ":"
          strip + lowercase
      - bnc_spoken.train:
          lowercase
      - gutenberg.train:
          split each line into sentences via spaCy sentencizer
          output: one sentence per line
          lowercase
      - open_subtitles.train:
          lowercase
      - simple_wiki.train:
          drop lines that begin with "= = ="
          split each line into sentences via spaCy sentencizer
          output: one sentence per line
          lowercase
      - switchboard.train:
          remove first two characters
          strip + lowercase
    """

    raw_root = PKG_DIR.parent.parent / "data" / "raw" / "corpora" / "BabyLM"
    out_root = PKG_DIR.parent.parent / "data" / "processed" / "BabyLM"

    # Lightweight sentence splitter
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    def ensure_parent(out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def write_lines(out_path: Path, lines: Iterable[str]) -> None:
        ensure_parent(out_path)
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                if line:  # skip empty lines that may arise after stripping/splitting
                    f.write(line + "\n")
    
    def sentencize_line(line: str) -> List[str]:
        text = line.strip()
        if not text:
            return []
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


    def process_childes(in_path: Path, out_path: Path) -> None:
        def gen() -> Iterable[str]:
            with in_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in tqdm(f, desc=in_path.name, unit="lines"):
                    if not line.startswith("*"):
                        continue
                    idx = line.find(":")
                    if idx == -1:
                        continue
                    text = (
                        line[idx + 1 :].strip().lower()
                        .replace("xxx", "")
                        .replace("yyy", "")
                        .replace("www", "")
                        .strip()
                    )
                    if text:
                        yield text

        write_lines(out_path, gen())

    def process_lowercase_only(in_path: Path, out_path: Path) -> None:
        def gen() -> Iterable[str]:
            with in_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in tqdm(f, desc=in_path.name, unit="lines"):
                    text = line.strip().lower()
                    if text:
                        yield text

        write_lines(out_path, gen())

    def process_gutenberg(in_path: Path, out_path: Path) -> None:
        def gen() -> Iterable[str]:
            with in_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in tqdm(f, desc=in_path.name, unit="lines"):
                    for sent in sentencize_line(line):
                        yield sent.lower()

        write_lines(out_path, gen())

    def process_simple_wiki(in_path: Path, out_path: Path) -> None:
        def gen() -> Iterable[str]:
            with in_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in tqdm(f, desc=in_path.name, unit="lines"):
                    if line.startswith("= = ="):
                        continue
                    for sent in sentencize_line(line):
                        yield sent.lower()

        write_lines(out_path, gen())

    def process_switchboard(in_path: Path, out_path: Path) -> None:
        def gen() -> Iterable[str]:
            with in_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in tqdm(f, desc=in_path.name, unit="lines"):
                    # Remove first two characters (even if shorter; that yields empty)
                    chopped = line[2:] if len(line) >= 2 else ""
                    text = chopped.strip().lower()
                    if text:
                        yield text

        write_lines(out_path, gen())

    # File list + handlers
    handlers = {
        "childes.train": process_childes,
        "bnc_spoken.train": process_lowercase_only,
        "gutenberg.train": process_gutenberg,
        "open_subtitles.train": process_lowercase_only,
        "simple_wiki.train": process_simple_wiki,
        "switchboard.train": process_switchboard,
    }

    for fname, fn in handlers.items():
        in_path = raw_root / fname
        if not in_path.exists():
            raise FileNotFoundError(f"Missing expected input file: {in_path}")

        out_path = out_root / fname 
        fn(in_path, out_path)


if __name__ == "__main__":
    preprocess_babylm()
