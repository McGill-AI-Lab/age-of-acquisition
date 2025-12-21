from typing import List

import spacy
from complexity_ranker.COMPLEXITY_RANKER_GLOBAL_VARIABLES import MODEL


class Preprocess:
    @classmethod
    def preprocess_text(cls, text: str) -> List[str]:
        """Clean input text: lowercase + remove punctuation."""
        model = spacy.load(MODEL)
        doc = model(text)
        words = [token.text.lower() for token in doc if token.is_alpha]
        if not words:
            raise ValueError("Please provide a non-empty text.")
        return words

    @classmethod
    def _prepare_text_file(cls, words: List[str], filename: str) -> None:
        if not filename.endswith(".txt"):
            filename = filename + ".txt"

        with open(filename, "w") as f:
            for word in words:
                f.write(word + "\n")

