from collections import defaultdict
from typing import List

from complexity_ranker.morphology.morphological_stats import MorphoStats
from spacy.tokens import Doc


class MorphologicalComplexity:

    def __init__(self, io, model, nlp):
        self.io = io
        self.model = model
        self.nlp = nlp

    def get_morphology_stats(self, words: List[str]) -> MorphoStats:
        if not words:
            raise ValueError("Please provide a non-empty input.")

        stats = MorphoStats()

        known_words = []

        for w in words:
            if not w.isalpha():
                continue

            # Stats related to morphemes
            morphemes, confidence = self.model.viterbi_segment(w)
            if confidence > 20: # very weird segmentations are ignored, count word as 1 morpheme
                morphemes = [w]

            stats.total_morpheme_count += len(morphemes)
            if len(morphemes) > 1:
                stats.complex_word_count += 1

            known_words.append(w)

        if len(known_words) == 0:
            raise ValueError("No known words found in input.")

        stats.known_word_count = len(known_words)

        doc = self.nlp(' '.join(known_words))

        stats.inflectional_variants = self._inflectional_variants_per_lemma(doc)

        return stats

    @staticmethod
    def _inflectional_variants_per_lemma(doc: Doc) -> dict:
        lemma_variants = defaultdict(set)

        for token in doc:
            if token.is_alpha:
                morph_features = str(token.morph)
                lemma_variants[token.lemma_].add(morph_features)

        # Convert sets to counts
        variants_count = {lemma: len(variants) for lemma, variants in lemma_variants.items()}
        return variants_count

