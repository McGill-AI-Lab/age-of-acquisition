from collections import Counter
from collections import defaultdict
from typing import List

import morfessor
import spacy
from spacy.tokens import Doc

from complexity_ranker.morphology.morphological_stats import MorphoStats


class MorphologicalComplexity:
    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file('model.bin')
    nlp = spacy.load("en_core_web_sm")

    @classmethod
    def get_morphology_stats(cls, text: str, words: List[str]) -> MorphoStats:
        if not words:
            raise ValueError("Please provide a non-empty input.")

        stats = MorphoStats(total_word_count=len(words))

        for w in words:
            morphemes, confidence = cls.model.viterbi_segment(w)
            stats.total_morpheme_count += len(morphemes)

            if len(morphemes) > 1:
                stats.complex_word_count += 1

        doc = cls.nlp(text)

        stats.ttr_value = cls.type_token_ratio(doc)
        stats.inflectional_variants = cls.inflectional_variants_per_lemma(doc)
        stats.hapax_value = cls.hapax_legomena_ratio(doc)

        return stats

    @classmethod
    def type_token_ratio(cls, doc: Doc) -> float:
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        total_tokens = len(tokens)
        unique_tokens = len(set(tokens))

        if total_tokens == 0:
            return 0.0

        return unique_tokens / total_tokens

    @classmethod
    def inflectional_variants_per_lemma(cls, doc: Doc) -> dict:
        lemma_variants = defaultdict(set)

        for token in doc:
            if token.is_alpha:
                morph_features = str(token.morph)
                lemma_variants[token.lemma_].add(morph_features)

        # Convert sets to counts
        variants_count = {lemma: len(variants) for lemma, variants in lemma_variants.items()}
        return variants_count

    @classmethod
    def hapax_legomena_ratio(cls, doc: Doc) -> float:

        # Choose lemmas or raw words
        tokens = [token.text.lower() for token in doc if token.is_alpha]

        counts = Counter(tokens)

        # Hapaxes: words appearing only once
        hapaxes = [word for word, freq in counts.items() if freq == 1]

        # Ratio
        ratio = len(hapaxes) / len(tokens)
        return ratio

