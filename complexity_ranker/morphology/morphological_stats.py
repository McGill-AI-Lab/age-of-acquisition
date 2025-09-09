from dataclasses import dataclass, field
from typing import Dict

from complexity_ranker.morphology.morphological_weights import MorphoWeights


@dataclass
class MorphoStats:
    known_word_count: int = 0
    total_morpheme_count: int = 0
    complex_word_count: int = 0
    inflectional_variants: Dict[str, int] = field(default_factory=dict)

    @property
    def mean_morphemes_per_word(self) -> float:
        return self.total_morpheme_count / self.known_word_count

    @property
    def percentage_complex(self) -> float:
        return self.complex_word_count / self.known_word_count * 100

    @property
    def avg_inflectional_variants(self) -> float:
        if not self.inflectional_variants:
            return 0.0
        return sum(self.inflectional_variants.values()) / len(self.inflectional_variants)

    @property
    def morphology_score(self) -> float:
        return (
            MorphoWeights.MEAN_MORPHEMES * self.mean_morphemes_per_word +
            MorphoWeights.PERCENT_COMPLEX * (self.percentage_complex / 100) +
            MorphoWeights.AVG_INFLECTIONAL_VARIANTS * self.avg_inflectional_variants
        )

    def __str__(self) -> str:
        return (
            f"MorphoStats(\n"
            f"  Known word count: {self.known_word_count}\n"
            f"  Morphemes count: {self.total_morpheme_count}\n"
            f"  Mean morphemes/word: {self.mean_morphemes_per_word:.2f}\n"
            f"  Complex words: {self.complex_word_count} ({self.percentage_complex:.1f}%)\n"
            f"  Avg inflectional variants: {self.avg_inflectional_variants:.2f}\n"
            f"  Morphology score: {self.morphology_score:.3f}\n"
            f")"
        )