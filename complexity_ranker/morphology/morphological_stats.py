from dataclasses import dataclass, field
from typing import Dict

from complexity_ranker.morphology.morphological_weights import MorphoWeights


@dataclass
class MorphoStats:
    total_word_count: int
    total_morpheme_count: int = 0
    complex_word_count: int = 0
    ttr_value: float = 0.0
    hapax_value: float = 0.0
    inflectional_variants: Dict[str, int] = field(default_factory=dict)

    @property
    def mean_morphemes_per_word(self) -> float:
        mean = self.total_morpheme_count / self.total_word_count
        return min(mean / 5, 1.0) # normalize

    @property
    def percentage_complex(self) -> float:
        return self.complex_word_count / self.total_word_count * 100

    @property
    def avg_inflectional_variants(self) -> float:
        if not self.inflectional_variants:
            return 0.0
        mean = sum(self.inflectional_variants.values()) / len(self.inflectional_variants)
        return min(mean / 5, 1.0) # normalize

    @property
    def morphology_score(self) -> float:
        return (
            MorphoWeights.TTR * self.ttr_value +
            MorphoWeights.HAPAX * self.hapax_value +
            MorphoWeights.MEAN_MORPHEMES * self.mean_morphemes_per_word +
            MorphoWeights.PERCENT_COMPLEX * (self.percentage_complex / 100) +
            MorphoWeights.AVG_INFLECTIONAL_VARIANTS * self.avg_inflectional_variants
        )

    def __str__(self) -> str:
        return (
            f"MorphoStats(\n"
            f"  Total words: {self.total_word_count}\n"
            f"  Mean morphemes/word: {self.mean_morphemes_per_word:.2f}\n"
            f"  Complex words: {self.complex_word_count} ({self.percentage_complex:.1f}%)\n"
            f"  TTR: {self.ttr_value:.3f}\n"
            f"  Hapax ratio: {self.hapax_value:.3f}\n"
            f"  Avg inflectional variants: {self.avg_inflectional_variants:.2f}\n"
            f"  Morphology score: {self.morphology_score:.3f}\n"
            f")"
        )