from dataclasses import dataclass, field
from typing import List

from complexity_ranker.phonology.phonological_weights import PhonoWeights


@dataclass
class PhonoStats:
    known_word_count: int = 0
    phoneme_count: int = 0
    syllable_count: int = 0
    long_word_count: int = 0
    unknown_words: List[str] = field(default_factory=list)
    total_clusters: int = 0
    clusters_size_1: int = 0
    clusters_size_2: int = 0
    clusters_size_3: int = 0

    @property
    def mean_phoneme_per_word(self) -> float:
        return self.phoneme_count / self.known_word_count

    @property
    def mean_syllable_per_word(self) -> float:
        return self.syllable_count / self.known_word_count

    @property
    def percentage_long(self) -> float:
        return self.long_word_count / self.known_word_count

    @property
    def clusters_size_over_3(self) -> int:
        return self.total_clusters - self.clusters_size_1 - self.clusters_size_2 - self.clusters_size_3

    @property
    def phonology_score(self) -> float:
        return (
            PhonoWeights.MEAN_PHONEME * self.mean_phoneme_per_word +
            PhonoWeights.MEAN_SYLLABLE * self.mean_syllable_per_word +
            PhonoWeights.LONG * self.percentage_long / 100 +
            PhonoWeights.CLUSTER_2 * (self.clusters_size_2 / self.total_clusters) +
            PhonoWeights.CLUSTER_3 * (self.clusters_size_3 / self.total_clusters) +
            PhonoWeights.CLUSTER_OVER_3 * (self.clusters_size_over_3 / self.total_clusters)
        )

    def __str__(self) -> str:
        return (
            f"PhonoStats(\n"
            f"  Known words: {self.known_word_count}\n"
            f"  Mean phonemes per word: {self.mean_phoneme_per_word:.2f}\n"
            f"  Mean syllable per word: {self.mean_syllable_per_word:.2f}\n"
            f"  Long words: {self.long_word_count}\n"
            f"  Clusters: total={self.total_clusters}, "
            f"size1={self.clusters_size_1}, "
            f"size2={self.clusters_size_2}, "
            f"size3={self.clusters_size_3}, "
            f"size>3={self.clusters_size_over_3}\n"
            f"  Phonology score: {self.phonology_score:.3f}\n"
            f")"
        )
