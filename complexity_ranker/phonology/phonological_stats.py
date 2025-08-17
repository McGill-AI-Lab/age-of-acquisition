from dataclasses import dataclass, field
from typing import List
from complexity_ranker.phonology.phonological_weights import PhonoWeights

@dataclass
class PhonoStats:
    total_word_count: int
    known_word_count: int = 0
    phoneme_count: int = 0
    syllable_count: int = 0
    long_word_count: int = 0
    unknown_words: List[str] = field(default_factory=list)
    all_clusters: List[int] = field(default_factory=list)
    single_consonant: int = 0
    clusters_size_2: int = 0
    clusters_size_3: int = 0

    @property
    def percentage_known(self) -> float:
        return self.known_word_count / self.total_word_count * 100

    @property
    def mean_phoneme_per_word(self) -> float:
        return self.phoneme_count / self.known_word_count

    @property
    def mean_syllable_per_word(self) -> float:
        return self.syllable_count / self.known_word_count

    @property
    def percentage_long(self) -> float:
        return self.long_word_count / self.known_word_count * 100

    @property
    def total_clusters(self) -> int:
        return len(self.all_clusters)

    @property
    def clusters_size_over_3(self) -> int:
        return self.total_clusters - self.single_consonant - self.clusters_size_2 - self.clusters_size_3

    @property
    def phonology_score(self) -> float:
        return (
            PhonoWeights.MEAN_PHONEME * self.mean_phoneme_per_word +
            PhonoWeights.MEAN_SYLLABLE * self.mean_syllable_per_word +
            PhonoWeights.LONG * (self.percentage_long / 100) +
            PhonoWeights.CLUSTER_2 * (self.clusters_size_2 / self.total_clusters) +
            PhonoWeights.CLUSTER_3 * (self.clusters_size_3 / self.total_clusters) +
            PhonoWeights.CLUSTER_OVER_3 * (self.clusters_size_over_3 / self.total_clusters) +
            PhonoWeights.PENALTY_UNKNOWN * (1 - self.percentage_known / 100)  # penalize unknown words
        )

    def __str__(self) -> str:
        return (
            f"PhonoStats(\n"
            f"  Total words: {self.total_word_count}\n"
            f"  Known words: {self.known_word_count} ({self.percentage_known:.1f}%)\n"
            f"  Phonemes: {self.phoneme_count}, Mean per word: {self.mean_phoneme_per_word:.2f}\n"
            f"  Syllables: {self.syllable_count}, Mean per word: {self.mean_syllable_per_word:.2f}\n"
            f"  Long words: {self.long_word_count} ({self.percentage_long:.1f}%)\n"
            f"  Clusters: total={self.total_clusters}, "
            f"size1={self.single_consonant}, "
            f"size2={self.clusters_size_2}, "
            f"size3={self.clusters_size_3}, "
            f"size>3={self.clusters_size_over_3}\n"
            f"  Unknown words: {self.unknown_words}\n"
            f"  Phonology score: {self.phonology_score:.3f}\n"
            f")"
        )
