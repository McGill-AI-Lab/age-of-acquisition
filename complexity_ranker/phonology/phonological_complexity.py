import pronouncing

from complexity_ranker.phonology.phonological_stats import PhonoStats


class PhonologicalComplexity:
    @classmethod
    def _is_vowel(cls, phoneme: str) -> bool:
        """Check if a phoneme is a vowel (ARPAbet vowels all end with a digit indicating stress)."""
        return phoneme[-1].isdigit()
    
    @classmethod
    def _consonant_clusters(cls, phonemes: list[str]) -> list[int]:
        """Find consonant clusters in a phoneme list."""
        clusters = []
        current_length = 0
    
        for p in phonemes:
            if cls._is_vowel(p):
                if current_length > 0:
                    clusters.append(current_length)
                    current_length = 0
            else:
                current_length += 1
    
        if current_length > 0:
            clusters.append(current_length)
    
        return clusters
    
    @classmethod
    def get_phonology_stats(cls, words: list[str]) -> PhonoStats:
        """Analyze phonological complexity of a text and return its phonology stats."""
        if not words:
            raise ValueError("Please provide a non-empty input.")

        stats = PhonoStats()

        for word in words:
            # Get list of possible pronunciations
            phones_list = pronouncing.phones_for_word(word)

            # Skip words that are not recognized by the pronouncing package
            if not phones_list:
                continue

            # Get the number of phonemes in the first pronunciation
            phonemes_str = phones_list[0]
            phonemes = phonemes_str.split()
            stats.phoneme_count += len(phonemes)

            # Get the number of syllables + the number of words with at least 3 syllables
            syllables = pronouncing.syllable_count(phonemes_str)
            stats.syllable_count += syllables
    
            if syllables >= 3:
                stats.long_word_count += 1
    
            # Get stats on consonant clusters
            clusters = cls._consonant_clusters(phonemes)
            stats.clusters_size_1 += clusters.count(1)
            stats.clusters_size_2 += clusters.count(2)
            stats.clusters_size_3 += clusters.count(3)
            stats.total_clusters += len(clusters)

            # Get known word count
            stats.known_word_count += 1
    
        if stats.known_word_count == 0 or stats.total_clusters == 0:
            raise ValueError("No known words found in input.")

        return stats
    
