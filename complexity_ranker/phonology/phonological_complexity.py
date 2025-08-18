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

        stats = PhonoStats(total_word_count=len(words))
    
        for word in words:
            phones_list = pronouncing.phones_for_word(word)
            if not phones_list:
                stats.unknown_words.append(word)
                continue
    
            phonemes_str = phones_list[0] # first pronunciation
            phonemes = phonemes_str.split()
    
            stats.phoneme_count += len(phonemes)
            syllables = pronouncing.syllable_count(phonemes_str)
            stats.syllable_count += syllables
    
            if syllables >= 3:
                stats.long_word_count += 1
    
            stats.known_word_count += 1
    
            clusters = cls._consonant_clusters(phonemes)
            stats.single_consonant += clusters.count(1)
            stats.clusters_size_2 += clusters.count(2)
            stats.clusters_size_3 += clusters.count(3)
            stats.all_clusters.extend(clusters)
    
        if stats.known_word_count == 0:
            raise ValueError("No known words found in input.")

        return stats
    
