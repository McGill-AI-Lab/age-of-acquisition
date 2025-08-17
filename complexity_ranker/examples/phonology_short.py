from complexity_ranker.utils import Preprocess
from complexity_ranker.phonology import PhonologicalComplexity

# Example of a simple sentence
sentence_1 = "This banana is very delicious!"
sentence_1_clean = Preprocess.preprocess_text(sentence_1)
print(PhonologicalComplexity.get_phonology_stats(sentence_1_clean))

# Example of a complicated sentence
sentence_2 = "Philosophical interpretations of anthropological documentation frequently necessitate comprehensive recontextualization."
sentence_2_clean = Preprocess.preprocess_text(sentence_2)
print(PhonologicalComplexity.get_phonology_stats(sentence_2_clean))