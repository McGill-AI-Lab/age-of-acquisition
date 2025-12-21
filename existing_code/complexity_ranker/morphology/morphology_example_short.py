from complexity_ranker.morphology.morphological_complexity import MorphologicalComplexity

# Example of a simple sentence
sentence_1 = "This banana is very delicious!"
words_1 = sentence_1.lower().split()
print(MorphologicalComplexity.get_morphology_stats(sentence_1, words_1))

# Example of a complicated sentence
sentence_2 = "Philosophical interpretations of anthropological documentation frequently necessitate comprehensive recontextualization."
words_2 = sentence_2.lower().split()
print(MorphologicalComplexity.get_morphology_stats(sentence_2, words_2))