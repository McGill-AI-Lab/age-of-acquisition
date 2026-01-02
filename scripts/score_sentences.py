"""
  This file has examples of how to score sentences.
  This is use to buil curricula.
"""

from sentence_scoring import * # for score_sentence()
from lexical_features import * # for conc()

# scoring sentence based on mean concreteness while attempting multiwords, skipping stopwords, and using the inflected lookup table
output = score_sentence(
  "this is a sentence.",
  metric="conc",
  multiword=True,
  method="mean",
  skip_stopwords=True,
  inflect=True
)
s = output[0] # get just the score
print(s)

# scoring sentence based on concreteness of max-aoa word
output = score_sentence(
  "this is another sentence",
  metric="aoa",
  method="max",
  skip_stopwords=True,
  inflect=True
)
word = output[1] # get just max-aoa word
s = conc(word)
print(s)