"""
  Running this file builds the corpus.

  Instructions to set up are in src/preprocessing/README.md
"""
from preprocessing import *

# build 50 shards
# entire BabyLM
# random 10% of refined book corpus 
load_corpus_shards(n_shards=50, percent_refined=10)