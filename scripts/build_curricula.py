"""
  This file shows how to build, validate, and shuffle curricula.

  !! These are just examples !!
  Edit the code and change things to get the desired curricula.
"""

from curricula import *

# build the curriculum into data/processed/corpora/training/
idx = build_curriculum(
  curriculum="aoa",
  scoring_method="max",
  sort_order="asc",
  tranche_type="word-based",
  tranche_size=500,
  aoa_agnostic=True, # don't care
  multiword=False, # don't care
  skip_stopwords=True,
  inflect=True,
  duplication_cap=5,
)

# build a corresponding *matching* shuffled curriculum
idx2 = build_curriculum(
  curriculum="shuffled",
  tranche_type="matching",
  matching_idx=idx,
  duplication_cap=5,
)

# curriculum index
print(idx)

# display tranche size 
plot_tranche_sizes(idx, metric="word")
plot_tranche_sizes(idx, metric="sentence")

# write samples.txt
write_samples(idx)

# shuffle the tranches in-place
# done before training
shuffle_tranches(idx)