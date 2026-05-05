"""
  This file shows how to make a fake aoa curriculum for controlled study.
"""
from curricula import *

# build the curriculum into data/processed/corpora/training/
idx = build_curriculum(
  curriculum="aoa",
  scoring_method="mean",
  sort_order="asc",
  tranche_type="word-count",
  tranche_size=40_000,
  aoa_agnostic=True, # don't care
  multiword=False, # don't care
  skip_stopwords=True,
  inflect=True,
  duplication_cap=5,
  max_tranches=-1,
  fake_aoa_seed=62,
)

# build a corresponding *matching* shuffled curriculum
idx2 = build_curriculum(
  curriculum="shuffled",
  tranche_type="word-count",
  tranche_size=40_000,
  duplication_cap=5,
  max_tranches=-1,
)

# curriculum index
print(idx)
print(idx2)

# display tranche size 
plot_tranche_sizes(idx, metric="word")
plot_tranche_sizes(idx, metric="sentence")

plot_tranche_sizes(idx2, metric="word")
plot_tranche_sizes(idx2, metric="sentence")

# write samples.txt
write_samples(idx)
write_samples(idx2)

# shuffle the tranches in-place
# done before training
# shuffle_tranches(idx)