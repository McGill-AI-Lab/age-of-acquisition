"""
  This file shows how to build, validate, and shuffle curricula.
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
  inflect=True
)

# curriculum index
print(idx)

# display tranche sized (must exit to continue script)
plot_tranche_sizes(idx)

# write samples.txt
write_samples(idx)

# shuffle the tranches in-place
# done before training
shuffle_tranches(idx)