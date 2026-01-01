from curricula import build_curriculum

out = build_curriculum(
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

print(out)