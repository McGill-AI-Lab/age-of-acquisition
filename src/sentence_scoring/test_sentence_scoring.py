from sentence_scoring import score_sentence
from lexical_features import conc

def run():
  print(score_sentence(
    "this is a sentence.",
    metric="conc",
    multiword=True,
    method="mean",
    skip_stopwords=True,
    inflect=True
  ))

  out = score_sentence(
    "this is another sentence",
    metric="aoa",
    method="max",
    skip_stopwords=True,
    inflect=True
  )
  print(out)
  print("conc(witness) =", conc(out[1], inflect=True) if out[1] else None)

  # Stopword-starting phrase behavior:
  print(score_sentence(
    "in the end we win.",
    metric="conc",
    multiword=True,
    method="min",
    skip_stopwords=True,
    inflect=True
  ))

  print(score_sentence(
    "bsdfjk fn jdsnfksndafkasd...",
    metric="aoa",
    method="add",
    inflect=False
  ))

if __name__ == "__main__":
  run()
