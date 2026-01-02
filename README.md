# age-of-acquisition

## Repository Purpose

This repository implements curriculum construction, embedding training, and evaluation for studying how **age-of-acquisition** and **related lexical signals** affect learned word representations.

This project investigates whether the order in which humans learn words, known as Age of Acquisition (AoA), affects the quality and stability of word embeddings trained with Word2Vec. To do so, we compare a standard shuffled curriculum, in which sentences are presented in random order, with an AoA-ordered curriculum that presents sentences in order based on when their words are learned developmentally in children. After training models under both conditions, we evaluate whether the benefits attributed to AoA-ordered learning persist once we account for correlated lexical properties such as word frequency, concreteness, phonological complexity, and semantic coherence. By evaluating stability, intrinsic semantic quality, and extrinsic downstream performance across curriculum conditions we aim to determine whether developmental ordering produces inherently better semantic representations or whether its benefits can be fully explained by correlated linguistic variables
