"""
Stability metrics for word embeddings across training runs.

Measures nearest neighbor overlap to assess embedding stability.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine


class StabilityEvaluator:
    """Evaluates stability of word embeddings across multiple training runs."""

    def __init__(self, embedding_dir: Path, k_values: List[int]):
        """
        Args:
            embedding_dir: Directory containing embedding files
            k_values: List of k values for k-nearest neighbors
        """
        self.embedding_dir = Path(embedding_dir)
        self.k_values = k_values

    def load_embeddings(
        self,
        curriculum: str,
        tranche: int,
        run: int,
        dim: int = 50
    ) -> KeyedVectors:
        """
        Load embeddings from a specific curriculum, tranche, and run.

        Expected filename format: {curriculum}_tranche{tranche}_run{run}_dim{dim}.txt
        E.g., "aoa_tranche10_run1_dim50.txt"

        Args:
            curriculum: Curriculum type (e.g., "aoa", "shuffled", "conc")
            tranche: Tranche number
            run: Run number (1-5)
            dim: Embedding dimension (50 or 300)

        Returns:
            Loaded KeyedVectors object
        """
        filename = f"{curriculum}_tranche{tranche}_run{run}_dim{dim}.txt"
        filepath = self.embedding_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Embedding file not found: {filepath}")

        # Load in word2vec text format
        return KeyedVectors.load_word2vec_format(str(filepath), binary=False)

    def get_nearest_neighbors(
        self,
        embeddings: KeyedVectors,
        word: str,
        k: int
    ) -> List[str]:
        """
        Get k nearest neighbors for a word.

        Args:
            embeddings: KeyedVectors object
            word: Target word
            k: Number of neighbors

        Returns:
            List of k nearest neighbor words (excluding the word itself)
        """
        if word not in embeddings:
            return []

        # Get k+1 neighbors (including the word itself) and exclude the word
        neighbors = embeddings.most_similar(word, topn=k+1)
        return [w for w, _ in neighbors if w != word][:k]

    def compute_neighbor_overlap(
        self,
        neighbors1: List[str],
        neighbors2: List[str]
    ) -> float:
        """
        Compute overlap between two neighbor lists.

        Args:
            neighbors1: First list of neighbors
            neighbors2: Second list of neighbors

        Returns:
            Overlap ratio (intersection / union)
        """
        if not neighbors1 or not neighbors2:
            return 0.0

        set1, set2 = set(neighbors1), set(neighbors2)
        intersection = len(set1 & set2)

        # Return proportion of overlap relative to k
        k = len(neighbors1)
        return intersection / k if k > 0 else 0.0

    def compute_pairwise_stability(
        self,
        curriculum: str,
        tranche: int,
        runs: List[int],
        dim: int = 50,
        k: int = 30,
        vocab_subset: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute pairwise stability across all pairs of runs.

        Args:
            curriculum: Curriculum type
            tranche: Tranche number
            runs: List of run numbers to compare
            dim: Embedding dimension
            k: Number of nearest neighbors
            vocab_subset: Optional list of words to evaluate (e.g., SimLex words)

        Returns:
            Dict with stability statistics
        """
        # Load all runs
        embeddings_list = []
        for run in runs:
            emb = self.load_embeddings(curriculum, tranche, run, dim)
            embeddings_list.append(emb)

        # Get common vocabulary across all runs
        common_vocab = set(embeddings_list[0].key_to_index.keys())
        for emb in embeddings_list[1:]:
            common_vocab &= set(emb.key_to_index.keys())

        # Filter to vocab subset if provided
        if vocab_subset:
            common_vocab &= set(vocab_subset)

        common_vocab = sorted(common_vocab)

        if not common_vocab:
            return {
                "mean_overlap": 0.0,
                "std_overlap": 0.0,
                "n_words": 0,
                "n_pairs": 0
            }

        # Compute pairwise overlaps
        overlaps = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                word_overlaps = []
                for word in common_vocab:
                    neighbors_i = self.get_nearest_neighbors(embeddings_list[i], word, k)
                    neighbors_j = self.get_nearest_neighbors(embeddings_list[j], word, k)
                    overlap = self.compute_neighbor_overlap(neighbors_i, neighbors_j)
                    word_overlaps.append(overlap)

                # Average overlap for this pair of runs
                pair_overlap = np.mean(word_overlaps) if word_overlaps else 0.0
                overlaps.append(pair_overlap)

        return {
            "mean_overlap": np.mean(overlaps) if overlaps else 0.0,
            "std_overlap": np.std(overlaps) if overlaps else 0.0,
            "n_words": len(common_vocab),
            "n_pairs": len(overlaps)
        }

    def compute_stability_over_time(
        self,
        curriculum: str,
        tranches: List[int],
        runs: List[int] = [1, 2, 3, 4, 5],
        dim: int = 50,
        k: int = 30
    ) -> pd.DataFrame:
        """
        Compute stability metrics across multiple tranches.

        Args:
            curriculum: Curriculum type
            tranches: List of tranche numbers to evaluate
            runs: List of run numbers
            dim: Embedding dimension
            k: Number of nearest neighbors

        Returns:
            DataFrame with columns: tranche, mean_overlap, std_overlap, n_words
        """
        results = []

        for tranche in tranches:
            stats = self.compute_pairwise_stability(
                curriculum, tranche, runs, dim, k
            )
            results.append({
                "tranche": tranche,
                "mean_overlap": stats["mean_overlap"],
                "std_overlap": stats["std_overlap"],
                "n_words": stats["n_words"]
            })

        return pd.DataFrame(results)

    def compare_curricula_stability(
        self,
        curricula: List[str],
        tranches: List[int],
        runs: List[int] = [1, 2, 3, 4, 5],
        dim: int = 50,
        k: int = 30
    ) -> pd.DataFrame:
        """
        Compare stability across different curricula.

        Args:
            curricula: List of curriculum types to compare
            tranches: List of tranche numbers
            runs: List of run numbers
            dim: Embedding dimension
            k: Number of nearest neighbors

        Returns:
            DataFrame with stability metrics for each curriculum
        """
        all_results = []

        for curriculum in curricula:
            df = self.compute_stability_over_time(curriculum, tranches, runs, dim, k)
            df["curriculum"] = curriculum
            df["k"] = k
            df["dim"] = dim
            all_results.append(df)

        return pd.concat(all_results, ignore_index=True)

    def compute_word_cohort_movement(
        self,
        curriculum: str,
        word_cohorts: Dict[int, List[str]],
        tranches: List[int],
        run: int = 1,
        dim: int = 50
    ) -> pd.DataFrame:
        """
        Compute how much word cohorts (introduced at same tranche) move over time.

        Args:
            curriculum: Curriculum type
            word_cohorts: Dict mapping tranche -> list of words introduced
            tranches: List of tranches to track
            run: Which run to analyze
            dim: Embedding dimension

        Returns:
            DataFrame with movement statistics per cohort per tranche
        """
        results = []

        for intro_tranche, words in word_cohorts.items():
            prev_embeddings = None

            for tranche in tranches:
                if tranche < intro_tranche:
                    continue

                curr_embeddings = self.load_embeddings(curriculum, tranche, run, dim)

                if prev_embeddings is not None:
                    # Compute movement for this cohort
                    movements = []
                    for word in words:
                        if word in prev_embeddings and word in curr_embeddings:
                            prev_vec = prev_embeddings[word]
                            curr_vec = curr_embeddings[word]
                            # Cosine distance
                            dist = cosine(prev_vec, curr_vec)
                            movements.append(dist)

                    if movements:
                        results.append({
                            "intro_tranche": intro_tranche,
                            "current_tranche": tranche,
                            "mean_movement": np.mean(movements),
                            "std_movement": np.std(movements),
                            "n_words": len(movements)
                        })

                prev_embeddings = curr_embeddings

        return pd.DataFrame(results)


def main():
    """Example usage of StabilityEvaluator."""

    # Example: Evaluate stability for a curriculum
    evaluator = StabilityEvaluator(
        embedding_dir=Path("embeddings/"),
        k_values=[10, 30, 100]
    )

    # Compare shuffled vs AoA curriculum
    curricula = ["shuffled", "aoa"]
    tranches = list(range(1, 21))  # First 20 tranches

    df = evaluator.compare_curricula_stability(
        curricula=curricula,
        tranches=tranches,
        runs=[1, 2, 3, 4, 5],
        dim=50,
        k=30
    )

    print("Stability comparison:")
    print(df.groupby("curriculum")[["mean_overlap"]].mean())

    # Analyze final tranche stability
    final_tranche = 20
    for curriculum in curricula:
        stats = evaluator.compute_pairwise_stability(
            curriculum=curriculum,
            tranche=final_tranche,
            runs=[1, 2, 3, 4, 5],
            dim=50,
            k=30
        )
        print(f"\n{curriculum} final stability: {stats['mean_overlap']:.3f} ± {stats['std_overlap']:.3f}")


if __name__ == "__main__":
    main()