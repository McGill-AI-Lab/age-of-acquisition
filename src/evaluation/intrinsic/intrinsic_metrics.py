"""
Intrinsic evaluation metrics for word embeddings.

Implements:
- SimLex-999: Similarity evaluation
- WordSim-353: Relatedness evaluation
- Google Analogy Task: Syntactic and semantic analogies
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
from gensim.models import KeyedVectors
from scipy.stats import spearmanr


@dataclass
class IntrinsicResult:
    """Result from an intrinsic evaluation."""
    score: float
    n_evaluated: int
    n_total: int
    coverage: float

    def __repr__(self):
        return f"Score: {self.score:.4f} | Coverage: {self.coverage:.2%} ({self.n_evaluated}/{self.n_total})"


class IntrinsicEvaluator:
    """Evaluates word embeddings on intrinsic benchmarks."""

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Directory containing benchmark datasets
        """
        self.data_dir = Path(data_dir)
        self._simlex_data = None
        self._wordsim_data = None
        self._analogy_data = None

    def load_simlex999(self) -> pd.DataFrame:
        """
        Load SimLex-999 dataset.

        Expected format: TSV with columns word1, word2, SimLex999
        Download from: https://fh295.github.io/simlex.html

        Returns:
            DataFrame with word pairs and similarity scores
        """
        if self._simlex_data is None:
            filepath = self.data_dir / "SimLex-999.txt"
            if not filepath.exists():
                raise FileNotFoundError(
                    f"SimLex-999 not found at {filepath}.\n"
                    "Download from: https://fh295.github.io/simlex.html"
                )
            self._simlex_data = pd.read_csv(filepath, sep="\t")
        return self._simlex_data

    def load_wordsim353(self) -> pd.DataFrame:
        """
        Load WordSim-353 dataset.

        Expected format: CSV with columns Word1, Word2, Human (mean)
        Download from: https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd

        Returns:
            DataFrame with word pairs and relatedness scores
        """
        if self._wordsim_data is None:
            filepath = self.data_dir / "wordsim353.csv"
            if not filepath.exists():
                raise FileNotFoundError(
                    f"WordSim-353 not found at {filepath}.\n"
                    "Download from: https://www.kaggle.com/datasets/julianschelb/wordsim353-crowd"
                )
            self._wordsim_data = pd.read_csv(filepath)
        return self._wordsim_data

    def load_google_analogies(self) -> Dict[str, List[Tuple[str, str, str, str]]]:
        """
        Load Google analogy dataset.

        Expected format: Text file with sections marked by ": " prefix
        Format: word1 word2 word3 word4 (word1:word2 :: word3:word4)

        Download from: https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt

        Returns:
            Dict mapping category -> list of (word1, word2, word3, word4) tuples
        """
        if self._analogy_data is None:
            filepath = self.data_dir / "questions-words.txt"
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Google analogies not found at {filepath}.\n"
                    "Download from: https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt"
                )

            analogies = {}
            current_category = None

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith(":"):
                        current_category = line[1:].strip()
                        analogies[current_category] = []
                    else:
                        parts = line.split()
                        if len(parts) == 4:
                            analogies[current_category].append(tuple(parts))

            self._analogy_data = analogies

        return self._analogy_data

    def evaluate_simlex999(
            self,
            embeddings: KeyedVectors,
            exclude_missing: bool = True
    ) -> IntrinsicResult:
        """
        Evaluate on SimLex-999.

        Args:
            embeddings: KeyedVectors object
            exclude_missing: If True, exclude pairs where either word is missing

        Returns:
            IntrinsicResult with Spearman correlation
        """
        df = self.load_simlex999()

        # Get word columns (handle different naming conventions)
        word1_col = "word1" if "word1" in df.columns else "Word1"
        word2_col = "word2" if "word2" in df.columns else "Word2"
        score_col = "SimLex999" if "SimLex999" in df.columns else "simlex999"

        similarities = []
        human_scores = []

        for _, row in df.iterrows():
            w1, w2 = str(row[word1_col]).lower(), str(row[word2_col]).lower()

            if w1 in embeddings and w2 in embeddings:
                sim = embeddings.similarity(w1, w2)
                similarities.append(sim)
                human_scores.append(row[score_col])
            elif not exclude_missing:
                # Assign 0 similarity if words missing
                similarities.append(0.0)
                human_scores.append(row[score_col])

        if len(similarities) < 2:
            return IntrinsicResult(0.0, 0, len(df), 0.0)

        correlation, _ = spearmanr(similarities, human_scores)

        return IntrinsicResult(
            score=correlation,
            n_evaluated=len(similarities),
            n_total=len(df),
            coverage=len(similarities) / len(df)
        )

    def evaluate_wordsim353(
            self,
            embeddings: KeyedVectors,
            exclude_missing: bool = True
    ) -> IntrinsicResult:
        """
        Evaluate on WordSim-353.

        Args:
            embeddings: KeyedVectors object
            exclude_missing: If True, exclude pairs where either word is missing

        Returns:
            IntrinsicResult with Spearman correlation
        """
        df = self.load_wordsim353()

        # Get word columns (handle different naming conventions)
        word1_col = "Word1" if "Word1" in df.columns else "word1"
        word2_col = "Word2" if "Word2" in df.columns else "word2"
        score_col = "Human (mean)" if "Human (mean)" in df.columns else "similarity"

        similarities = []
        human_scores = []

        for _, row in df.iterrows():
            w1, w2 = str(row[word1_col]).lower(), str(row[word2_col]).lower()

            if w1 in embeddings and w2 in embeddings:
                sim = embeddings.similarity(w1, w2)
                similarities.append(sim)
                human_scores.append(row[score_col])
            elif not exclude_missing:
                similarities.append(0.0)
                human_scores.append(row[score_col])

        if len(similarities) < 2:
            return IntrinsicResult(0.0, 0, len(df), 0.0)

        correlation, _ = spearmanr(similarities, human_scores)

        return IntrinsicResult(
            score=correlation,
            n_evaluated=len(similarities),
            n_total=len(df),
            coverage=len(similarities) / len(df)
        )

    def evaluate_analogies(
            self,
            embeddings: KeyedVectors,
            categories: Optional[List[str]] = None
    ) -> Dict[str, IntrinsicResult]:
        """
        Evaluate on Google analogy task.

        Uses 3CosAdd method: word4 = word2 - word1 + word3

        Args:
            embeddings: KeyedVectors object
            categories: Optional list of categories to evaluate

        Returns:
            Dict mapping category name -> IntrinsicResult with accuracy
        """
        analogies = self.load_google_analogies()

        if categories is not None:
            analogies = {k: v for k, v in analogies.items() if k in categories}

        results = {}

        for category, questions in analogies.items():
            correct = 0
            evaluated = 0

            for w1, w2, w3, w4 in questions:
                w1, w2, w3, w4 = w1.lower(), w2.lower(), w3.lower(), w4.lower()

                # Skip if any word missing
                if not all(w in embeddings for w in [w1, w2, w3, w4]):
                    continue

                evaluated += 1

                # Predict: word4_pred = word2 - word1 + word3
                try:
                    # Get most similar, excluding input words
                    predicted = embeddings.most_similar(
                        positive=[w2, w3],
                        negative=[w1],
                        topn=1,
                        restrict_vocab=None
                    )

                    if predicted and predicted[0][0] == w4:
                        correct += 1

                except Exception:
                    # Handle edge cases (e.g., zero vector)
                    pass

            accuracy = correct / evaluated if evaluated > 0 else 0.0

            results[category] = IntrinsicResult(
                score=accuracy,
                n_evaluated=evaluated,
                n_total=len(questions),
                coverage=evaluated / len(questions) if questions else 0.0
            )

        return results

    def evaluate_all(
            self,
            embeddings: KeyedVectors
    ) -> Dict[str, IntrinsicResult]:
        """
        Run all intrinsic evaluations.

        Args:
            embeddings: KeyedVectors object

        Returns:
            Dict with results for each benchmark
        """
        results = {}

        try:
            results["simlex999"] = self.evaluate_simlex999(embeddings)
        except FileNotFoundError as e:
            print(f"Skipping SimLex-999: {e}")

        try:
            results["wordsim353"] = self.evaluate_wordsim353(embeddings)
        except FileNotFoundError as e:
            print(f"Skipping WordSim-353: {e}")

        try:
            analogy_results = self.evaluate_analogies(embeddings)
            # Compute overall analogy accuracy
            total_correct = sum(r.score * r.n_evaluated for r in analogy_results.values())
            total_evaluated = sum(r.n_evaluated for r in analogy_results.values())

            if total_evaluated > 0:
                results["analogies_overall"] = IntrinsicResult(
                    score=total_correct / total_evaluated,
                    n_evaluated=total_evaluated,
                    n_total=sum(r.n_total for r in analogy_results.values()),
                    coverage=total_evaluated / sum(r.n_total for r in analogy_results.values())
                )

            # Store individual categories
            results["analogies_by_category"] = analogy_results

        except FileNotFoundError as e:
            print(f"Skipping analogies: {e}")

        return results


def evaluate_over_time(
        evaluator: IntrinsicEvaluator,
        embedding_dir: Path,
        curriculum: str,
        tranches: List[int],
        run: int = 1,
        dim: int = 50
) -> pd.DataFrame:
    """
    Evaluate embeddings across training tranches.

    Args:
        evaluator: IntrinsicEvaluator instance
        embedding_dir: Directory with embeddings
        curriculum: Curriculum type
        tranches: List of tranche numbers
        run: Which run to evaluate
        dim: Embedding dimension

    Returns:
        DataFrame with metrics over time
    """
    results = []

    for tranche in tranches:
        filename = f"{curriculum}_tranche{tranche}_run{run}_dim{dim}.txt"
        filepath = embedding_dir / filename

        if not filepath.exists():
            print(f"Skipping {filename} (not found)")
            continue

        embeddings = KeyedVectors.load_word2vec_format(str(filepath), binary=False)
        metrics = evaluator.evaluate_all(embeddings)

        row = {
            "curriculum": curriculum,
            "tranche": tranche,
            "dim": dim,
            "run": run
        }

        # Add scores
        if "simlex999" in metrics:
            row["simlex999"] = metrics["simlex999"].score
            row["simlex999_coverage"] = metrics["simlex999"].coverage

        if "wordsim353" in metrics:
            row["wordsim353"] = metrics["wordsim353"].score
            row["wordsim353_coverage"] = metrics["wordsim353"].coverage

        if "analogies_overall" in metrics:
            row["analogies"] = metrics["analogies_overall"].score
            row["analogies_coverage"] = metrics["analogies_overall"].coverage

        results.append(row)

    return pd.DataFrame(results)


def main():
    """Example usage."""

    evaluator = IntrinsicEvaluator(data_dir=Path("data/benchmarks/"))

    # Load embeddings
    embeddings = KeyedVectors.load_word2vec_format(
        "embeddings/aoa_tranche20_run1_dim50.txt",
        binary=False
    )

    # Evaluate
    results = evaluator.evaluate_all(embeddings)

    print("Intrinsic Evaluation Results:")
    print("=" * 50)
    for name, result in results.items():
        if name != "analogies_by_category":
            print(f"{name}: {result}")


if __name__ == "__main__":
    main()