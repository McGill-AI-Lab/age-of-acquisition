import logging
import os
import random

import gensim
import gensim.downloader as api
import matplotlib.pyplot as plt
from gensim.models import FastText
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Set up logging to see gensim's output during training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_data(corpus_path=None, eval_method='analogy'):
    """
    Loads data for training and evaluation.
    If corpus_path is provided, it loads a local text file.
    Otherwise, it downloads the 'text8' corpus from gensim as a default.

    Args:
        eval_method (str): The evaluation method to use ('analogy', 'simlex', 'bats', 'cosine').

    Returns the training corpus and the path to the evaluation file.
    """
    if corpus_path:
        print(f"Loading custom corpus from: {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = [line.strip().split() for line in f]
    else:
        print("No custom corpus path provided. Downloading 'text8' from gensim.")
        try:
            corpus = api.load('text8')
        except Exception as e:
            print(f"Could not download 'text8' corpus: {e}")
            corpus = [["placeholder"]]

    # Download/locate the appropriate evaluation dataset.
    eval_filepath = None
    if eval_method == 'analogy':
        print("Loading 'word2vec-google-analogy-testset' for evaluation.")
        try:
            eval_filepath = api.load('word2vec-google-analogy-testset').fn
        except Exception as e:
            print(f"Could not download analogy test set: {e}")
    elif eval_method == 'simlex':
        print("Loading 'SimLex-999' for evaluation.")
        try:
            eval_filepath = api.load('simlex999').fn
        except Exception as e:
            print(f"Could not download SimLex-999 dataset: {e}")
    elif eval_method == 'bats':
        # Expect a local BATS dataset directory. Try env var, then ./BATS
        bats_dir = os.getenv('BATS_DIR') or os.path.join(os.getcwd(), 'BATS')
        if os.path.isdir(bats_dir):
            print(f"Using local BATS directory for evaluation: {bats_dir}")
            eval_filepath = bats_dir
        else:
            print("BATS directory not found. Set environment variable 'BATS_DIR' to point to your local BATS dataset.")
    elif eval_method == 'cosine':
        # No external eval file needed
        eval_filepath = None

    return corpus, eval_filepath


# 1. Define Hyperparameter Search Space
# These are the FastText hyperparameters we want to tune.
search_space = [
    Integer(50, 300, name='vector_size'),  # Dimensionality of the word vectors.
    Integer(2, 10, name='window'),  # Max distance between current and predicted word.
    Integer(1, 10, name='min_count'),  # Ignores words with frequency lower than this.
    Categorical([0, 1], name='sg'),  # 0 for CBOW, 1 for Skip-Gram.
    Real(1e-5, 1e-3, "log-uniform", name='sample'),  # Downsampling of high-frequency words.
    Integer(5, 20, name='negative'),  # Number of "noise words" for negative sampling.
    Integer(5, 15, name='epochs'),  # Number of iterations over the corpus.
    Integer(2, 5, name='min_n'),  # Min length of char n-grams.
    Integer(3, 7, name='max_n')   # Max length of char n-grams.
]

def evaluate_bats_accuracy(model: FastText, bats_dir: str, max_pairs_per_category: int = 50, comparisons_per_category: int = 100) -> float:
    """
    Lightweight BATS evaluation. For each category file, sample pairs and form analogies
    a:b::c:? expecting d, and measure top-1 accuracy.

    Returns accuracy in [0, 1]. If no valid comparisons, returns 0.0.
    """
    if not os.path.isdir(bats_dir):
        return 0.0

    random.seed(42)
    total = 0
    correct = 0

    for root, _dirs, files in os.walk(bats_dir):
        for filename in files:
            if not filename.lower().endswith('.txt'):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    pairs = []
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        a, b = parts[0], parts[1]
                        pairs.append((a, b))
            except Exception:
                continue

            if not pairs:
                continue
            if len(pairs) > max_pairs_per_category:
                pairs = pairs[:max_pairs_per_category]

            if len(pairs) < 2:
                continue

            # Sample a fixed number of cross-analogies per category for speed
            for _ in range(comparisons_per_category):
                (a1, b1), (a2, b2) = random.sample(pairs, 2)
                try:
                    # FastText can handle OOV words, so KeyError is less likely for analogies
                    # unless the words are completely out of character set.
                    predicted = model.wv.most_similar(positive=[b1, a2], negative=[a1], topn=1)
                    if predicted and predicted[0][0] == b2:
                        correct += 1
                    total += 1
                except KeyError:
                    # One of the words OOV; skip counting
                    continue
                except Exception:
                    continue

    if total == 0:
        return 0.0
    return correct / total


def create_objective_function(training_corpus, eval_filepath, search_space_dims, eval_method='analogy'):
    """
    Factory function that creates and returns the objective function.
    """
    @use_named_args(search_space_dims)
    def objective_function(vector_size, window, min_count, sg, sample, negative, epochs, min_n, max_n):
        """
        Objective function to be minimized by scikit-optimize.
        """
        # FastText requires min_n <= max_n. We'll enforce it here.
        if min_n > max_n:
            min_n, max_n = max_n, min_n

        params = {
            'vector_size': vector_size, 'window': window, 'min_count': min_count,
            'sg': sg, 'sample': sample, 'negative': negative, 'epochs': epochs,
            'min_n': min_n, 'max_n': max_n
        }
        print(f"\nTesting parameters: {params}")

        # Train model differently depending on evaluation method
        if eval_method == 'cosine':
            # Manual epoch-by-epoch training to track similarity evolution
            model = FastText(
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                sg=sg,
                sample=sample,
                negative=negative,
                min_n=min_n,
                max_n=max_n,
                workers=os.cpu_count(),
                hs=0
            )
            model.build_vocab(training_corpus)
        else:
            model = FastText(
                sentences=training_corpus,
                workers=os.cpu_count(),
                hs=0,
                **params
            )

        score = 1.0  # Default to a bad score
        if eval_method == 'cosine':
            pair_env = os.getenv('COSINE_PAIR', 'king,queen')
            try:
                w1, w2 = [w.strip() for w in pair_env.split(',')]
            except Exception:
                w1, w2 = 'king', 'queen'

            similarities = []
            for _epoch in range(epochs):
                # Train one epoch at a time
                model.train(training_corpus, total_examples=model.corpus_count, epochs=1)
                try:
                    sim = model.wv.similarity(w1, w2)
                    similarities.append(sim)
                except KeyError:
                    # This is less likely with FastText but could happen if chars are not in vocab
                    continue
            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                score = -avg_sim
                print(f"Cosine similarity across epochs for ({w1}, {w2}): avg={avg_sim:.4f} -> Score: {score:.4f}")
            else:
                score = 1.0
                print(f"Cosine similarity across epochs for ({w1}, {w2}): no valid measurements -> Score: {score:.4f}")

        elif eval_filepath:
            try:
                if eval_method == 'analogy':
                    # FastText is good at analogies, even with OOV words.
                    eval_result = model.wv.evaluate_word_analogies(eval_filepath)
                    accuracy = eval_result[0]
                    score = -accuracy  # Minimize negative accuracy
                    print(f"Analogy accuracy: {accuracy:.4f} -> Score: {score:.4f}")

                elif eval_method == 'simlex':
                    # evaluate_word_pairs returns (pearson, spearman, oov_ratio)
                    # OOV ratio should be 0.0 with FastText.
                    correlations = model.wv.evaluate_word_pairs(eval_filepath)
                    spearman_corr = correlations[1][0]
                    score = -spearman_corr  # Minimize negative correlation
                    print(f"SimLex-999 Spearman correlation: {spearman_corr:.4f} -> Score: {score:.4f}")

                elif eval_method == 'bats':
                    bats_acc = evaluate_bats_accuracy(model, eval_filepath)
                    score = -bats_acc
                    print(f"BATS accuracy: {bats_acc:.4f} -> Score: {score:.4f}")

            except Exception as e:
                print(f"Evaluation failed for params {params}: {e}")
                score = 1.0  # Penalize failures
        else:
            # Fallback if evaluation file is not available
            try:
                similarity = model.wv.similarity('king', 'queen')
                score = -similarity
                print(f"Fallback similarity ('king', 'queen'): {similarity:.4f} -> Score: {score:.4f}")
            except KeyError:
                score = 1.0

        return score

    return objective_function


if __name__ == '__main__':
    # --- CHOOSE YOUR EVALUATION METHOD ---
    # 'analogy': Good for syntactic/semantic relations.
    # 'simlex': Good for pure semantic similarity.
    # 'bats': Evaluate morphological/semantic analogies from a local BATS dataset.
    # 'cosine': Track cosine similarity of a word pair across epochs (set COSINE_PAIR env var like "king,queen").
    EVALUATION_METHOD = 'bats'

    my_corpus_file = None  # Set to a file path to use a local corpus
    training_corpus, eval_filepath = load_data(
        corpus_path=my_corpus_file,
        eval_method=EVALUATION_METHOD
    )

    objective = create_objective_function(training_corpus, eval_filepath, search_space, eval_method=EVALUATION_METHOD)

    if EVALUATION_METHOD in ('analogy', 'simlex', 'bats') and not eval_filepath:
        print("\nWARNING: Could not load the requested evaluation dataset.")
        print("The optimization will proceed using a simple similarity score ('king' vs 'queen').")
        print("This is NOT a robust evaluation method. For real use cases, use a proper evaluation set.\n")

    # 3. Run Bayesian Optimization
    n_calls = 5  # Number of times to call the objective function. 5 is the minimum number of calls.

    print(f"Starting Bayesian Optimization for {n_calls} iterations...")

    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        acq_func='gp_hedge',  # Explicitly using the default acquisition function
        n_calls=n_calls,
        n_initial_points=5,  # How many steps to run randomly before fitting the model
        random_state=42,
        verbose=True
    )

    # 4. Analyze the Results
    print("\nOptimization finished.")
    # The result.fun is the minimum value of the objective function (negated metric).
    if EVALUATION_METHOD == 'simlex':
        print(f"Best Spearman Correlation: {-result.fun:.4f}")
    elif EVALUATION_METHOD == 'analogy':
        print(f"Best Analogy Accuracy: {-result.fun:.4f}")
    elif EVALUATION_METHOD == 'bats':
        print(f"Best BATS Accuracy: {-result.fun:.4f}")
    elif EVALUATION_METHOD == 'cosine':
        print(f"Best Average Cosine Similarity: {-result.fun:.4f}")

    print("Best parameters found:")
    best_params = dict(zip([dim.name for dim in search_space], result.x))
    for param, value in best_params.items():
        print(f"- {param}: {value}")

    # 5. Visualize the Results
    print("\nPlotting convergence...")
    plot_convergence(result)

    # Set a dynamic title for the y-axis based on the evaluation method
    ylabel = "Objective value"
    if EVALUATION_METHOD == 'simlex':
        ylabel += " (-Spearman Correlation)"
    elif EVALUATION_METHOD in ('analogy', 'bats'):
        ylabel += " (-Accuracy)"
    elif EVALUATION_METHOD == 'cosine':
        ylabel += " (-Avg Cosine Similarity)"

    plt.title("Convergence Plot")
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.show()

    print("Plotting objective function (may be slow)...")
    _ = plot_objective(result, n_points=10)
    plt.show()