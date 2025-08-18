import logging
import os

import gensim
import gensim.downloader as api
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
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
        eval_method (str): The evaluation method to use ('analogy' or 'simlex').

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

    # Download the appropriate evaluation dataset.
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

    return corpus, eval_filepath


# 1. Define Hyperparameter Search Space
# These are the Word2Vec hyperparameters we want to tune.
search_space = [
    Integer(50, 300, name='vector_size'),  # Dimensionality of the word vectors.
    Integer(2, 10, name='window'),  # Max distance between current and predicted word.
    Integer(1, 10, name='min_count'),  # Ignores words with frequency lower than this.
    Categorical([0, 1], name='sg'),  # 0 for CBOW, 1 for Skip-Gram.
    Real(1e-5, 1e-3, "log-uniform", name='sample'),  # Downsampling of high-frequency words.
    Integer(5, 20, name='negative'),  # Number of "noise words" for negative sampling.
    Integer(5, 15, name='epochs')  # Number of iterations over the corpus.
]

def create_objective_function(training_corpus, eval_filepath, search_space_dims, eval_method='analogy'):
    """
    Factory function that creates and returns the objective function.
    """
    @use_named_args(search_space_dims)
    def objective_function(vector_size, window, min_count, sg, sample, negative, epochs):
        """
        Objective function to be minimized by scikit-optimize.
        """
        params = {
            'vector_size': vector_size, 'window': window, 'min_count': min_count,
            'sg': sg, 'sample': sample, 'negative': negative, 'epochs': epochs
        }
        print(f"\nTesting parameters: {params}")

        model = Word2Vec(
            sentences=training_corpus,
            workers=os.cpu_count(),
            hs=0,
            **params
        )

        score = 1.0  # Default to a bad score
        if eval_filepath:
            try:
                if eval_method == 'analogy':
                    eval_result = model.wv.evaluate_word_analogies(eval_filepath)
                    accuracy = eval_result[0]
                    score = -accuracy  # Minimize negative accuracy
                    print(f"Analogy accuracy: {accuracy:.4f} -> Score: {score:.4f}")

                elif eval_method == 'simlex':
                    # evaluate_word_pairs returns (pearson, spearman, oov_ratio)
                    correlations = model.wv.evaluate_word_pairs(eval_filepath)
                    spearman_corr = correlations[1][0]
                    score = -spearman_corr  # Minimize negative correlation
                    print(f"SimLex-999 Spearman correlation: {spearman_corr:.4f} -> Score: {score:.4f}")

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
    EVALUATION_METHOD = 'simlex'

    my_corpus_file = None  # Set to a file path to use a local corpus
    training_corpus, eval_filepath = load_data(
        corpus_path=my_corpus_file,
        eval_method=EVALUATION_METHOD
    )

    objective = create_objective_function(training_corpus, eval_filepath, search_space, eval_method=EVALUATION_METHOD)

    if not eval_filepath:
        print("\nWARNING: Could not download the evaluation dataset.")
        print("The optimization will proceed using a simple similarity score ('king' vs 'queen').")
        print("This is NOT a robust evaluation method. For real use cases, use a proper evaluation set.\n")

    # 3. Run Bayesian Optimization
    n_calls = 25  # Number of times to call the objective function.

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
    # The result.fun is the minimum value of the objective function (our negative accuracy).
    if EVALUATION_METHOD == 'simlex':
        print(f"Best Spearman Correlation: {-result.fun:.4f}")
    else:
        print(f"Best Analogy Accuracy: {-result.fun:.4f}")

    print("Best parameters found:")
    best_params = dict(zip([dim.name for dim in search_space], result.x))
    for param, value in best_params.items():
        print(f"- {param}: {value}")

    # 5. Visualize the Results of BO
    print("\nPlotting convergence...")
    plot_convergence(result)

    # Set a dynamic title for the y-axis based on the evaluation method
    ylabel = "Objective value"
    if EVALUATION_METHOD == 'simlex':
        ylabel += " (-Spearman Correlation)"
    else:
        ylabel += " (-Accuracy)"

    plt.title("Convergence Plot")
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.show()

    print("Plotting objective function (may be slow)...")
    _ = plot_objective(result, n_points=10)

    plt.show()
