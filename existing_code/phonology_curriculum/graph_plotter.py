import matplotlib.pyplot as plt
import numpy as np

def plot_score_distribution(file_path, bins=100):
    """
    Reads a large text file with lines formatted as:
        float_score \t sentence
    and plots the score distribution as a histogram.

    Args:
        file_path (str): Path to the text file
        bins (int): Number of histogram bins
    """

    # --- First pass: find min and max ---
    score_min, score_max = float("inf"), float("-inf")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                score = float(line.split("\t", 1)[0])
                if score < score_min:
                    score_min = score
                if score > score_max:
                    score_max = score
            except ValueError:
                continue  # skip malformed lines

    print(f"Score range: {score_min} to {score_max}")

    # --- Prepare bins ---
    bin_edges = np.linspace(score_min, score_max, bins + 1)
    counts = np.zeros(bins, dtype=np.int64)

    # --- Second pass: bin the scores ---
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                score = float(line.split("\t", 1)[0])
                # find which bin the score belongs to
                idx = np.searchsorted(bin_edges, score, side="right") - 1
                if 0 <= idx < bins:
                    counts[idx] += 1
            except ValueError:
                continue

    # --- Plot ---
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]), align="center")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.show()


if __name__ == "__main__":
    plot_score_distribution(r"phonology_curriculum_with_scores.txt", bins=100)
