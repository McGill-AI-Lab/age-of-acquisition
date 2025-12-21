"""
This file loads data/outputs/word_concreteness_means.parquet into a dict {word: mean_rating}
Scale ranges from 1 to 5.
1 ~ abstract
5 ~ concrete

Usage:
from create_concreteness_lookup import load_concreteness_word_ratings

Running this file will load concreteness ratings and plot the distribution.
"""

import os
from os import PathLike
from pathlib import Path
from typing import Dict, Union, Mapping, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

script_path = Path(__file__).resolve()
script_dir = script_path.parent

__all__ = ["load_concreteness_word_ratings"]

def load_concreteness_word_ratings (
    path: Union[str, Path] = Path(script_dir, "data/outputs/word_concreteness_mean.parquet")
) -> Dict[str, float]:

  # Confirm path
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"File not found: {path}")

  # Confirm parquet dependencies
  try:
    df = pd.read_parquet(path)[["Word", "rating_mean"]]
  except ImportError as e:
    raise ImportError("Reading Parquet requires 'pyarrow' or 'fastparquet'") from e

  if not df["Word"].is_unique:
    raise ValueError("Duplicate words found.")

  return (
    df.set_index("Word")["rating_mean"]
      .round(2)
      .astype(float)
      .to_dict()
  )

def plot_rating_distribution(
  ratings: Mapping[str, float],
  *,
  bins: Union[int, str] = "auto",
  round_to: Optional[int] = None,
  discrete: bool = False,
  density: bool = False,
  figsize: Tuple[float, float] = (8, 5),
  title: Optional[str] = "Concreteness score distribution",
  show: bool = True,
  save_path: Optional[Union[str, PathLike]] = None,
):
  """
  Plot a distribution of word ratings: x = rating, y = frequency.

  Parameters
  ----------
  ratings : Mapping[str, float]
      Dict-like mapping {word: rating}.
  bins : int or str, default "auto"
      Number of bins or a numpy/matplotlib bin rule ("auto", "fd", "sturges", ...).
      Ignored when `discrete=True` and `round_to` is provided.
  round_to : int or None, default None
      If provided, values are rounded to this many decimals before plotting.
      When `discrete=True`, this also sets the bucket width = 10^(-round_to).
  discrete : bool, default False
      If True, uses fixed-width buckets aligned to the rounding step.
      (Great for 2-decimal ratings like 1.23, 1.24, ...).
  density : bool, default False
      If True, normalize to form a probability density (area = 1).
  figsize : (float, float), default (8, 5)
      Figure size in inches.
  title : str or None
      Title text; pass None to omit.
  show : bool, default True
      If True, calls plt.show().
  save_path : str or None
      If provided, saves the figure to this path.

  Returns
  -------
  (fig, ax)
      Matplotlib figure and axes.
  """
  if not ratings:
    raise ValueError("`ratings` is empty.")

  # Extract numeric values and drop NaNs
  vals = np.array([float(v) for v in ratings.values()], dtype=float)
  vals = vals[~np.isnan(vals)]
  if vals.size == 0:
    raise ValueError("No valid numeric ratings found.")

  # Optional rounding pre-processing
  if round_to is not None:
    vals = np.round(vals, round_to)

  fig, ax = plt.subplots(figsize=figsize)

  if discrete:
    # Use buckets aligned to the rounding step (default to 2 decimals if not provided)
    if round_to is None:
        round_to = 2  # sensible default for concreteness use-cases
        vals = np.round(vals, round_to)

    step = 10 ** (-round_to)
    # Build edges centered on each step (so a value like 3.45 falls in [3.445, 3.455))
    vmin, vmax = float(vals.min()), float(vals.max())
    # pad by one step on either side so the edge bars are fully visible
    left = vmin - 0.5 * step
    right = vmax + 1.5 * step
    edges = np.arange(left, right, step)

    ax.hist(vals, bins=edges, density=density)
  else:
    ax.hist(vals, bins=bins, density=density)

  ax.set_xlabel("Concreteness rating")
  ax.set_ylabel("Frequency" if not density else "Density")
  if title:
    ax.set_title(title)
  ax.grid(axis="y", linestyle="--", alpha=0.4)

  if save_path:
    fig.savefig(os.fspath(save_path), bbox_inches="tight")

  if show:
    plt.show()

  return fig, ax

if __name__ == "__main__":
  word_to_rating = load_concreteness_word_ratings()
  for i, (key, value) in enumerate(word_to_rating.items()):
    if i < 10:
      print(f"{key}: {value}")
    else:
      break
  # Uncomment to save in plots/
  # plot_rating_distribution(word_to_rating, bins=30, show=False, save_path=Path(script_dir, "plots/concreteness_dist.png"))
  plot_rating_distribution(word_to_rating, bins=30)