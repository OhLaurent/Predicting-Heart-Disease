"""Visualisation helpers.

Centralises reusable plotting logic so notebooks stay concise.

Note: this module is a work in progress — helpers will be extracted
from the notebooks as the project matures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_disease.config import FIGURES_DIR, TARGET_COLUMN


def save_figure(name: str, dpi: int = 150) -> Path:
    """Save the current matplotlib figure to ``reports/figures/``.

    Parameters
    ----------
    name : str
        Filename without extension (PNG is used).
    dpi : int, optional
        Resolution. Default 150.

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Bar chart of the ``Heart Disease`` class balance."""
    counts = df[TARGET_COLUMN].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Heart Disease — class distribution")
    ax.set_xlabel(TARGET_COLUMN)
    ax.set_ylabel("Count")
    plt.tight_layout()


def plot_cv_score_distributions(
    scores: dict[str, "np.ndarray"],  # noqa: F821
) -> None:
    """Overlapping KDE plot of cross-validation AUC-ROC scores per model.

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        Mapping of model name → array of per-fold AUC-ROC scores.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, cv_scores in scores.items():
        sns.kdeplot(cv_scores, label=model_name, fill=True, ax=ax)
    ax.set_title("Distribution of AUC-ROC Scores for Candidate Models")
    ax.set_xlabel("AUC-ROC Score")
    ax.legend()
    plt.tight_layout()
