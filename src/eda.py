"""Exploratory Data Analysis for the Heart Disease dataset (Assignment Task 1).

Generates a set of professional visualizations and a text summary:
  * missing-value summary
  * target class balance
  * feature histograms (numeric)
  * categorical feature distributions by target
  * correlation heatmap

Plots are saved to report/screenshots/ so they can be embedded in the report
and used as submission evidence. The same functions are imported by the EDA
notebook (notebooks/01_eda.ipynb).
"""
from __future__ import annotations

import logging

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend so this runs headless in CI

from src import config
from src.data_loader import load_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {0: "#4C9F70", 1: "#D1495B"}  # green = no disease, red = disease


def _save(fig: plt.Figure, name: str) -> None:
    config.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.SCREENSHOTS_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot -> %s", path)


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": miss, "missing_pct": pct})
    summary = summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    return summary


def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["No Disease (0)", "Disease (1)"],
        counts.values,
        color=[PALETTE[0], PALETTE[1]],
    )
    ax.bar_label(bars, padding=3)
    ax.set_title("Target Class Balance")
    ax.set_ylabel("Patient count")
    _save(fig, "class_balance.png")


def plot_numeric_histograms(df: pd.DataFrame) -> None:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(
            data=df, x=col, hue=config.TARGET, kde=True, palette=PALETTE,
            ax=axes[i], element="step", common_norm=False,
        )
        axes[i].set_title(f"Distribution of {col}")
    # hide any unused subplot
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numeric Feature Distributions by Target", fontsize=18)
    fig.tight_layout()
    _save(fig, "numeric_histograms.png")


def plot_categorical_distributions(df: pd.DataFrame) -> None:
    cols = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.countplot(data=df, x=col, hue=config.TARGET, palette=PALETTE, ax=axes[i])
        axes[i].set_title(f"{col} vs target")
        if axes[i].get_legend():
            axes[i].get_legend().remove()
    fig.suptitle("Categorical Feature Distributions by Target", fontsize=18)
    fig.tight_layout()
    _save(fig, "categorical_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "correlation_heatmap.png")


def run_eda() -> dict:
    """Run the full EDA suite and return a small summary dict."""
    df = load_processed()
    logger.info("Loaded processed dataset: %d rows x %d cols", *df.shape)

    miss = missing_value_summary(df)
    logger.info("Missing-value summary:\n%s", miss if not miss.empty else "  (none)")

    plot_class_balance(df)
    plot_numeric_histograms(df)
    plot_categorical_distributions(df)
    plot_correlation_heatmap(df)

    summary = {
        "n_rows": len(df),
        "n_features": df.shape[1] - 1,
        "class_balance": df[config.TARGET].value_counts().sort_index().to_dict(),
        "missing_cells": int(df.isna().sum().sum()),
    }
    logger.info("EDA summary: %s", summary)
    return summary


if __name__ == "__main__":
    run_eda()
