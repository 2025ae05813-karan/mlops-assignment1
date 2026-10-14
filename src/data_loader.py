"""Data loading and cleaning for the UCI Heart Disease dataset.

Responsibilities (Assignment Task 1 — Data Acquisition & Cleaning):
  * Parse the raw, header-less `processed.*.data` files.
  * Replace the UCI '?' missing-value marker with NaN.
  * Coerce all columns to numeric.
  * Binarize the multi-class target (0 vs 1-4) into {0, 1}.
  * Write a clean, analysis-ready CSV to data/processed/heart.csv.

The functions are intentionally small and pure so they can be unit-tested
(Assignment Task 5).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_raw_file(path: Path) -> pd.DataFrame:
    """Load a single raw `processed.*.data` file into a typed DataFrame.

    Raw files are comma-separated, have no header, and use '?' for missing
    values. All 14 columns are numeric once the marker is handled.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. "
            "Run `python scripts/download_data.py` first."
        )

    df = pd.read_csv(path, header=None, names=config.COLUMNS, na_values=config.NA_MARKER)
    # Every column should be numeric; coerce any stragglers to NaN.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Loaded %d rows x %d cols from %s", len(df), df.shape[1], path.name)
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 0-4 severity target `num` into a binary `target`.

    UCI convention: 0 = no heart disease, 1-4 = presence of heart disease.
    """
    out = df.copy()
    out[config.TARGET] = (out[config.RAW_TARGET] > 0).astype(int)
    out = out.drop(columns=[config.RAW_TARGET])
    return out


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dataset-level cleaning that is safe to do before splitting.

    NOTE: value imputation and scaling are deliberately NOT done here — those
    are fit inside the sklearn Pipeline (src/preprocess.py) so that statistics
    are learned only from the training fold and we avoid data leakage.

    This step only:
      * drops exact duplicate rows,
      * binarizes the target.
    """
    out = df.drop_duplicates().reset_index(drop=True)
    out = binarize_target(out)
    return out


def build_dataset(sites: list[str] | None = None) -> pd.DataFrame:
    """Build the cleaned dataset.

    Parameters
    ----------
    sites : optional list of {"cleveland", "hungarian", "switzerland", "va"}
        Which site files to include. Defaults to Cleveland only (the canonical
        303-row subset used by nearly all published work on this dataset).
    """
    site_paths = {
        "cleveland": config.RAW_CLEVELAND,
        "hungarian": config.RAW_HUNGARIAN,
        "switzerland": config.RAW_SWITZERLAND,
        "va": config.RAW_VA,
    }
    sites = sites or ["cleveland"]

    frames = [load_raw_file(site_paths[s]) for s in sites]
    raw = pd.concat(frames, ignore_index=True)
    logger.info("Combined raw dataset: %d rows from sites=%s", len(raw), sites)

    cleaned = clean_data(raw)
    logger.info(
        "Cleaned dataset: %d rows, %d missing cells, class balance=%s",
        len(cleaned),
        int(cleaned.isna().sum().sum()),
        cleaned[config.TARGET].value_counts().to_dict(),
    )
    return cleaned


def save_processed(df: pd.DataFrame, path: Path = config.PROCESSED_CSV) -> Path:
    """Persist the cleaned dataset to CSV, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote cleaned dataset -> %s", path)
    return path


def load_processed(path: Path = config.PROCESSED_CSV) -> pd.DataFrame:
    """Load the cleaned CSV produced by `save_processed`."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run `python -m src.data_loader` first."
        )
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned Heart Disease dataset.")
    parser.add_argument(
        "--sites",
        nargs="+",
        default=["cleveland"],
        choices=["cleveland", "hungarian", "switzerland", "va"],
        help="Which UCI site files to combine (default: cleveland).",
    )
    args = parser.parse_args()

    df = build_dataset(sites=args.sites)
    save_processed(df)


if __name__ == "__main__":
    main()
