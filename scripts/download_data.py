"""Reproducible download of the UCI Heart Disease dataset.

Fetches the official archive from the UCI Machine Learning Repository and
extracts it into data/raw/. If the archive is already present locally
(e.g. shipped with the assignment), it is reused instead of re-downloading.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Official UCI archive for the Heart Disease dataset (ID 45).
UCI_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
LOCAL_ZIP = RAW_DIR / "heart+disease.zip"
# Marker file that indicates a successful extraction.
MARKER = RAW_DIR / "processed.cleveland.data"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def download(url: str, dest: Path) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())
    logger.info("Download complete (%d bytes)", dest.stat().st_size)


def extract(zip_path: Path, dest_dir: Path) -> None:
    logger.info("Extracting %s -> %s", zip_path, dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    logger.info("Extraction complete")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if MARKER.exists():
        logger.info("Raw data already present (%s). Nothing to do.", MARKER.name)
        return 0

    if not LOCAL_ZIP.exists():
        try:
            download(UCI_URL, LOCAL_ZIP)
        except Exception as exc:  # noqa: BLE001 - surface a clear message to the user
            logger.error("Automatic download failed: %s", exc)
            logger.error(
                "Manually download the dataset from %s and place the zip at %s, "
                "then re-run this script.",
                UCI_URL,
                LOCAL_ZIP,
            )
            return 1

    extract(LOCAL_ZIP, RAW_DIR)

    if not MARKER.exists():
        logger.error("Extraction did not produce %s. Check the archive.", MARKER.name)
        return 1

    logger.info("Dataset ready in %s", RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
