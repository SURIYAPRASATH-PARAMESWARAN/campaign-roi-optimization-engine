# src/bank_roi/data/loader.py
"""Data loading and feature engineering for the Bank Marketing dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from bank_roi.config import cfg

logger = logging.getLogger(__name__)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV and return it as a DataFrame.

    Parameters
    ----------
    path:
        Override the data path from config. Useful in tests/notebooks.
    """
    data_path = Path(path) if path else Path(cfg["data"]["raw_path"])
    logger.info("Loading data from %s", data_path)

    df = pd.read_csv(data_path, sep=cfg["data"]["sep"])
    logger.info("Loaded %d rows × %d cols", *df.shape)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering steps documented in the research notebook.

    Steps
    -----
    1. Encode binary target ``y`` as 0/1.
    2. Drop ``duration`` — it leaks call outcome at prediction time.
    3. Create ``previous_contacted`` flag from ``pdays`` sentinel (999 = never).
    4. Drop ``pdays`` after flag creation.

    Returns a *copy* — original DataFrame is not mutated.
    """
    df = df.copy()
    target = cfg["data"]["target_col"]
    sentinel = cfg["data"]["pdays_sentinel"]
    drop_cols = cfg["data"]["drop_cols"]

    # 1. Encode target
    df[target] = df[target].map({"no": 0, "yes": 1})

    # 2. Drop leaky / config-specified columns
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. Previous-contact flag
    if "pdays" in df.columns:
        df["previous_contacted"] = (df["pdays"] != sentinel).astype(int)
        df = df.drop(columns=["pdays"])

    logger.info("After feature engineering: %d rows × %d cols", *df.shape)
    return df


def split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    target = cfg["data"]["target_col"]
    split_cfg = cfg["split"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg["stratify"] else None,
    )

    logger.info(
        "Split: train=%d (pos=%.1f%%)  test=%d (pos=%.1f%%)",
        len(X_train),
        100 * y_train.mean(),
        len(X_test),
        100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test