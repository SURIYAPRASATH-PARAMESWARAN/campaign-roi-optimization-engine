# src/bank_roi/models/factory.py
"""Build sklearn Pipelines for each candidate model.

All hyperparameters are read from configs/config.yaml so nothing is
hardcoded in the source.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from bank_roi.config import cfg

logger = logging.getLogger(__name__)

ModelName = Literal["logistic_regression", "random_forest", "xgboost"]


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Return a ColumnTransformer fitted to the column dtypes of *X*."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    logger.info("Numeric cols (%d): %s", len(num_cols), num_cols)
    logger.info("Categorical cols (%d): %s", len(cat_cols), cat_cols)

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown=cfg["preprocessing"]["handle_unknown"], sparse_output=False),
                cat_cols,
            ),
            ("num", StandardScaler(), num_cols),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model_name: ModelName, X: pd.DataFrame) -> Pipeline:
    """Construct a preprocessing + model Pipeline.

    Parameters
    ----------
    model_name:
        One of ``logistic_regression``, ``random_forest``, ``xgboost``.
    X:
        Training features (used only to detect column dtypes).

    Returns
    -------
    An unfitted ``sklearn.pipeline.Pipeline``.
    """
    preprocessor = _build_preprocessor(X)
    model_cfg = cfg["models"][model_name]

    if model_name == "logistic_regression":
        estimator = LogisticRegression(**model_cfg)

    elif model_name == "random_forest":
        estimator = RandomForestClassifier(**model_cfg)

    elif model_name == "xgboost":
        estimator = XGBClassifier(
            **model_cfg,
            use_label_encoder=False,
            verbosity=0,
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    logger.info("Built pipeline: %s", model_name)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def all_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Return a dict of {name: unfitted_pipeline} for all configured models."""
    names: list[ModelName] = ["logistic_regression", "random_forest", "xgboost"]
    return {name: build_pipeline(name, X) for name in names}