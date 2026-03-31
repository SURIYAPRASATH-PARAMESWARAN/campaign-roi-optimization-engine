# src/bank_roi/models/factory.py
from __future__ import annotations
import logging
from typing import Literal
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier
from bank_roi.config import cfg

logger = logging.getLogger(__name__)
ModelName = Literal["logistic_regression", "random_forest", "xgboost", "lightgbm"]

def _build_preprocessor(X: pd.DataFrame, use_ordinal: bool = False) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()
    logger.info("Numeric cols (%d): %s", len(num_cols), num_cols)
    logger.info("Categorical cols (%d): %s", len(cat_cols), cat_cols)
    cat_transformer = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if use_ordinal
        else OneHotEncoder(handle_unknown=cfg["preprocessing"]["handle_unknown"], sparse_output=False)
    )
    return ColumnTransformer(
        transformers=[("cat", cat_transformer, cat_cols), ("num", StandardScaler(), num_cols)],
        verbose_feature_names_out=False,
    )

def build_pipeline(model_name: ModelName, X: pd.DataFrame) -> Pipeline:
    use_ordinal = model_name == "lightgbm"
    preprocessor = _build_preprocessor(X, use_ordinal=use_ordinal)
    model_cfg = cfg["models"][model_name]
    if model_name == "logistic_regression":
        estimator = LogisticRegression(**model_cfg)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(**model_cfg)
    elif model_name == "xgboost":
        estimator = XGBClassifier(**model_cfg, verbosity=0)
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(**model_cfg)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")
    logger.info("Built pipeline: %s", model_name)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

def all_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:
    names: list[ModelName] = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    return {name: build_pipeline(name, X) for name in names}