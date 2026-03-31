# src/bank_roi/evaluation/metrics.py
"""Model evaluation: cross-validation, hold-out metrics, and comparison tables."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    brier_score_loss,
)

from bank_roi.config import cfg

logger = logging.getLogger(__name__)


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "model",
) -> pd.DataFrame:
    """Run stratified k-fold CV and return per-fold metric DataFrame.

    Metrics evaluated
    -----------------
    - ROC-AUC
    - PR-AUC (average precision)
    - Fit time (seconds)

    Parameters
    ----------
    pipeline:
        Unfitted sklearn Pipeline.
    X, y:
        Full training data — CV will split internally.
    model_name:
        Label used in the returned DataFrame.
    """
    cv_cfg = cfg["cross_validation"]

    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg["shuffle"],
        random_state=cv_cfg["random_state"],
    )

    logger.info("Running %d-fold CV for %s …", cv_cfg["n_splits"], model_name)

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["roc_auc", "average_precision"],
        return_train_score=True,
        n_jobs=-1,
    )

    df = pd.DataFrame(
        {
            "model": model_name,
            "fold": range(1, cv_cfg["n_splits"] + 1),
            "roc_auc_train": results["train_roc_auc"],
            "roc_auc_val": results["test_roc_auc"],
            "pr_auc_train": results["train_average_precision"],
            "pr_auc_val": results["test_average_precision"],
            "fit_time_s": results["fit_time"],
        }
    )
    logger.info(
        "%s — val ROC-AUC %.4f ± %.4f | val PR-AUC %.4f ± %.4f",
        model_name,
        df["roc_auc_val"].mean(),
        df["roc_auc_val"].std(),
        df["pr_auc_val"].mean(),
        df["pr_auc_val"].std(),
    )
    return df


def compare_models_cv(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Run CV for all pipelines and return a stacked results DataFrame."""
    frames = [
        cross_validate_pipeline(pipe, X, y, name)
        for name, pipe in pipelines.items()
    ]
    return pd.concat(frames, ignore_index=True)


def holdout_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate a *fitted* pipeline on the hold-out test set.

    Returns
    -------
    dict with keys: roc_auc, pr_auc, brier_score, classification_report
    """
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }
    return metrics


def summary_table(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CV results into a clean comparison table (mean ± std)."""
    agg = (
        cv_results.groupby("model")
        .agg(
            roc_auc_mean=("roc_auc_val", "mean"),
            roc_auc_std=("roc_auc_val", "std"),
            pr_auc_mean=("pr_auc_val", "mean"),
            pr_auc_std=("pr_auc_val", "std"),
            fit_time_mean=("fit_time_s", "mean"),
        )
        .reset_index()
        .sort_values("roc_auc_mean", ascending=False)
    )
    agg["roc_auc"] = agg.apply(
        lambda r: f"{r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f}", axis=1
    )
    agg["pr_auc"] = agg.apply(
        lambda r: f"{r.pr_auc_mean:.4f} ± {r.pr_auc_std:.4f}", axis=1
    )
    return agg[["model", "roc_auc", "pr_auc", "fit_time_mean"]].rename(
        columns={"fit_time_mean": "avg_fit_time_s"}
    )