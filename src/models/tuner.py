# src/bank_roi/models/tuner.py
"""Optuna-powered hyperparameter tuning for all four models.


- Industry standard in 2026, replacing GridSearchCV/RandomizedSearchCV
- Bayesian optimisation (TPE sampler) — smarter than random search
- Native pruning — stops unpromising trials early (Hyperband pruner)
- Clean trial history stored as DataFrame for analysis

Usage
-----
    from bank_roi.models.tuner import tune_model
    best_params = tune_model("lightgbm", X_train, y_train)
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from bank_roi.config import cfg

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _get_preprocessor(X: pd.DataFrame, use_ordinal: bool = False) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()
    cat_enc = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if use_ordinal
        else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    return ColumnTransformer(
        [("cat", cat_enc, cat_cols), ("num", StandardScaler(), num_cols)],
        verbose_feature_names_out=False,
    )


def _objective(
    trial: optuna.Trial,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> float:
    """Optuna objective — returns mean PR-AUC across inner CV folds."""
    opt_cfg = cfg["optuna"]

    # ── Search spaces ─────────────────────────────────────────────────────────
    if model_name == "logistic_regression":
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": 5000,
            "class_weight": "balanced",
        }
        estimator = LogisticRegression(**params)
        use_ordinal = False

    elif model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        estimator = RandomForestClassifier(**params)
        use_ordinal = False

    elif model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5.0, 15.0),
            "verbosity": 0,
            "random_state": 42,
            "n_jobs": -1,
        }
        estimator = XGBClassifier(**params)
        use_ordinal = False

    elif model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        estimator = LGBMClassifier(**params)
        use_ordinal = True

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    # ── Inner CV ──────────────────────────────────────────────────────────────
    preprocessor = _get_preprocessor(X, use_ordinal=use_ordinal)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    cv = StratifiedKFold(
        n_splits=opt_cfg["cv_splits"],
        shuffle=True,
        random_state=opt_cfg["random_state"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            scoring="average_precision",
            n_jobs=1,  # parallelism handled by Optuna
        )

    return float(np.mean(scores))


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int | None = None,
    timeout: int | None = None,
) -> dict:
    """Run Optuna hyperparameter search for a single model.

    Parameters
    ----------
    model_name:
        One of ``logistic_regression``, ``random_forest``, ``xgboost``,
        ``lightgbm``.
    X, y:
        Training data.
    n_trials, timeout:
        Override config defaults.

    Returns
    -------
    dict with keys: best_params, best_value, study
    """
    opt_cfg = cfg["optuna"]
    n_trials = n_trials or opt_cfg["n_trials"]
    timeout = timeout or opt_cfg["timeout"]

    logger.info("Tuning %s — %d trials, %ds timeout", model_name, n_trials, timeout)

    sampler = optuna.samplers.TPESampler(seed=opt_cfg["random_state"])
    pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(
        direction=opt_cfg["direction"],
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_name}_tuning",
    )

    study.optimize(
        lambda trial: _objective(trial, model_name, X, y),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=1,
    )

    logger.info(
        "%s best PR-AUC: %.4f | params: %s",
        model_name,
        study.best_value,
        study.best_params,
    )

    return {
        "model_name": model_name,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
        "trials_df": study.trials_dataframe(),
    }


def tune_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[str] | None = None,
) -> dict[str, dict]:
    """Tune all models and return results keyed by model name.

    Parameters
    ----------
    models:
        Subset of models to tune. Defaults to all four.
    """
    if models is None:
        models = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

    results = {}
    for name in models:
        results[name] = tune_model(name, X, y)

    # Summary table
    summary = pd.DataFrame([
        {
            "model": r["model_name"],
            "best_pr_auc": round(r["best_value"], 4),
            "best_params": r["best_params"],
        }
        for r in results.values()
    ]).sort_values("best_pr_auc", ascending=False)

    logger.info("\n=== Optuna Tuning Summary ===\n%s", summary[["model", "best_pr_auc"]].to_string(index=False))

    return results
