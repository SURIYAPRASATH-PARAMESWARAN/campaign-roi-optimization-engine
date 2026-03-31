# src/bank_roi/explainability/shap_analysis.py
"""SHAP-based model explainability.

Why SHAP?
---------
SHAP (SHapley Additive exPlanations) is the 2026 standard for ML explainability.
It gives each feature a consistent, theoretically grounded contribution value
for every individual prediction — not just global feature importance.

For a business decision tool like this one, SHAP answers:
  "Why did we rank customer #4821 at position 12 instead of 500?"

Key outputs
-----------
- Global summary plot  — which features drive conversions most
- Waterfall plot        — per-customer explanation (why THIS person ranked here)
- Feature importance DF — clean table for the README / dashboard
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _get_explainer(pipeline: Pipeline, X_sample: pd.DataFrame) -> tuple:
    """Return the right SHAP explainer for the model type.

    Tree-based models (RF, XGBoost, LightGBM) use TreeExplainer — fast and exact.
    Linear models use LinearExplainer.
    Fallback: KernelExplainer (slow but universal).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    model_type = type(model).__name__.lower()

    # Transform features once for the explainer
    X_transformed = preprocessor.transform(X_sample)

    # Get feature names after preprocessing
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    if any(name in model_type for name in ["forest", "xgb", "lgbm", "gradient", "tree"]):
        logger.info("Using TreeExplainer for %s", type(model).__name__)
        explainer = shap.TreeExplainer(model)
    elif "logistic" in model_type or "linear" in model_type:
        logger.info("Using LinearExplainer for %s", type(model).__name__)
        explainer = shap.LinearExplainer(model, X_df)
    else:
        logger.info("Falling back to KernelExplainer for %s", type(model).__name__)
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_df, 100))

    return explainer, X_df, feature_names


def explain_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_samples: int = 2000,
) -> dict:
    """Compute SHAP values for a fitted pipeline.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline.
    X:
        Feature DataFrame (raw, before preprocessing).
    max_samples:
        Cap samples for speed — SHAP is O(n·features).

    Returns
    -------
    dict with keys: shap_values, X_transformed, feature_names, explainer
    """
    sample = X.sample(min(max_samples, len(X)), random_state=42)
    logger.info("Computing SHAP values for %d samples …", len(sample))

    explainer, X_df, feature_names = _get_explainer(pipeline, sample)

    shap_values = explainer.shap_values(X_df)

    # For binary classifiers, shap_values is a list [class0, class1]
    # We want class 1 (positive = subscribe)
    import numpy as np
    sv = np.array(shap_values)
    if sv.ndim == 3:
        shap_values = sv[:, :, 1]
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    logger.info("SHAP values computed: shape=%s", np.array(shap_values).shape)

    return {
        "shap_values": shap_values,
        "X_transformed": X_df,
        "feature_names": feature_names,
        "explainer": explainer,
    }


def feature_importance_df(shap_result: dict, top_n: int = 20) -> pd.DataFrame:
    """Return mean absolute SHAP values as a ranked DataFrame.

    This is the global feature importance — averaged across all samples.
    """
    shap_vals = np.array(shap_result["shap_values"])
    names = shap_result["feature_names"]

    importance = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    importance["rank"] = importance.index + 1
    return importance[["rank", "feature", "mean_abs_shap"]]


def shap_summary(
    shap_result: dict,
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> None:
    """Plot SHAP beeswarm summary — shows direction AND magnitude of each feature.

    The beeswarm is more informative than a bar chart:
    - Each dot = one customer
    - Color = feature value (red = high, blue = low)
    - X position = SHAP value (right = pushes toward subscription)
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_result["shap_values"],
        shap_result["X_transformed"],
        feature_names=shap_result["feature_names"],
        max_display=top_n,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Impact — What Drives Subscription Probability?", pad=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved SHAP summary → %s", save_path)
    plt.show()


def shap_waterfall(
    pipeline: Pipeline,
    X: pd.DataFrame,
    customer_idx: int = 0,
    save_path: str | Path | None = None,
) -> None:
    """Waterfall plot for a single customer — the 'why this person?' explanation.

    Parameters
    ----------
    customer_idx:
        Row index in X to explain.
    """
    single = X.iloc[[customer_idx]]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(single)
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    model_type = type(model).__name__.lower()
    if any(name in model_type for name in ["forest", "xgb", "lgbm", "gradient"]):
        explainer = shap.TreeExplainer(model)
        explanation = explainer(X_df)
        # For binary classifiers get class 1
        if len(explanation.shape) == 3:
            explanation = explanation[:, :, 1]
    else:
        explainer = shap.LinearExplainer(model, X_df)
        explanation = explainer(X_df)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[0], show=False)
    plt.title(f"Customer #{customer_idx} — Why This Ranking?", pad=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved waterfall → %s", save_path)
    plt.show()