# src/bank_roi/optimization/profit.py
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from bank_roi.config import cfg

logger = logging.getLogger(__name__)


def score_customers(pipeline, X, contact_cost=None, revenue=None):
    biz = cfg["business"]
    cost = contact_cost if contact_cost is not None else biz["contact_cost"]
    rev = revenue if revenue is not None else biz["revenue_per_conversion"]

    logger.info("Scoring %d customers (cost=%.2f, revenue=%.2f)", len(X), cost, rev)

    scored = X.copy()
    scored["p_subscribe"] = pipeline.predict_proba(X)[:, 1]
    scored["expected_profit"] = scored["p_subscribe"] * rev - cost

    # CRITICAL: reset_index(drop=False) preserves original row number in 'index' column
    # This lets train.py correctly align y_all labels after sorting
    scored = (
        scored.sort_values("expected_profit", ascending=False)
        .reset_index(drop=False)   # <-- drop=False keeps original index as column
    )
    scored["rank"] = range(1, len(scored) + 1)
    scored["cum_cost"] = scored["rank"] * cost
    scored["cum_profit"] = scored["expected_profit"].cumsum()
    scored["cum_roi"] = (scored["cum_profit"] / scored["cum_cost"]).replace(
        [np.inf, -np.inf], np.nan
    )

    logger.info(
        "Top-1000 expected profit: £%.2f | Max total: £%.2f",
        scored.loc[scored["rank"] <= 1000, "cum_profit"].iloc[-1],
        scored["cum_profit"].max(),
    )
    return scored


def profit_at_capacity(scored, capacity):
    biz = cfg["business"]
    cost = biz["contact_cost"]
    subset = scored[scored["rank"] <= capacity]
    total_profit = subset["expected_profit"].sum()
    total_cost = len(subset) * cost
    roi = total_profit / total_cost if total_cost > 0 else 0.0
    return {
        "capacity": capacity,
        "n_called": len(subset),
        "total_expected_profit": round(total_profit, 2),
        "total_cost": round(total_cost, 2),
        "roi": round(roi, 4),
        "max_achievable_profit": round(scored["cum_profit"].max(), 2),
    }


def profit_curve(scored):
    return scored[["rank", "cum_cost", "cum_profit", "cum_roi"]].copy()