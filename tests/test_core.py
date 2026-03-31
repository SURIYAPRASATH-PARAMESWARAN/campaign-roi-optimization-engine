# tests/test_data.py
"""Unit tests for data loading and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from bank_roi.data.loader import engineer_features


def _make_raw_df(n: int = 100) -> pd.DataFrame:
    """Minimal fake dataset with the columns the real data has."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, n),
            "job": rng.choice(["admin.", "blue-collar", "technician"], n),
            "marital": rng.choice(["married", "single"], n),
            "education": rng.choice(["basic.4y", "university.degree"], n),
            "duration": rng.integers(0, 600, n),   # should be dropped
            "pdays": rng.choice([999, 5, 12], n),   # 999 = never
            "previous": rng.integers(0, 5, n),
            "y": rng.choice(["yes", "no"], n),
        }
    )


class TestEngineerFeatures:
    def test_duration_dropped(self):
        df = engineer_features(_make_raw_df())
        assert "duration" not in df.columns

    def test_pdays_dropped_flag_created(self):
        df = engineer_features(_make_raw_df())
        assert "pdays" not in df.columns
        assert "previous_contacted" in df.columns

    def test_previous_contacted_binary(self):
        df = engineer_features(_make_raw_df())
        assert set(df["previous_contacted"].unique()).issubset({0, 1})

    def test_target_encoded(self):
        df = engineer_features(_make_raw_df())
        assert set(df["y"].unique()).issubset({0, 1})

    def test_no_mutation(self):
        raw = _make_raw_df()
        _ = engineer_features(raw)
        assert "duration" in raw.columns  # original untouched


# tests/test_profit.py
"""Unit tests for profit optimization logic."""

import numpy as np
import pandas as pd
import pytest

from bank_roi.optimization.profit import profit_at_capacity


def _make_scored(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    scored = pd.DataFrame(
        {
            "rank": range(1, n + 1),
            "p_subscribe": rng.uniform(0.01, 0.9, n),
            "expected_profit": rng.uniform(-1, 49, n),
        }
    )
    scored = scored.sort_values("expected_profit", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    scored["cum_cost"] = scored["rank"] * 1.0
    scored["cum_profit"] = scored["expected_profit"].cumsum()
    scored["cum_roi"] = scored["cum_profit"] / scored["cum_cost"]
    return scored


class TestProfitAtCapacity:
    def test_capacity_respected(self):
        scored = _make_scored(1000)
        result = profit_at_capacity(scored, 500)
        assert result["n_called"] == 500

    def test_roi_positive_for_top_customers(self):
        scored = _make_scored(1000)
        result = profit_at_capacity(scored, 100)
        # top customers should have positive expected profit sum
        assert result["total_expected_profit"] > 0

    def test_max_profit_gte_capacity_profit(self):
        scored = _make_scored(1000)
        result = profit_at_capacity(scored, 200)
        assert result["max_achievable_profit"] >= result["total_expected_profit"]

    def test_zero_capacity(self):
        scored = _make_scored(1000)
        result = profit_at_capacity(scored, 0)
        assert result["n_called"] == 0
        assert result["total_expected_profit"] == 0.0