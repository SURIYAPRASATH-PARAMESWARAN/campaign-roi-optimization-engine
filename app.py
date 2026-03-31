# app.py — Campaign ROI Optimization Dashboard
# Run: streamlit run app.py

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from bank_roi.config import cfg
from bank_roi.data import load_raw, engineer_features
from bank_roi.optimization import score_customers, profit_at_capacity

st.set_page_config(
    page_title="Campaign ROI Engine",
    page_icon="📊",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    model_dir = Path(cfg["artifacts"]["model_dir"])
    candidates = sorted(model_dir.glob("best_model_*.joblib"))
    if not candidates:
        st.error("No trained model found. Run `python src/bank_roi/train.py` first.")
        st.stop()
    return joblib.load(candidates[-1]), candidates[-1].stem.replace("best_model_", "")

@st.cache_data(show_spinner="Scoring customers …")
def get_scored(contact_cost: float, revenue: float) -> pd.DataFrame:
    pipeline, _ = load_model()
    df = load_raw()
    df = engineer_features(df)
    X = df.drop(columns=[cfg["data"]["target_col"]])
    return score_customers(pipeline, X, contact_cost=contact_cost, revenue=revenue), X, df

@st.cache_data(show_spinner="Loading SHAP importance …")
def load_shap_importance():
    path = Path("outputs/shap_feature_importance.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data(show_spinner="Loading CV results …")
def load_cv_results():
    path = Path(cfg["artifacts"]["cv_results"])
    if path.exists():
        return pd.read_csv(path)
    return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parameters")
    st.markdown("---")

    capacity = st.slider(
        "📞 Daily call capacity",
        min_value=500, max_value=41_000,
        value=cfg["business"]["default_capacity"], step=500,
    )
    st.markdown("#### 💰 Business Assumptions")
    contact_cost = st.number_input("Cost per call (£)", min_value=0.1, max_value=50.0,
                                    value=float(cfg["business"]["contact_cost"]), step=0.5)
    revenue = st.number_input("Revenue per conversion (£)", min_value=1.0, max_value=500.0,
                               value=float(cfg["business"]["revenue_per_conversion"]), step=5.0)
    st.markdown("---")
    _, model_name = load_model()
    st.caption(f"Model: **{model_name}** (best by 5-fold CV ROC-AUC)")

# ── Load data ─────────────────────────────────────────────────────────────────
scored, X_all, df_full = get_scored(contact_cost, revenue)
kpis = profit_at_capacity(scored, capacity)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🔬 Model Comparison", "🧠 SHAP Explainability", "📋 Top Customers"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1: Dashboard
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.title("📊 Campaign ROI Optimization Engine")
    st.markdown(
        "Profit-driven customer targeting. Ranks all customers by "
        "**Expected Profit = P(subscribe) × Revenue − Cost** and simulates capacity constraints."
    )
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📞 Capacity", f"{kpis['capacity']:,}")
    c2.metric("💷 Expected Profit", f"£{kpis['total_expected_profit']:,.0f}")
    c3.metric("💸 Total Cost", f"£{kpis['total_cost']:,.0f}")
    c4.metric("📈 ROI", f"{kpis['roi']:.1%}")
    c5.metric("🏆 Max Achievable", f"£{kpis['max_achievable_profit']:,.0f}")
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Cumulative Expected Profit Curve")
        curve = scored[["rank", "cum_profit"]].copy()
        cap_row = scored[scored["rank"] == min(capacity, len(scored))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve["rank"], y=curve["cum_profit"],
            mode="lines", name="Cumulative Profit",
            line=dict(color="#2563eb", width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        ))
        if not cap_row.empty:
            fig.add_vline(x=capacity, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Capacity: {capacity:,}", annotation_position="top right")
            fig.add_scatter(x=[capacity], y=[cap_row["cum_profit"].values[0]],
                            mode="markers", marker=dict(color="#dc2626", size=10),
                            name=f"£{cap_row['cum_profit'].values[0]:,.0f} at capacity")
        fig.update_layout(
            xaxis_title="Customers ranked by expected profit",
            yaxis_title="Cumulative Expected Profit (£)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380, margin=dict(t=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Profit by Job Segment")
        top_n = scored[scored["rank"] <= capacity].copy()
        if "job" in top_n.columns:
            seg = top_n.groupby("job")["expected_profit"].sum().sort_values(ascending=True)
            fig2 = px.bar(seg.reset_index(), x="expected_profit", y="job", orientation="h",
                          labels={"expected_profit": "Total Expected Profit (£)", "job": ""},
                          color="expected_profit", color_continuous_scale="Blues")
            fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                               height=380, margin=dict(t=20))
            st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2: Model Comparison
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.title("🔬 Model Comparison — 5-Fold CV Results")
    st.markdown("""
    All four models evaluated with **stratified 5-fold cross-validation**.
    Hyperparameters optimised with **Optuna TPE Bayesian search**.
    Primary metric: **PR-AUC** (more informative than ROC-AUC under class imbalance).
    """)

    cv_df = load_cv_results()
    if cv_df is not None:
        # Summary table
        summary = cv_df.groupby("model").agg(
            roc_auc_mean=("roc_auc_val", "mean"),
            roc_auc_std=("roc_auc_val", "std"),
            pr_auc_mean=("pr_auc_val", "mean"),
            pr_auc_std=("pr_auc_val", "std"),
        ).reset_index().sort_values("pr_auc_mean", ascending=False)
        summary["ROC-AUC"] = summary.apply(lambda r: f"{r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f}", axis=1)
        summary["PR-AUC"] = summary.apply(lambda r: f"{r.pr_auc_mean:.4f} ± {r.pr_auc_std:.4f}", axis=1)
        st.dataframe(summary[["model", "ROC-AUC", "PR-AUC"]], use_container_width=True, hide_index=True)

        # Box plots
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(cv_df, x="model", y="roc_auc_val", color="model",
                         title="ROC-AUC per Fold", labels={"roc_auc_val": "ROC-AUC"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(cv_df, x="model", y="pr_auc_val", color="model",
                         title="PR-AUC per Fold", labels={"pr_auc_val": "PR-AUC"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Optuna best params
        import json
        params_path = Path("outputs/best_params.json")
        if params_path.exists():
            st.subheader("Optuna Best Hyperparameters")
            with open(params_path) as f:
                best_params = json.load(f)
            for model, params in best_params.items():
                with st.expander(f"🔧 {model}"):
                    st.json(params)
    else:
        st.info("Run `python src/bank_roi/train.py` to generate CV results.")

# ════════════════════════════════════════════════════════════════════
# TAB 3: SHAP Explainability
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.title("🧠 SHAP Explainability — Why Did We Rank Each Customer Here?")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** assigns each feature a contribution value
    for every prediction. It answers the business question:
    *"Why did we recommend calling this customer and not that one?"*
    """)

    fi_df = load_shap_importance()
    if fi_df is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Global Feature Importance (mean |SHAP|)")
            st.caption("Higher = this feature drives subscription probability more")
            fig = px.bar(
                fi_df.head(15), x="mean_abs_shap", y="feature",
                orientation="h", color="mean_abs_shap",
                color_continuous_scale="Blues",
                labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("What each feature means")
            st.markdown("""
            | Feature | Business meaning |
            |---|---|
            | `euribor3m` | Macroeconomic climate — low rates = more subscriptions |
            | `nr.employed` | Labour market — fewer employed = worse economic climate |
            | `emp.var.rate` | Employment variation rate |
            | `poutcome` | Previous campaign result — huge signal |
            | `previous_contacted` | Was customer contacted before? |
            | `month` | Call seasonality |
            | `age` | Older customers more likely to subscribe |
            | `campaign` | Number of times contacted — diminishing returns |
            """)

        # SHAP plot image
        shap_img = Path("outputs/plots/shap_summary.png")
        if shap_img.exists():
            st.subheader("SHAP Beeswarm Plot")
            st.caption("Each dot = one customer. Color = feature value. X = impact on prediction.")
            st.image(str(shap_img), use_container_width=True)
    else:
        st.info("Run `python src/bank_roi/train.py` (without --skip-shap) to generate SHAP analysis.")

# ════════════════════════════════════════════════════════════════════
# TAB 4: Top Customers
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.title("📋 Top Customers by Expected Profit")
    st.caption(f"Showing top 50 from {capacity:,} selected customers")

    display_cols = ["rank", "p_subscribe", "expected_profit"]
    extra = [c for c in ["age", "job", "education", "marital"] if c in scored.columns]
    display_cols += extra

    top50 = scored[display_cols].head(50).copy()
    top50["p_subscribe"] = top50["p_subscribe"].map("{:.1%}".format)
    top50["expected_profit"] = top50["expected_profit"].map("£{:.2f}".format)

    st.dataframe(top50, use_container_width=True, hide_index=True)

    # Download
    csv = scored[display_cols].to_csv(index=False)
    st.download_button("⬇️ Download all scored customers", csv,
                       "scored_customers.csv", "text/csv")

st.markdown("---")
st.caption("Bank Marketing Dataset (UCI) · LR / RF / XGBoost / LightGBM · Optuna tuning · SHAP explainability")