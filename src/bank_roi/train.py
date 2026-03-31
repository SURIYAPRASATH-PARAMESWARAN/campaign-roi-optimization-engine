#!/usr/bin/env python3
# src/bank_roi/train.py
from __future__ import annotations
import argparse, logging, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from bank_roi.config import cfg
from bank_roi.data import load_raw, engineer_features, split
from bank_roi.models import build_pipeline, all_pipelines
from bank_roi.evaluation import compare_models_cv, holdout_metrics, summary_table
from bank_roi.optimization import score_customers, profit_curve
from bank_roi.optimization.profit import profit_at_capacity
from bank_roi.explainability import explain_model, shap_summary, feature_importance_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--skip-tuning", action="store_true")
    p.add_argument("--skip-shap", action="store_true")
    return p.parse_args()

def threshold_analysis(scored, y_all):
    base_rate = float(y_all.mean())
    total_positives = int(y_all.sum())
    rows = []
    for k in [500, 1000, 2000, 5000, 10000, 20000]:
        if k > len(scored):
            continue
        top_k = scored.head(k)
        tp = int(top_k["actual_y"].sum())
        precision = tp / k
        recall = tp / total_positives
        lift = precision / base_rate
        rows.append({"top_k": k, "true_positives": tp, "precision": round(precision, 4), "recall": round(recall, 4), "lift_vs_random": round(lift, 2)})
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    out_dir = Path("outputs")
    model_dir = Path(cfg["artifacts"]["model_dir"])
    plots_dir = out_dir / "plots"
    for d in [out_dir, model_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load & reset index cleanly
    df = load_raw(args.data)
    df = engineer_features(df)
    df = df.reset_index(drop=True)
    X_train, X_test, y_train, y_test = split(df)

    # 2. Optuna
    tuned_params = {}
    if not args.skip_tuning:
        from bank_roi.models.tuner import tune_all_models
        logger.info("=== Optuna Hyperparameter Tuning ===")
        tuning_results = tune_all_models(X_train, y_train)
        for name, result in tuning_results.items():
            tuned_params[name] = result["best_params"]
            result["trials_df"].to_csv(out_dir / f"optuna_trials_{name}.csv", index=False)
        with open(out_dir / "best_params.json", "w") as f:
            json.dump(tuned_params, f, indent=2)
    else:
        logger.info("Skipping Optuna — using config defaults")
        params_path = out_dir / "best_params.json"
        if params_path.exists():
            with open(params_path) as f:
                tuned_params = json.load(f)
            logger.info("Loaded previous tuning results")

    # 3. Build all 4 pipelines
    logger.info("Building all 4 pipelines: LR, RF, XGBoost, LightGBM")
    pipelines = all_pipelines(X_train)

    # 4. CV all 4
    logger.info("=== 5-Fold Cross-Validation (all 4 models) ===")
    cv_results = compare_models_cv(pipelines, X_train, y_train)
    cv_results.to_csv(cfg["artifacts"]["cv_results"], index=False)
    print("\n=== Model Comparison (5-fold CV) ===")
    print(summary_table(cv_results).to_string(index=False))

    # 5. Best model
    best_name = cv_results.groupby("model")["pr_auc_val"].mean().idxmax()
    logger.info("Best model by PR-AUC: %s", best_name)
    best_pipeline = pipelines[best_name]
    best_pipeline.fit(X_train, y_train)

    # 6. Hold-out
    metrics = holdout_metrics(best_pipeline, X_test, y_test)
    print(f"\n=== {best_name} — Hold-out Metrics ===")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC      : {metrics['pr_auc']:.4f}")
    print(f"  Brier Score : {metrics['brier_score']:.4f}")

    # 7. Precision@K — using preserved original index from score_customers
    target = cfg["data"]["target_col"]
    X_all = df.drop(columns=[target])
    y_all = df[target]

    scored = score_customers(best_pipeline, X_all)
    # score_customers now does reset_index(drop=False) so original row numbers
    # are stored in the 'index' column — use that to look up actual labels
    scored["actual_y"] = y_all.iloc[scored["index"]].values

    threshold_df = threshold_analysis(scored, y_all)
    threshold_df.to_csv(out_dir / "threshold_analysis.csv", index=False)
    lift_1k = threshold_df[threshold_df["top_k"] == 1000]["lift_vs_random"].values[0]
    print(f"\n=== Precision@K — Operational Targeting Performance ===")
    print(threshold_df.to_string(index=False))
    print(f"\n  Base rate (random calling) : {y_all.mean():.1%}")
    print(f"  Lift at top 1,000 customers: {lift_1k:.1f}x better than random")

    # 8. SHAP
    if not args.skip_shap:
        logger.info("=== SHAP Explainability ===")
        shap_result = explain_model(best_pipeline, X_train, max_samples=1500)
        fi_df = feature_importance_df(shap_result, top_n=20)
        fi_df.to_csv(out_dir / "shap_feature_importance.csv", index=False)
        print("\n=== Top 10 Features by SHAP Importance ===")
        print(fi_df.head(10).to_string(index=False))
        shap_summary(shap_result, top_n=20, save_path=plots_dir / "shap_summary.png")
    else:
        logger.info("Skipping SHAP (--skip-shap)")

    # 9. Save everything
    model_path = model_dir / f"best_model_{best_name}.joblib"
    joblib.dump(best_pipeline, model_path)
    scored.to_csv(cfg["artifacts"]["scored_customers"], index=False)
    profit_curve(scored).to_csv(cfg["artifacts"]["profit_curve"], index=False)
    kpi_df = pd.DataFrame([profit_at_capacity(scored, c) for c in [1000, 2500, 5000, 10000, 20000]])
    kpi_df.to_csv(out_dir / "capacity_sensitivity.csv", index=False)
    print("\n=== Capacity Sensitivity ===")
    print(kpi_df[["capacity", "total_expected_profit", "roi"]].to_string(index=False))
    logger.info("=== Done ===")

if __name__ == "__main__":
    main()