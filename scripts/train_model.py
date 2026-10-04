"""
train_model.py
--------------
Trains XGBoost and LightGBM classifiers on the feature matrix.
Handles class imbalance (stockouts are rare events) via:
  - scale_pos_weight (XGBoost)
  - is_unbalance (LightGBM)
  - Threshold tuning on validation set

Evaluation:
  - Time-aware train/test split (no data leakage — future used to test past)
  - Reports: Accuracy, Precision, Recall, F1, AUC-ROC, Average Precision
  - Saves confusion matrix and feature importance plots
  - Saves the best model (by F1) to saved_models/

Input:
    data/processed/features_final.csv

Output:
    saved_models/best_model.pkl       (best model by F1)
    saved_models/xgb_model.pkl
    saved_models/lgbm_model.pkl
    saved_models/feature_names.txt
    saved_models/evaluation_report.txt
    saved_models/confusion_matrix.png
    saved_models/feature_importance.png
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROCESSED_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR     = os.path.join(os.path.dirname(__file__), "..", "saved_models")
FEATURES_PATH  = os.path.join(PROCESSED_DIR, "features_final.csv")

FEATURE_COLS = [
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_rolling_mean_7",  "sales_rolling_mean_14",  "sales_rolling_mean_28",
    "sales_rolling_std_7",   "sales_rolling_std_14",   "sales_rolling_std_28",
    "sales_rolling_max_7",   "sales_rolling_max_14",   "sales_rolling_max_28",
    "sales_velocity",
    "day_of_week", "day_of_month", "month",
    "is_weekend", "is_payday", "is_mid_month",
    "is_holiday", "is_snap",
    "days_to_festival", "is_near_festival",
    "stock_ratio", "stock_after", "reorder_point",
    "consecutive_low_stock",
]
TARGET = "stockout"


def time_aware_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split chronologically to prevent data leakage."""
    df = df.sort_values("date").reset_index(drop=True)
    n    = len(df)
    cutoff = int(n * (1 - test_frac))
    train = df.iloc[:cutoff]
    test  = df.iloc[cutoff:]
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"Train dates: {train['date'].min()} → {train['date'].max()}")
    print(f"Test  dates: {test['date'].min()}  → {test['date'].max()}")
    return train, test


def tune_threshold(model, X_val, y_val):
    """Find the probability threshold that maximises F1 on validation set."""
    probs = model.predict_proba(X_val)[:, 1]
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.3, 0.75, 0.05):
        preds = (probs >= thresh).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return round(best_thresh, 2), round(best_f1, 4)


def evaluate(name, model, X_test, y_test, threshold, report_lines):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, zero_division=0)
    rec   = recall_score(y_test, preds, zero_division=0)
    f1    = f1_score(y_test, preds, zero_division=0)
    auc   = roc_auc_score(y_test, probs)
    ap    = average_precision_score(y_test, probs)
    cm    = confusion_matrix(y_test, preds)

    lines = [
        f"\n{'='*55}",
        f"  {name}  (threshold={threshold})",
        f"{'='*55}",
        f"  Accuracy          : {acc:.4f}",
        f"  Precision         : {prec:.4f}",
        f"  Recall            : {rec:.4f}",
        f"  F1 Score          : {f1:.4f}",
        f"  AUC-ROC           : {auc:.4f}",
        f"  Average Precision : {ap:.4f}",
        f"\n  Confusion Matrix:\n{cm}",
        f"\n{classification_report(y_test, preds, zero_division=0)}",
    ]
    for l in lines:
        print(l)
    report_lines.extend(lines)

    return {"name": name, "f1": f1, "auc": auc, "threshold": threshold,
            "probs": probs, "preds": preds, "cm": cm}


def plot_confusion_matrix(results, y_test, out_dir):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Stockout","Stockout"],
                    yticklabels=["No Stockout","Stockout"], ax=ax)
        ax.set_title(f"{res['name']}\nF1={res['f1']:.3f} | AUC={res['auc']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {path}")


def plot_feature_importance(models_dict, feature_names, out_dir):
    fig, axes = plt.subplots(1, len(models_dict), figsize=(12, 8))
    if len(models_dict) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models_dict.items()):
        try:
            importances = model.feature_importances_
            idx = np.argsort(importances)[-20:]
            ax.barh(np.array(feature_names)[idx], importances[idx], color="steelblue")
            ax.set_title(f"{name} — Top 20 Features")
            ax.set_xlabel("Importance")
        except Exception as e:
            ax.set_title(f"{name} — importance unavailable ({e})")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved feature importance → {path}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(FEATURES_PATH):
        print("ERROR: features_final.csv not found. Run feature_engineering.py first.")
        return

    print("Loading features...")
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: Missing feature columns (skipping): {missing}")

    df = df.dropna(subset=available + [TARGET]).reset_index(drop=True)
    print(f"Feature matrix: {df.shape}  |  Stockout rate: {df[TARGET].mean():.2%}")

    train_df, test_df = time_aware_split(df)
    X_train = train_df[available].values
    y_train = train_df[TARGET].values
    X_test  = test_df[available].values
    y_test  = test_df[TARGET].values

    # Class imbalance ratio
    neg, pos   = np.bincount(y_train)
    scale_wt   = neg / max(pos, 1)
    print(f"Class ratio (neg:pos) = {neg}:{pos}  →  scale_pos_weight={scale_wt:.1f}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_wt,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False)
    xgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    xgb_thresh, _ = tune_threshold(xgb_model, X_val, y_val)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    print("Training LightGBM...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        is_unbalance=True,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    lgbm_model.fit(X_tr, y_tr,
                   eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(20, verbose=False),
                               lgb.log_evaluation(-1)])
    lgbm_thresh, _ = tune_threshold(lgbm_model, X_val, y_val)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    report_lines = ["STOCKOUT PREDICTOR — MODEL EVALUATION REPORT", "="*55]
    report_lines.append(f"Feature count : {len(available)}")
    report_lines.append(f"Train rows    : {len(X_train)}")
    report_lines.append(f"Test rows     : {len(X_test)}")
    report_lines.append(f"Stockout rate : {y_test.mean():.2%}")

    all_results = []
    all_results.append(evaluate("XGBoost",  xgb_model,  X_test, y_test, xgb_thresh,  report_lines))
    all_results.append(evaluate("LightGBM", lgbm_model, X_test, y_test, lgbm_thresh, report_lines))

    plot_confusion_matrix(all_results, y_test, MODELS_DIR)
    plot_feature_importance({"XGBoost": xgb_model, "LightGBM": lgbm_model},
                             available, MODELS_DIR)

    # ── Save models ──────────────────────────────────────────────────────────
    joblib.dump(xgb_model,  os.path.join(MODELS_DIR, "xgb_model.pkl"))
    joblib.dump(lgbm_model, os.path.join(MODELS_DIR, "lgbm_model.pkl"))

    # Save feature names
    with open(os.path.join(MODELS_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(available))

    # Save thresholds
    thresholds = {"xgb": xgb_thresh, "lgbm": lgbm_thresh}
    with open(os.path.join(MODELS_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f)

    # Choose best model by F1
    best = max(all_results, key=lambda r: r["f1"])
    best_model = xgb_model if best["name"] == "XGBoost" else lgbm_model
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model_meta.json"), "w") as f:
        json.dump({"name": best["name"], "f1": best["f1"],
                   "auc": best["auc"], "threshold": best["threshold"]}, f)

    print(f"\n✓ Best model: {best['name']}  F1={best['f1']:.4f}  AUC={best['auc']:.4f}")

    # Save text report
    report_path = os.path.join(MODELS_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"✓ Evaluation report → {report_path}")


if __name__ == "__main__":
    main()
