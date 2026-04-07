import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_absolute_error, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # Prevents popup windows
import matplotlib.pyplot as plt

print("Loading feature data...")
df = pd.read_csv("data/processed/features_final.csv", parse_dates=["date"])
df = df.sort_values(["date", "sku_id"]).reset_index(drop=True)

print(f"Loaded {len(df):,} rows")
print(f"Stockout rate: {df['stockout_label'].mean():.1%}")

# ─────────────────────────────────────────────
# STEP A: Define features and targets
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "current_stock", "available_stock", "reserved_in_carts",
    "cart_reservation_ratio", "stock_pct_of_baseline",
    "sales_last_5min", "sales_last_30min", "sales_last_1hr",
    "velocity_per_day", "time_to_zero_naive", "velocity_acceleration",
    "day_of_week", "is_weekend", "is_month_start", "quarter",
    "historical_avg_dow", "historical_avg_month",
    "demand_vs_dow_avg", "demand_ratio_vs_avg",
    "stock_change_3d", "stock_change_7d", "rolling_avg_sales_7d",
    "has_event", "is_snap_day", "cat_encoded", "event_encoded",
]

X      = df[FEATURE_COLS].values
y_cls  = df["stockout_label"].values        # Classification target (0 or 1)
y_reg  = df["days_to_zero"].values          # Regression target (days until zero)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Positive class (stockouts): {y_cls.sum():,} / {len(y_cls):,}")

# ─────────────────────────────────────────────
# STEP B: Time-series train/test split
# ─────────────────────────────────────────────
# CRITICAL CONCEPT: Never use random splits for time-series data.
#
# Imagine training on data from March 2013 and testing on January 2012.
# The model would be evaluated on the past, which it "already saw" —
# that's cheating and produces fake accuracy numbers.
#
# TimeSeriesSplit always ensures:
#   - Training data comes BEFORE test data in time
#   - The model is always predicting the FUTURE, never the past
#
# We use 5 splits. The last split gives us the most data to train on,
# so we use that split's indices for our final evaluation.

print("\nSetting up time-series cross validation...")
tscv = TimeSeriesSplit(n_splits=5)

splits = list(tscv.split(X))
train_idx, test_idx = splits[-1]  # Use the last (largest) split

X_train, X_test = X[train_idx], X[test_idx]
y_cls_train, y_cls_test = y_cls[train_idx], y_cls[test_idx]
y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]

print(f"Training rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")
print(f"Train stockout rate: {y_cls_train.mean():.1%}")
print(f"Test stockout rate:  {y_cls_test.mean():.1%}")

# ─────────────────────────────────────────────
# STEP C: Train XGBoost Classifier
# ─────────────────────────────────────────────
# XGBoost for the classification task:
# "Will this SKU stock out within the next 7 days? YES or NO (+ probability)"
#
# Key parameters explained:
#
# n_estimators=500     — how many trees to build. More = slower but more accurate.
#                        early_stopping_rounds stops us before all 500 if accuracy plateaus.
#
# max_depth=6          — how deep each tree can go. Deeper = learns more complex patterns
#                        but risks overfitting (memorising training data instead of learning).
#
# learning_rate=0.05   — how much each new tree adjusts the model. Lower = more careful,
#                        needs more trees but generalises better.
#
# scale_pos_weight=5   — because only 15% of rows are stockouts, we tell the model to treat
#                        each stockout as 5x more important. Without this, the model ignores
#                        the minority class. Rule of thumb: (non-stockout count / stockout count)
#
# subsample=0.8        — each tree only sees 80% of the training data, randomly chosen.
#                        Prevents overfitting, makes trees more independent of each other.
#
# colsample_bytree=0.8 — each tree only sees 80% of features. Same benefit as subsample.

neg   = (y_cls_train == 0).sum()
pos   = (y_cls_train == 1).sum()
scale = round(neg / pos)
print(f"\nClass weight scale_pos_weight: {scale} (neg={neg:,} / pos={pos:,})")

print("\nTraining XGBoost classifier...")
clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="auc",
    early_stopping_rounds=30,
    verbosity=0,
)

clf.fit(
    X_train, y_cls_train,
    eval_set=[(X_test, y_cls_test)],
    verbose=False
)

print(f"Best iteration: {clf.best_iteration}")

# ─────────────────────────────────────────────
# STEP D: Evaluate Classifier
# ─────────────────────────────────────────────
y_pred_cls  = clf.predict(X_test)
y_prob      = clf.predict_proba(X_test)[:, 1]  # Probability of stockout
roc_auc     = roc_auc_score(y_cls_test, y_prob)
cm          = confusion_matrix(y_cls_test, y_pred_cls)

print("\n" + "=" * 50)
print("CLASSIFIER RESULTS (XGBoost)")
print("=" * 50)
print(f"\nROC-AUC Score: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(f"                 Predicted NO  Predicted YES")
print(f"Actual NO        {cm[0][0]:>10,}   {cm[0][1]:>12,}")
print(f"Actual YES       {cm[1][0]:>10,}   {cm[1][1]:>12,}")
print("\nWhat this means:")
print(f"  True Negatives  (correctly said 'no stockout'): {cm[0][0]:,}")
print(f"  False Positives (false alarm - said stockout, wasn't): {cm[0][1]:,}")
print(f"  False Negatives (missed a real stockout): {cm[1][0]:,}")
print(f"  True Positives  (correctly caught stockout): {cm[1][1]:,}")
print(f"\nDetailed report:")
print(classification_report(y_cls_test, y_pred_cls, target_names=["No Stockout","Stockout"]))

# ─────────────────────────────────────────────
# STEP E: Train LightGBM Regressor
# ─────────────────────────────────────────────
# LightGBM for the regression task:
# "How many days until stock hits zero?"
#
# We only train on rows where a stockout actually happens (days_to_zero is not NaN/0).
# Training a regression model on rows where there's no stockout would just
# teach it to predict nonsense numbers for rows that were never going to stock out.
#
# LightGBM vs XGBoost for this task:
# LightGBM uses "leaf-wise" tree growth instead of "level-wise".
# This means it finds the most impactful splits first,
# making it faster and often more accurate on larger datasets.

reg_mask_train = (y_reg_train > 0) & (~np.isnan(y_reg_train))
reg_mask_test  = (y_reg_test  > 0) & (~np.isnan(y_reg_test))

X_reg_train = X_train[reg_mask_train]
y_reg_train_clean = y_reg_train[reg_mask_train]
X_reg_test  = X_test[reg_mask_test]
y_reg_test_clean  = y_reg_test[reg_mask_test]

print(f"\nRegression training rows (stockout events only): {len(X_reg_train):,}")

print("Training LightGBM regressor...")
reg = lgb.LGBMRegressor(
    n_estimators=500,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)

reg.fit(
    X_reg_train, y_reg_train_clean,
    eval_set=[(X_reg_test, y_reg_test_clean)],
    callbacks=[lgb.early_stopping(30, verbose=False)]
)

# ─────────────────────────────────────────────
# STEP F: Evaluate Regressor
# ─────────────────────────────────────────────
y_reg_pred = reg.predict(X_reg_test)
y_reg_pred = np.clip(y_reg_pred, 0, 7)  # Clamp to 0-7 days range
mae  = mean_absolute_error(y_reg_test_clean, y_reg_pred)
# What % of predictions are within 1 day of correct?
within_1day = np.mean(np.abs(y_reg_test_clean - y_reg_pred) <= 1.0)
within_2day = np.mean(np.abs(y_reg_test_clean - y_reg_pred) <= 2.0)

print("\n" + "=" * 50)
print("REGRESSOR RESULTS (LightGBM)")
print("=" * 50)
print(f"Mean Absolute Error: {mae:.2f} days")
print(f"Within 1 day:        {within_1day:.1%} of predictions")
print(f"Within 2 days:       {within_2day:.1%} of predictions")
print(f"\nPrediction samples (predicted vs actual):")
sample_idx = np.random.choice(len(y_reg_test_clean), min(10, len(y_reg_test_clean)), replace=False)
for i in sample_idx:
    print(f"  Predicted: {y_reg_pred[i]:.1f} days  |  Actual: {y_reg_test_clean[i]:.0f} days")

# ─────────────────────────────────────────────
# STEP G: Feature importance chart
# ─────────────────────────────────────────────
print("\nGenerating feature importance chart...")
importance = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(importance["feature"], importance["importance"], color="#2E6DA4")
ax.set_xlabel("Importance Score (Gain)")
ax.set_title("Top 15 Features — XGBoost Classifier")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
os.makedirs("data/processed", exist_ok=True)
plt.savefig("data/processed/feature_importance.png", dpi=150)
print("Saved feature importance chart to data/processed/feature_importance.png")

# ─────────────────────────────────────────────
# STEP H: Save models
# ─────────────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)

joblib.dump(clf,          "saved_models/xgb_classifier_v1.pkl")
joblib.dump(reg,          "saved_models/lgbm_regressor_v1.pkl")
joblib.dump(FEATURE_COLS, "saved_models/feature_cols_v1.pkl")

print("\n" + "=" * 50)
print("MODELS SAVED")
print("=" * 50)
for fname in os.listdir("saved_models"):
    size = os.path.getsize(f"saved_models/{fname}") / 1024
    print(f"  {fname:<40} {size:.0f} KB")

print("\n✅ Training complete.")
print("\nQuick interpretation guide:")
print("  ROC-AUC > 0.85 = excellent  |  0.75-0.85 = good  |  < 0.75 = needs work")
print("  TTZ MAE < 1.5 days = excellent  |  1.5-2.5 = good  |  > 2.5 = needs work")