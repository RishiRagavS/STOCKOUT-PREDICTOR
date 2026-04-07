import pandas as pd
import numpy as np
import os

print("Loading inventory data...")
df = pd.read_csv("data/processed/inventory_features.csv", parse_dates=["date"])
df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

print(f"Loaded {len(df):,} rows, {df['sku_id'].nunique()} SKUs")

# ─────────────────────────────────────────────
# FEATURE GROUP 1: Stock-based features
# ─────────────────────────────────────────────
# These directly measure how close to zero we are.

# Cart reservation ratio: what fraction of stock is already spoken for
# If 40 units are in stock and 12 are in carts, ratio = 0.30
# A high ratio means demand is active RIGHT NOW
df["cart_reservation_ratio"] = (
    df["reserved_in_carts"] / df["current_stock"].replace(0, 1)
).round(4)

# Stock as a percentage of a "comfortable" level (we use 50 as baseline)
# Tells the model whether stock is dangerously low relative to normal
df["stock_pct_of_baseline"] = (df["available_stock"] / 50).clip(upper=2.0).round(4)

# ─────────────────────────────────────────────
# FEATURE GROUP 2: Velocity and acceleration
# ─────────────────────────────────────────────
# These measure HOW FAST stock is being consumed.
# A product with 5 units but selling 0.1/day is fine.
# A product with 5 units but selling 2/day will be gone in 2.5 days.

# Naive time-to-zero: current stock divided by recent sales rate
# This is the simplest possible prediction — the ML model will improve on it
# by incorporating all the other signals
df["velocity_per_day"] = df["sales_last_1hr"] * 6  # 1hr velocity × 6 = daily estimate

df["time_to_zero_naive"] = (
    df["available_stock"] / df["velocity_per_day"].replace(0, 0.01)
).clip(upper=999).round(2)

# Velocity acceleration: is demand speeding up or slowing down?
# 5-min rate vs 30-min average rate
# ratio > 1.0 means demand is ACCELERATING (more dangerous)
# ratio < 1.0 means demand is SLOWING DOWN (less dangerous)
v30_per_5min = (df["sales_last_30min"] / 6).replace(0, 0.01)
df["velocity_acceleration"] = (df["sales_last_5min"] / v30_per_5min).round(4)

# ─────────────────────────────────────────────
# FEATURE GROUP 3: Time-based features
# ─────────────────────────────────────────────
# Demand is not uniform across time. A Friday evening and a Tuesday
# morning have completely different demand profiles for the same product.

df["hour_of_day"] = df["date"].dt.hour  # 0-23
df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
df["is_month_start"] = (df["date"].dt.day <= 5).astype(int)  # Payday week
df["quarter"]     = df["date"].dt.quarter  # 1-4, captures seasonal patterns

# ─────────────────────────────────────────────
# FEATURE GROUP 4: Historical demand averages
# ─────────────────────────────────────────────
# These are the most powerful features.
# Instead of asking "how much is selling right now?"
# we ask "how much USUALLY sells at this time for this product?"
# The ratio between actual and expected tells us if today is unusual.

print("Computing historical demand averages (this takes ~30 seconds)...")

# Average sales for each SKU on each day of week
# e.g. FOODS_1_001 sells an average of 3.2 units on Sundays
dow_avg = (
    df.groupby(["sku_id", "day_of_week"])["qty_sold"]
    .mean()
    .round(3)
    .reset_index()
    .rename(columns={"qty_sold": "historical_avg_dow"})
)
df = df.merge(dow_avg, on=["sku_id", "day_of_week"], how="left")

# Average sales for each SKU in each month
# Captures seasonal patterns (ice cream sells more in summer)
month_avg = (
    df.groupby(["sku_id", "month"])["qty_sold"]
    .mean()
    .round(3)
    .reset_index()
    .rename(columns={"qty_sold": "historical_avg_month"})
)
df = df.merge(month_avg, on=["sku_id", "month"], how="left")

# How much is today's actual sales deviating from the historical average?
# Positive = selling more than usual (dangerous if stock is low)
# Negative = selling less than usual (safer)
df["demand_vs_dow_avg"] = (
    df["qty_sold"] - df["historical_avg_dow"]
).round(3)

df["demand_ratio_vs_avg"] = (
    df["qty_sold"] / df["historical_avg_dow"].replace(0, 0.01)
).clip(upper=10).round(3)

# ─────────────────────────────────────────────
# FEATURE GROUP 5: Rolling stock depletion rate
# ─────────────────────────────────────────────
# How fast has stock been falling over the last 3 and 7 days?
# This captures the trend, not just the current snapshot.

print("Computing rolling depletion rates...")

df["stock_change_3d"] = (
    df.groupby("sku_id")["available_stock"]
    .transform(lambda x: x.diff(3))
    .round(2)
)  # Negative = stock is falling (normal), Positive = restock happened

df["stock_change_7d"] = (
    df.groupby("sku_id")["available_stock"]
    .transform(lambda x: x.diff(7))
    .round(2)
)

# Rolling 7-day average sales per SKU
# Smoother signal than single-day sales
df["rolling_avg_sales_7d"] = (
    df.groupby("sku_id")["qty_sold"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
    .round(3)
)

# ─────────────────────────────────────────────
# FEATURE GROUP 6: Categorical encodings
# ─────────────────────────────────────────────
# ML models cannot use strings directly — "FOODS" means nothing to XGBoost.
# We convert categories to numbers.

# Label encoding: each unique category gets a unique integer
df["cat_encoded"]   = df["cat_id"].astype("category").cat.codes
df["event_encoded"] = df["event_name"].astype("category").cat.codes

# ─────────────────────────────────────────────
# Final cleanup
# ─────────────────────────────────────────────
# Fill any NaN that appeared from rolling/diff operations
# (first few rows of each SKU will have NaN for rolling features)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# ─────────────────────────────────────────────
# Quality report
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE ENGINEERING REPORT")
print("=" * 50)

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

print(f"Total features built: {len(FEATURE_COLS)}")
print(f"Total rows:           {len(df):,}")
print(f"Stockout rate:        {df['stockout_label'].mean():.1%}")
print(f"\nFeature list:")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {f}")

print(f"\nMissing values in feature columns:")
missing = df[FEATURE_COLS].isnull().sum()
print(missing[missing > 0].to_string() if missing.sum() > 0 else "  None - all clean!")

print(f"\nSample: top 5 rows with stockout_label=1")
print(df[df["stockout_label"]==1][FEATURE_COLS[:8] + ["stockout_label","days_to_zero"]].head().to_string())

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
output_path = "data/processed/features_final.csv"
df.to_csv(output_path, index=False)

size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"\n✅ Saved to {output_path}")
print(f"   Rows: {len(df):,}  |  Columns: {len(df.columns)}  |  Size: {size_mb:.1f} MB")