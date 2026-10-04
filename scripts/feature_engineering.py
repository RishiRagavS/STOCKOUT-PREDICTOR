"""
feature_engineering.py
-----------------------
Builds the feature matrix used to train the stockout prediction model.

Features created:
  - Lag features: sales 7, 14, 28 days ago
  - Rolling statistics: mean, std, max over 7 / 14 / 28-day windows
  - Sales velocity: % change over 7-day window (is demand accelerating?)
  - Calendar: day_of_week, day_of_month, month, is_weekend
  - Holiday / SNAP flags from M5 calendar
  - Indian festival proximity: days until / since nearest major festival
  - Inventory features: stock_after, stock_ratio (stock/reorder_point),
    days_since_reorder, consecutive_low_stock_days

Target:
  stockout (binary 0/1) — did a stockout occur on this day?

Input:
    data/processed/sales_long.csv
    data/processed/inventory_snapshots.csv

Output:
    data/processed/features_final.csv
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Indian festivals with approximate fixed dates (month, day)
# In a production system these would come from a dynamic calendar API
FESTIVALS = [
    (1, 14,  "pongal"),
    (1, 26,  "republic_day"),
    (3, 25,  "holi"),
    (4, 14,  "tamil_new_year"),
    (8, 15,  "independence_day"),
    (9, 7,   "onam"),
    (10, 2,  "dussehra"),
    (10, 24, "diwali"),
    (11, 4,  "diwali_end"),
    (12, 25, "christmas"),
]


def days_to_nearest_festival(date: pd.Timestamp) -> int:
    """Return days to the nearest festival (past or future), capped at 30."""
    year = date.year
    dists = []
    for month, day, _ in FESTIVALS:
        try:
            fd = pd.Timestamp(year=year, month=month, day=day)
            dists.append(abs((date - fd).days))
            # Also check previous/next year wrapping
            fd_prev = pd.Timestamp(year=year - 1, month=month, day=day)
            dists.append(abs((date - fd_prev).days))
        except ValueError:
            pass
    return min(dists) if dists else 30


def build_features(sales: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    print("Merging sales and inventory...")
    df = inv.merge(
        sales[["item_id","store_id","date","wday","month","year",
               "is_holiday","is_snap","event_name_1"]],
        on=["item_id","store_id","date"], how="left"
    )
    df = df.sort_values(["item_id","store_id","date"]).reset_index(drop=True)

    print("Building lag and rolling features...")
    grp = df.groupby(["item_id","store_id"])["sales"]

    for lag in [7, 14, 28]:
        df[f"sales_lag_{lag}"]        = grp.shift(lag)
        df[f"sales_rolling_mean_{lag}"] = grp.shift(1).transform(
            lambda x: x.rolling(lag, min_periods=1).mean())
        df[f"sales_rolling_std_{lag}"]  = grp.shift(1).transform(
            lambda x: x.rolling(lag, min_periods=1).std().fillna(0))
        df[f"sales_rolling_max_{lag}"]  = grp.shift(1).transform(
            lambda x: x.rolling(lag, min_periods=1).max())

    # Sales velocity: % change in 7-day rolling mean vs 14-day (is demand growing?)
    epsilon = 1e-6
    df["sales_velocity"] = (
        (df["sales_rolling_mean_7"] - df["sales_rolling_mean_14"])
        / (df["sales_rolling_mean_14"] + epsilon)
    )

    print("Building calendar features...")
    df["day_of_week"]   = pd.to_datetime(df["date"]).dt.dayofweek
    df["day_of_month"]  = pd.to_datetime(df["date"]).dt.day
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_payday"]     = df["day_of_month"].isin([1, 2, 30, 31]).astype(int)
    df["is_mid_month"]  = df["day_of_month"].between(14, 16).astype(int)
    df["month"]         = pd.to_datetime(df["date"]).dt.month

    print("Building festival proximity feature (this may take a moment)...")
    dates_series = pd.to_datetime(df["date"])
    unique_dates = dates_series.unique()
    date_to_festival_dist = {
        d: days_to_nearest_festival(pd.Timestamp(d)) for d in unique_dates
    }
    df["days_to_festival"] = dates_series.map(date_to_festival_dist)
    df["is_near_festival"]  = (df["days_to_festival"] <= 3).astype(int)

    print("Building inventory ratio features...")
    df["stock_ratio"] = df["stock_after"] / (df["reorder_point"] + 1e-6)
    df["stock_ratio"] = df["stock_ratio"].clip(0, 10)

    # Consecutive days with stock below reorder point (risk escalation signal)
    df["below_reorder"] = (df["stock_after"] <= df["reorder_point"]).astype(int)
    df["consecutive_low_stock"] = (
        df.groupby(["item_id","store_id"])["below_reorder"]
          .transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
    )

    print("Dropping rows with NaN lag features...")
    feature_cols = [c for c in df.columns if c.startswith("sales_lag_")]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


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
    # Identifiers (kept for reference, not used as features)
    "item_id", "store_id", "date",
    # Target
    "stockout",
]


def main():
    sales_path = os.path.join(PROCESSED_DIR, "sales_long.csv")
    inv_path   = os.path.join(PROCESSED_DIR, "inventory_snapshots.csv")

    for p in [sales_path, inv_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run prior scripts first.")
            return

    print("Loading data...")
    sales = pd.read_csv(sales_path, parse_dates=["date"])
    inv   = pd.read_csv(inv_path,   parse_dates=["date"])

    df = build_features(sales, inv)

    # Keep only defined columns that exist
    keep = [c for c in FEATURE_COLS if c in df.columns]
    df_out = df[keep]

    out_path = os.path.join(PROCESSED_DIR, "features_final.csv")
    print(f"Saving → {out_path}  shape={df_out.shape}")
    df_out.to_csv(out_path, index=False)

    stockout_rate = df_out["stockout"].mean()
    print(f"Stockout rate in feature set: {stockout_rate:.2%}")
    print("Done.")


if __name__ == "__main__":
    main()
