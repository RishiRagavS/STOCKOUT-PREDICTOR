"""
process_data.py
---------------
Reads the raw M5 Forecasting dataset and transforms it into a long-format
time-series DataFrame. Merges calendar information (dates, holidays, SNAP days,
promotional events) so downstream scripts have full date context.

Input:
    data/raw/sales_train_evaluation.csv
    data/raw/calendar.csv

Output:
    data/processed/sales_long.csv
"""

import os, sys
import pandas as pd
import numpy as np

RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def check_raw_files():
    required = ["sales_train_evaluation.csv", "calendar.csv"]
    missing  = [f for f in required if not os.path.exists(os.path.join(RAW_DIR, f))]
    if missing:
        print("ERROR: Missing raw data files:", missing)
        print("Download from: https://www.kaggle.com/c/m5-forecasting-accuracy")
        print(f"Place them in: {os.path.abspath(RAW_DIR)}")
        sys.exit(1)


def load_calendar(path):
    cal = pd.read_csv(path)
    cal["date"] = pd.to_datetime(cal["date"])
    cal["is_holiday"]  = cal["event_name_1"].notna().astype(int)
    cal["is_snap_ca"]  = (cal["snap_CA"] == 1).astype(int)
    cal["is_snap_tx"]  = (cal["snap_TX"] == 1).astype(int)
    cal["is_snap_wi"]  = (cal["snap_WI"] == 1).astype(int)
    keep = ["d","date","wm_yr_wk","weekday","wday","month","year",
            "event_name_1","event_type_1","event_name_2","event_type_2",
            "is_holiday","is_snap_ca","is_snap_tx","is_snap_wi"]
    return cal[keep]


def melt_sales(path, calendar):
    print("Loading sales data (this may take a moment)...")
    sales  = pd.read_csv(path)
    id_cols = ["id","item_id","dept_id","cat_id","store_id","state_id"]
    d_cols  = [c for c in sales.columns if c.startswith("d_")]
    print(f"Melting {len(d_cols)} day-columns for {len(sales)} products...")
    sales = sales.head(500)
    long = sales.melt(id_vars=id_cols, value_vars=d_cols,
                      var_name="d", value_name="sales")
    long = long.merge(calendar, on="d", how="left")
    long = long.sort_values(["id","date"]).reset_index(drop=True)
    long["sales"] = long["sales"].fillna(0).astype(int)
    return long


def add_snap_flag(df):
    conditions = [
        (df["state_id"] == "CA"),
        (df["state_id"] == "TX"),
        (df["state_id"] == "WI"),
    ]
    choices = [df["is_snap_ca"], df["is_snap_tx"], df["is_snap_wi"]]
    return np.select(conditions, choices, default=0).astype(int)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    check_raw_files()
    calendar   = load_calendar(os.path.join(RAW_DIR, "calendar.csv"))
    sales_long = melt_sales(os.path.join(RAW_DIR, "sales_train_evaluation.csv"), calendar)
    print("Adding per-state SNAP flags...")
    sales_long["is_snap"] = add_snap_flag(sales_long)
    out = os.path.join(PROCESSED_DIR, "sales_long.csv")
    print(f"Saving → {out}")
    sales_long.to_csv(out, index=False)
    print(f"Done. Shape: {sales_long.shape}")


if __name__ == "__main__":
    main()
