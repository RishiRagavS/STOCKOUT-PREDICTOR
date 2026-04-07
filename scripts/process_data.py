import pandas as pd
import numpy as np
import os

print("Loading raw files...")
sales    = pd.read_csv("data/raw/sales_train_evaluation.csv")
calendar = pd.read_csv("data/raw/calendar.csv")

# ─────────────────────────────────────────────
# STEP A: Take a working subset
# ─────────────────────────────────────────────
# Full dataset = 30,490 products × 1,941 days = 59 million rows after melting.
# That would take forever to process on a laptop.
# We take 100 products and 2 years (730 days) — still 73,000 rows, plenty for ML.

# Pick 100 products spread across different categories
products_per_category = 25
subset_ids = (
    sales.groupby("cat_id")
    .apply(lambda x: x.head(products_per_category))
    .reset_index(drop=True)
)

print(f"Working with {len(subset_ids)} products across categories:")
print(subset_ids["cat_id"].value_counts().to_string())

# Take only 730 days (2 years of data)
day_cols = [c for c in sales.columns if c.startswith("d_")][:730]

sales_subset = subset_ids[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + day_cols].copy()

print(f"\nSubset shape before melting: {sales_subset.shape}")

# ─────────────────────────────────────────────
# STEP B: Melt from wide to long format
# ─────────────────────────────────────────────
# This is the core transformation.
# id_vars    = columns to KEEP as-is (the metadata)
# value_vars = columns to COLLAPSE into rows (d_1 through d_730)
# var_name   = what to call the new "which day" column
# value_name = what to call the new "sales number" column

print("\nMelting wide format to long format...")
sales_long = sales_subset.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    value_vars=day_cols,
    var_name="day",
    value_name="qty_sold"
)

print(f"Shape after melting: {sales_long.shape}")
print(f"Expected rows: {len(subset_ids)} products × {len(day_cols)} days = {len(subset_ids) * len(day_cols)}")

# ─────────────────────────────────────────────
# STEP C: Join with calendar to get real dates
# ─────────────────────────────────────────────
# Right now "day" column contains "d_1", "d_2" etc.
# We join with calendar to replace those with real dates like "2011-01-29"
# We also bring in event data and snap days from the calendar.

print("\nJoining with calendar...")
calendar_slim = calendar[[
    "d", "date", "weekday", "wday", "month", "year",
    "event_name_1", "event_type_1",
    "snap_CA", "snap_TX", "snap_WI"
]].copy()

sales_long = sales_long.merge(
    calendar_slim,
    left_on="day",      # match "d_1" in sales_long
    right_on="d",       # with "d" in calendar
    how="left"
)

# Convert date column to actual datetime type
# Right now it's a string like "2011-01-29"
# After this it becomes a proper date Python understands
sales_long["date"] = pd.to_datetime(sales_long["date"])

# Rename columns to our naming convention
sales_long.rename(columns={
    "item_id":   "sku_id",
    "store_id":  "dark_store_id",
    "event_name_1": "event_name",
    "event_type_1": "event_type"
}, inplace=True)

# ─────────────────────────────────────────────
# STEP D: Add a snap_flag column
# ─────────────────────────────────────────────
# SNAP days (food stamp benefit days) cause demand spikes.
# Each row knows which state the store is in, so we pick
# the right snap column for that store.

def get_snap_flag(row):
    state = row["state_id"]
    if state == "CA":
        return row["snap_CA"]
    elif state == "TX":
        return row["snap_TX"]
    elif state == "WI":
        return row["snap_WI"]
    return 0

sales_long["is_snap_day"] = sales_long.apply(get_snap_flag, axis=1)

# ─────────────────────────────────────────────
# STEP E: Add has_event column
# ─────────────────────────────────────────────
# 1 if there was a special event that day, 0 otherwise
sales_long["has_event"] = sales_long["event_name"].notna().astype(int)

# ─────────────────────────────────────────────
# STEP F: Clean up
# ─────────────────────────────────────────────
# Drop columns we no longer need
sales_long.drop(columns=["id", "d", "day", "snap_CA", "snap_TX", "snap_WI"], inplace=True)

# Make sure qty_sold has no negatives or NaN
sales_long["qty_sold"] = sales_long["qty_sold"].fillna(0).clip(lower=0).astype(int)

# Sort by product and date — important for time-series work
sales_long = sales_long.sort_values(["sku_id", "dark_store_id", "date"]).reset_index(drop=True)

# ─────────────────────────────────────────────
# STEP G: Data quality checks
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("DATA QUALITY REPORT")
print("=" * 50)

print(f"Total rows:           {len(sales_long):,}")
print(f"Unique SKUs:          {sales_long['sku_id'].nunique()}")
print(f"Unique stores:        {sales_long['dark_store_id'].nunique()}")
print(f"Date range:           {sales_long['date'].min().date()} → {sales_long['date'].max().date()}")
print(f"Missing values:       {sales_long.isnull().sum().sum()}")
print(f"Negative sales:       {(sales_long['qty_sold'] < 0).sum()}")
print(f"Zero-sales rows:      {(sales_long['qty_sold'] == 0).sum():,} ({(sales_long['qty_sold'] == 0).mean():.1%})")
print(f"\nSales stats:")
print(sales_long["qty_sold"].describe().round(2).to_string())
print(f"\nCategories:")
print(sales_long["cat_id"].value_counts().to_string())
print(f"\nStores:")
print(sales_long["dark_store_id"].value_counts().to_string())
print(f"\nEvent days in dataset:")
print(sales_long[sales_long["has_event"]==1]["event_name"].value_counts().head(10).to_string())

# ─────────────────────────────────────────────
# STEP H: Save processed file
# ─────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/sales_long.csv"
sales_long.to_csv(output_path, index=False)

print(f"\n✅ Saved to {output_path}")
print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
print(f"\nFinal columns: {sales_long.columns.tolist()}")
print("\nSample rows:")
print(sales_long[sales_long["qty_sold"] > 0].head(5).to_string())