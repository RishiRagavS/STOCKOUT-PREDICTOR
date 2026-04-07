import pandas as pd
import numpy as np
import os

np.random.seed(42)  # Makes random numbers reproducible — same output every run

print("Loading processed sales data...")
df = pd.read_csv("data/processed/sales_long.csv", parse_dates=["date"])

print(f"Loaded {len(df):,} rows, {df['sku_id'].nunique()} SKUs")

# ─────────────────────────────────────────────
# STEP A: Fill missing event columns
# ─────────────────────────────────────────────
# NaN in event_name means "no event" — replace with the string "none"
# so MongoDB and the ML model can handle it cleanly
df["event_name"] = df["event_name"].fillna("none")
df["event_type"] = df["event_type"].fillna("none")

# ─────────────────────────────────────────────
# STEP B: Simulate inventory for each SKU
# ─────────────────────────────────────────────
# We process one SKU at a time, building a realistic stock history
# for each one independently.

all_snapshots = []

skus = df["sku_id"].unique()
print(f"\nSimulating inventory for {len(skus)} SKUs...")

for i, sku_id in enumerate(skus):
    # Get all rows for this SKU, sorted by date
    sku_df = df[df["sku_id"] == sku_id].sort_values("date").copy()

    # Each SKU starts with a different random stock level
    # This makes the dataset more realistic — not all products start equal
    stock = np.random.randint(20, 55)

    for _, row in sku_df.iterrows():
        # ── Cart reservations ──
        # Items sitting in active customer carts haven't been "sold" yet
        # but they're effectively unavailable.
        # We simulate this as 8-18% of current stock.
        if stock > 0:
            reservation_pct = np.random.uniform(0.08, 0.18)
            reserved = int(stock * reservation_pct)
            reserved = min(reserved, stock - 1)  # Can't reserve more than we have
        else:
            reserved = 0

        available = max(0, stock - reserved)

        # ── Velocity metrics ──
        # In a real system these come from the last N minutes of sales events.
        # Here we simulate them based on today's qty_sold with some noise.
        # The "noise" makes each 5-min and 30-min window feel independent.
        qty = row["qty_sold"]

        # Scale daily sales down to 5-min and 30-min windows
        # A day has 288 five-minute slots, 48 thirty-minute slots
        # But quick-commerce is concentrated in peak hours,
        # so we use a higher concentration factor (divide by smaller numbers)
        sales_5m  = max(0, int(qty / 60  + np.random.randint(0, 2)))
        sales_30m = max(0, int(qty / 12  + np.random.randint(0, 3)))
        sales_1hr = max(0, int(qty / 6   + np.random.randint(0, 4)))

        # ── Build the snapshot document ──
        snapshot = {
            "sku_id":            sku_id,
            "dept_id":           row["dept_id"],
            "cat_id":            row["cat_id"],
            "dark_store_id":     row["dark_store_id"],
            "state_id":          row["state_id"],
            "date":              row["date"],
            "qty_sold":          qty,

            # Inventory state
            "current_stock":     int(stock),
            "reserved_in_carts": int(reserved),
            "available_stock":   int(available),

            # Velocity (simulated real-time windows)
            "sales_last_5min":   sales_5m,
            "sales_last_30min":  sales_30m,
            "sales_last_1hr":    sales_1hr,

            # Time signals
            "weekday":           row["weekday"],
            "wday":              row["wday"],
            "month":             row["month"],
            "year":              row["year"],

            # Demand drivers
            "event_name":        row["event_name"],
            "event_type":        row["event_type"],
            "has_event":         int(row["has_event"]),
            "is_snap_day":       int(row["is_snap_day"]),
        }
        all_snapshots.append(snapshot)

        # ── Deplete stock ──
        stock = max(0, stock - qty)

        # ── Simulate replenishment ──
        # When stock gets critically low, there's a chance a restock arrives.
        # 40% chance when below 15 units — simulates the operations team
        # noticing and triggering a replenishment order.
        if stock < 8 and np.random.random() < 0.15:
             restock_qty = np.random.randint(15, 35)
             stock += restock_qty

    if (i + 1) % 25 == 0:
        print(f"  Processed {i+1}/{len(skus)} SKUs...")

# ─────────────────────────────────────────────
# STEP C: Build the final dataframe
# ─────────────────────────────────────────────
print("\nBuilding final dataframe...")
inventory_df = pd.DataFrame(all_snapshots)

# ─────────────────────────────────────────────
# STEP D: Create the ML labels
# ─────────────────────────────────────────────
# This is the most important part.
# For every row, we look 7 days into the future.
# If available_stock hits zero in that window, label = 1 (stockout incoming)
# and we record exactly how many days until it happens.
#
# Why 7 days? Because our data is daily (M5 is daily sales).
# In a real system with 5-minute snapshots you'd use 30-60 minutes.
# The concept is identical — just a different time unit.

print("Creating ML labels (looking 7 days ahead for each SKU)...")
inventory_df = inventory_df.sort_values(["sku_id", "date"]).reset_index(drop=True)

inventory_df["stockout_label"]      = 0
inventory_df["days_to_zero"]        = np.nan

LOOKAHEAD = 7  # Look 7 days into the future

for sku_id, group in inventory_df.groupby("sku_id"):
    group = group.reset_index(drop=False)  # keep original index in 'index' column

    for i in range(len(group)):
        future = group.iloc[i+1 : i+1+LOOKAHEAD]

        if len(future) == 0:
            continue

        # Find the first future row where available_stock hits zero
        zero_rows = future[future["available_stock"] <= 0]

        if len(zero_rows) > 0:
            days_until = zero_rows.iloc[0].name - i  # how many steps away
            orig_idx   = group.iloc[i]["index"]

            inventory_df.loc[orig_idx, "stockout_label"] = 1
            inventory_df.loc[orig_idx, "days_to_zero"]   = days_until

# ─────────────────────────────────────────────
# STEP E: Quality report
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("INVENTORY DATASET QUALITY REPORT")
print("=" * 50)

total = len(inventory_df)
stockouts = inventory_df["stockout_label"].sum()

print(f"Total rows:              {total:,}")
print(f"Stockout events (label=1): {stockouts:,} ({stockouts/total:.1%})")
print(f"Non-stockout (label=0):    {total-stockouts:,} ({(total-stockouts)/total:.1%})")
print(f"\nStock level stats:")
print(inventory_df[["current_stock","available_stock","reserved_in_carts"]].describe().round(1).to_string())
print(f"\nVelocity stats:")
print(inventory_df[["sales_last_5min","sales_last_30min","sales_last_1hr"]].describe().round(2).to_string())
print(f"\nDays-to-zero stats (stockout rows only):")
print(inventory_df["days_to_zero"].dropna().describe().round(1).to_string())
print(f"\nStockout rate by category:")
print(inventory_df.groupby("cat_id")["stockout_label"].mean().round(3).to_string())

# ─────────────────────────────────────────────
# STEP F: Save
# ─────────────────────────────────────────────
output_path = "data/processed/inventory_features.csv"
inventory_df.to_csv(output_path, index=False)

size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"\n✅ Saved to {output_path}")
print(f"   Rows: {len(inventory_df):,}  |  Columns: {len(inventory_df.columns)}  |  Size: {size_mb:.1f} MB")
print(f"\nAll columns:")
print(inventory_df.columns.tolist())