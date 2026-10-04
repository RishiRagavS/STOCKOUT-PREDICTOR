"""
generate_inventory.py
---------------------
Creates synthetic inventory snapshots for each (item_id, store_id) pair.

IMPORTANT NOTE:
    The M5 dataset contains sales quantities only — not actual stock levels.
    This script simulates a plausible inventory system using a simple model:
      - Each SKU starts with a random initial stock (drawn from realistic ranges
        based on its sales velocity).
      - Daily sales deplete the stock.
      - When stock drops below a reorder_point, a reorder arrives after a
        configurable lead_time (default 2 days).
      - A stockout is recorded when stock reaches 0 before reorder arrives.

    In production, this layer would be replaced by live POS / WMS inventory feeds.

Input:
    data/processed/sales_long.csv

Output:
    data/processed/inventory_snapshots.csv
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SALES_PATH    = os.path.join(PROCESSED_DIR, "sales_long.csv")
OUT_PATH      = os.path.join(PROCESSED_DIR, "inventory_snapshots.csv")

LEAD_TIME_DAYS   = 2      # days until reorder arrives
SAFETY_STOCK_MUL = 1.5    # reorder_point = safety_stock_mul * avg_7d_sales
REORDER_QTY_MUL  = 14     # order enough for ~14 days of avg sales
SEED             = 42

np.random.seed(SEED)


def simulate_inventory(group: pd.DataFrame) -> pd.DataFrame:
    """Simulate daily inventory for a single (item_id, store_id) pair."""
    group = group.sort_values("date").reset_index(drop=True)
    sales = group["sales"].values

    avg_sales  = max(sales.mean(), 0.1)
    reorder_pt = max(int(SAFETY_STOCK_MUL * avg_sales * LEAD_TIME_DAYS), 1)
    reorder_qty = max(int(REORDER_QTY_MUL * avg_sales), 5)

    # Random starting stock between 1–3 weeks of average sales
    init_stock = int(np.random.uniform(7, 21) * avg_sales)
    stock = init_stock

    records = []
    pending_reorder_day = None

    for i, day_sales in enumerate(sales):
        date = group.loc[i, "date"]

        # Receive reorder if it arrived
        if pending_reorder_day is not None and i >= pending_reorder_day:
            stock += reorder_qty
            pending_reorder_day = None

        # Deplete stock by today's sales (can't go below 0)
        stock_before = stock
        sold = min(day_sales, stock)
        stock = max(stock - day_sales, 0)

        stockout_flag = int(stock == 0 and day_sales > 0)

        records.append({
            "item_id":         group.loc[i, "item_id"],
            "store_id":        group.loc[i, "store_id"],
            "date":            date,
            "stock_before":    stock_before,
            "sales":           int(sold),
            "stock_after":     int(stock),
            "reorder_point":   reorder_pt,
            "reorder_qty":     reorder_qty,
            "stockout":        stockout_flag,
        })

        # Trigger reorder if stock dropped below reorder point
        if stock <= reorder_pt and pending_reorder_day is None:
            pending_reorder_day = i + LEAD_TIME_DAYS

    return pd.DataFrame(records)


def main():
    if not os.path.exists(SALES_PATH):
        print("ERROR: sales_long.csv not found. Run process_data.py first.")
        return

    print("Loading sales data...")
    sales = pd.read_csv(SALES_PATH, parse_dates=["date"])

    # Work on a manageable subset: first 500 unique SKUs per store for speed
    # In production you'd run on all ~30k SKUs
    skus = sales[["item_id","store_id"]].drop_duplicates()
    print(f"Total SKU×Store pairs: {len(skus)}")

    print("Simulating inventory (this takes a few minutes for large datasets)...")
    results = []
    groups = sales.groupby(["item_id","store_id"])
    total  = len(groups)
    for i, ((item_id, store_id), grp) in enumerate(groups):
        if i % 1000 == 0:
            print(f"  {i}/{total} pairs processed...")
        results.append(simulate_inventory(grp))

    print("Concatenating results...")
    inv = pd.concat(results, ignore_index=True)

    stockout_rate = inv["stockout"].mean()
    print(f"Stockout rate across simulation: {stockout_rate:.2%}")
    print(f"Saving → {OUT_PATH}")
    inv.to_csv(OUT_PATH, index=False)
    print(f"Done. Shape: {inv.shape}")


if __name__ == "__main__":
    main()
