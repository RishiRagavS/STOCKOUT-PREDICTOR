import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db     = client[os.getenv("MONGO_DB_NAME")]

print(f"Connected to MongoDB: {db.name}")

# ─────────────────────────────────────────────
# Load the processed data
# ─────────────────────────────────────────────
print("\nLoading features_final.csv...")
df = pd.read_csv("data/processed/features_final.csv", parse_dates=["date"])
df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
print(f"Loaded {len(df):,} rows, {df['sku_id'].nunique()} SKUs")

# ─────────────────────────────────────────────
# COLLECTION 1: products
# One document per unique SKU — the reference catalogue
# ─────────────────────────────────────────────
print("\n--- Seeding products collection ---")
db["products"].drop()

products = []
for sku_id, group in df.groupby("sku_id"):
    row = group.iloc[0]
    products.append({
        "sku_id":         sku_id,
        "name":           sku_id.replace("_", " ").title(),
        "category":       row["cat_id"],
        "department":     row["dept_id"],
        "dark_store_id":  row["dark_store_id"],
        "state_id":       row["state_id"],
        "reorder_point":  8,
        "lead_time_days": 2,
        "created_at":     datetime.utcnow()
    })

db["products"].insert_many(products)
print(f"✅ Inserted {len(products)} products")

# ─────────────────────────────────────────────
# COLLECTION 2: sales_events
# One document per day per SKU where qty_sold > 0
# In a real system this would be one per transaction,
# but daily is fine for our dataset
# ─────────────────────────────────────────────
print("\n--- Seeding sales_events collection ---")
db["sales_events"].drop()

sales_rows = df[df["qty_sold"] > 0].copy()
sales_docs = []
for _, row in sales_rows.iterrows():
    sales_docs.append({
        "timestamp":     row["date"].to_pydatetime(),
        "sku_id":        row["sku_id"],
        "dark_store_id": row["dark_store_id"],
        "quantity_sold": int(row["qty_sold"]),
        "event_name":    row["event_name"],
        "is_snap_day":   int(row["is_snap_day"]),
    })

# Insert in batches of 5000 — faster than one by one
BATCH = 5000
for i in range(0, len(sales_docs), BATCH):
    db["sales_events"].insert_many(sales_docs[i:i+BATCH])
    print(f"  Inserted sales batch {i//BATCH + 1}/{(len(sales_docs)//BATCH)+1}")

print(f"✅ Inserted {len(sales_docs):,} sales events")

# ─────────────────────────────────────────────
# COLLECTION 3: inventory_snapshots
# One document per day per SKU — the full feature snapshot
# This is what the API will query to build features for prediction
# ─────────────────────────────────────────────
print("\n--- Seeding inventory_snapshots collection ---")

# Drop and recreate as a proper time series collection
if "inventory_snapshots" in db.list_collection_names():
    db["inventory_snapshots"].drop()
    print("  Dropped existing inventory_snapshots")

db.create_collection(
    "inventory_snapshots",
    timeseries={
        "timeField":   "timestamp",
        "metaField":   "sku_id",
        "granularity": "hours"   # daily data fits "hours" granularity better than "minutes"
    }
)
print("  Created time series collection")

snapshot_docs = []
for _, row in df.iterrows():
    snapshot_docs.append({
        "timestamp":             row["date"].to_pydatetime(),
        "sku_id":                row["sku_id"],
        "dark_store_id":         row["dark_store_id"],

        # Raw inventory state
        "current_stock":         int(row["current_stock"]),
        "reserved_in_carts":     int(row["reserved_in_carts"]),
        "available_stock":       int(row["available_stock"]),
        "qty_sold":              int(row["qty_sold"]),

        # Velocity
        "sales_last_5min":       int(row["sales_last_5min"]),
        "sales_last_30min":      int(row["sales_last_30min"]),
        "sales_last_1hr":        int(row["sales_last_1hr"]),

        # All engineered features (stored so API can retrieve them directly)
        "cart_reservation_ratio":  float(row["cart_reservation_ratio"]),
        "stock_pct_of_baseline":   float(row["stock_pct_of_baseline"]),
        "velocity_per_day":        float(row["velocity_per_day"]),
        "time_to_zero_naive":      float(row["time_to_zero_naive"]),
        "velocity_acceleration":   float(row["velocity_acceleration"]),
        "historical_avg_dow":      float(row["historical_avg_dow"]),
        "historical_avg_month":    float(row["historical_avg_month"]),
        "demand_vs_dow_avg":       float(row["demand_vs_dow_avg"]),
        "demand_ratio_vs_avg":     float(row["demand_ratio_vs_avg"]),
        "stock_change_3d":         float(row["stock_change_3d"]),
        "stock_change_7d":         float(row["stock_change_7d"]),
        "rolling_avg_sales_7d":    float(row["rolling_avg_sales_7d"]),
        "has_event":               int(row["has_event"]),
        "is_snap_day":             int(row["is_snap_day"]),
        "cat_encoded":             int(row["cat_encoded"]),
        "event_encoded":           int(row["event_encoded"]),
        "day_of_week":             int(row["day_of_week"]),
        "is_weekend":              int(row["is_weekend"]),
        "is_month_start":          int(row["is_month_start"]),
        "quarter":                 int(row["quarter"]),

        # Ground truth labels (useful for later model evaluation)
        "stockout_label":          int(row["stockout_label"]),
        "days_to_zero":            float(row["days_to_zero"]) if row["days_to_zero"] > 0 else None,
    })

for i in range(0, len(snapshot_docs), BATCH):
    db["inventory_snapshots"].insert_many(snapshot_docs[i:i+BATCH])
    print(f"  Inserted snapshot batch {i//BATCH + 1}/{(len(snapshot_docs)//BATCH)+1}")

print(f"✅ Inserted {len(snapshot_docs):,} inventory snapshots")

# ─────────────────────────────────────────────
# Create indexes for fast queries
# ─────────────────────────────────────────────
print("\n--- Creating indexes ---")
db["products"].create_index([("sku_id", ASCENDING)], unique=True)
db["sales_events"].create_index([("sku_id", ASCENDING), ("timestamp", DESCENDING)])
db["products"].create_index([("category", ASCENDING)])
print("✅ Indexes created")

# ─────────────────────────────────────────────
# Verify everything
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("MONGODB SEEDING COMPLETE")
print("=" * 50)
for coll in ["products", "sales_events", "inventory_snapshots"]:
    count = db[coll].count_documents({})
    print(f"  {coll:<30} {count:>8,} documents")

print("\nSample product document:")
sample = db["products"].find_one({}, {"_id": 0})
for k, v in sample.items():
    print(f"  {k}: {v}")

print("\nSample snapshot document (latest for first SKU):")
first_sku = db["products"].find_one({}, {"sku_id": 1, "_id": 0})["sku_id"]
snap = db["inventory_snapshots"].find_one(
    {"sku_id": first_sku},
    sort=[("timestamp", DESCENDING)],
    projection={"_id": 0, "sku_id": 1, "timestamp": 1,
                "current_stock": 1, "available_stock": 1,
                "stockout_label": 1, "days_to_zero": 1}
)
for k, v in snap.items():
    print(f"  {k}: {v}")