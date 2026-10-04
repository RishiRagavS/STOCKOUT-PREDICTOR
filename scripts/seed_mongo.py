"""
seed_mongo.py
-------------
Seeds MongoDB with processed data:
  - products collection        : unique (item_id, store_id) pairs with metadata
  - sales_snapshots collection : time-series sales data
  - inventory collection       : inventory snapshots
  - features collection        : latest feature vector per SKU (for live prediction)

Run AFTER all other scripts.
"""

import os, asyncio, json
from datetime import datetime
import pandas as pd
import motor.motor_asyncio
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MONGO_URI     = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME       = os.getenv("MONGO_DB_NAME", "stockout_predictor")

CATEGORY_MAP = {
    "FOODS":    "Food & Grocery",
    "HOBBIES":  "Hobbies & Leisure",
    "HOUSEHOLD":"Household Essentials",
}


async def seed():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    db     = client[DB_NAME]

    # ── Products ─────────────────────────────────────────────────────────────
    print("Seeding products...")
    inv = pd.read_csv(os.path.join(PROCESSED_DIR, "inventory_snapshots.csv"),
                      parse_dates=["date"])
    skus = inv[["item_id","store_id"]].drop_duplicates()

    await db.products.drop()
    products = []
    for _, row in skus.iterrows():
        cat_key  = str(row["item_id"]).split("_")[0]
        category = CATEGORY_MAP.get(cat_key, "General")
        products.append({
            "sku_id":    f"{row['item_id']}_{row['store_id']}",
            "item_id":   row["item_id"],
            "store_id":  row["store_id"],
            "category":  category,
            "created_at": datetime.utcnow(),
        })
    if products:
        await db.products.insert_many(products)
    print(f"  Inserted {len(products)} products")

    # ── Inventory Snapshots (time-series collection) ──────────────────────────
    print("Seeding inventory snapshots (time-series collection)...")
    try:
        await db.create_collection(
            "inventory_snapshots",
            timeseries={"timeField": "date", "metaField": "sku_id",
                        "granularity": "hours"}
        )
    except Exception:
        pass  # already exists

    await db.inventory_snapshots.drop()
    chunk_size = 5000
    records = []
    for _, row in inv.iterrows():
        records.append({
            "sku_id":     f"{row['item_id']}_{row['store_id']}",
            "item_id":    row["item_id"],
            "store_id":   row["store_id"],
            "date":       row["date"].to_pydatetime(),
            "stock_before": int(row["stock_before"]),
            "stock_after":  int(row["stock_after"]),
            "sales":        int(row["sales"]),
            "reorder_point":int(row["reorder_point"]),
            "stockout":     int(row["stockout"]),
        })
        if len(records) >= chunk_size:
            await db.inventory_snapshots.insert_many(records)
            records = []
    if records:
        await db.inventory_snapshots.insert_many(records)
    print(f"  Inserted {inv.shape[0]} inventory rows")

    # ── Latest Features per SKU (for live prediction) ─────────────────────────
    print("Seeding latest features...")
    feats = pd.read_csv(os.path.join(PROCESSED_DIR, "features_final.csv"),
                        parse_dates=["date"])
    latest = feats.sort_values("date").groupby(["item_id","store_id"]).last().reset_index()

    await db.features.drop()
    feat_docs = []
    for _, row in latest.iterrows():
        doc = row.to_dict()
        doc["sku_id"] = f"{row['item_id']}_{row['store_id']}"
        if "date" in doc:
            doc["date"] = str(doc["date"])
        feat_docs.append(doc)
    if feat_docs:
        await db.features.insert_many(feat_docs)
    print(f"  Inserted {len(feat_docs)} feature documents")

    # ── Indexes ───────────────────────────────────────────────────────────────
    print("Creating indexes...")
    await db.products.create_index([("sku_id", 1)], unique=True)
    await db.products.create_index([("store_id", 1)])
    await db.features.create_index([("sku_id", 1)], unique=True)
    await db.predictions.create_index([("sku_id", 1), ("generated_at", -1)])
    await db.alerts.create_index([("store_id", 1), ("status", 1)])
    await db.alerts.create_index([("triggered_at", -1)])

    print("✓ MongoDB seeding complete.")
    client.close()


if __name__ == "__main__":
    asyncio.run(seed())
