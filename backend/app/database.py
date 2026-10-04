"""
database.py
-----------
Async MongoDB connection and collection setup using Motor.
Includes time-series collection creation for sales data.
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB_NAME", "stockout_predictor")

client: AsyncIOMotorClient = None
db     = None


async def connect_db():
    global client, db
    client = AsyncIOMotorClient(MONGO_URI)
    db     = client[DB_NAME]
    await setup_collections()
    print(f"[DB] Connected to MongoDB: {DB_NAME}")


async def close_db():
    if client:
        client.close()
        print("[DB] MongoDB connection closed.")


async def setup_collections():
    """Create time-series and regular collections with indexes."""
    existing = await db.list_collection_names()

    # Time-series collection for inventory snapshots
    if "inventory_snapshots" not in existing:
        try:
            await db.create_collection(
                "inventory_snapshots",
                timeseries={
                    "timeField":   "date",
                    "metaField":   "sku_id",
                    "granularity": "hours",
                }
            )
        except Exception as e:
            print(f"[DB] Time-series collection note: {e}")

    # Indexes
    await db.products.create_index([("sku_id", 1)], unique=True, background=True)
    await db.features.create_index([("sku_id", 1)], unique=True, background=True)
    await db.predictions.create_index([("sku_id", 1), ("generated_at", -1)], background=True)
    await db.alerts.create_index([("store_id", 1), ("status", 1)], background=True)
    await db.alerts.create_index([("triggered_at", -1)], background=True)


def get_db():
    return db
