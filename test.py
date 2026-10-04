"""
test.py
-------
Quick smoke test — verifies MongoDB connection and collection setup.
Run this after starting MongoDB and before running the full pipeline.

Usage:
    python test.py
"""

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

import motor.motor_asyncio

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB_NAME", "stockout_predictor")


async def test():
    print(f"Connecting to {MONGO_URI} ...")
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    db     = client[DB_NAME]

    # Ping
    await db.command("ping")
    print(f"✓ Connected to DB: {db.name}")

    # List collections
    colls = await db.list_collection_names()
    print(f"✓ Collections: {colls if colls else '(none yet — run seed_mongo.py)'}")

    # Count documents in key collections
    for coll_name in ["products", "features", "predictions", "alerts", "inventory_snapshots"]:
        try:
            count = await db[coll_name].count_documents({})
            print(f"  {coll_name}: {count} documents")
        except Exception:
            pass

    client.close()
    print("\n✓ All checks passed.")


if __name__ == "__main__":
    asyncio.run(test())
