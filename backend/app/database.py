from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
import os
from dotenv import load_dotenv

load_dotenv()  # Reads your .env file

client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB_NAME")]

# These are your 4 collection references.
# Think of each one like a table in a spreadsheet.
products_col    = db["products"]
snapshots_col   = db["inventory_snapshots"]
sales_col       = db["sales_events"]
predictions_col = db["predictions"]


async def create_indexes():
    """
    Indexes make queries fast. Without an index, MongoDB reads every
    single document to find what you want — like reading a whole book
    to find one word. An index is like the book's index at the back.
    
    We index on sku_id + timestamp because 95% of our queries are:
    "give me the latest snapshot for SKU_001"
    """
    await snapshots_col.create_index(
        [("sku_id", ASCENDING), ("timestamp", DESCENDING)]
    )
    await snapshots_col.create_index([("timestamp", DESCENDING)])
    await sales_col.create_index(
        [("sku_id", ASCENDING), ("timestamp", DESCENDING)]
    )
    await predictions_col.create_index(
        [("sku_id", ASCENDING), ("timestamp", DESCENDING)]
    )
    print("✅ Indexes created")


async def setup_time_series_collection():
    """
    MongoDB Time Series collections are specially optimised for data
    that arrives in time order (like sensor readings, stock prices,
    or our inventory snapshots). They compress and query faster than
    a regular collection for this use case.
    """
    existing = await db.list_collection_names()
    if "inventory_snapshots" not in existing:
        await db.create_collection(
            "inventory_snapshots",
            timeseries={
                "timeField": "timestamp",   # Which field holds the time
                "metaField": "sku_id",      # Which field identifies the series
                "granularity": "minutes"    # Optimises internal storage for minute-level data
            }
        )
        print("✅ Time series collection created")
    else:
        print("ℹ️  inventory_snapshots already exists")