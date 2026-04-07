
import asyncio
from backend.app.database import db, create_indexes, setup_time_series_collection

async def test():
    await setup_time_series_collection()
    await create_indexes()
    print('DB name:', db.name)
    colls = await db.list_collection_names()
    print('Collections:', colls)

asyncio.run(test())
