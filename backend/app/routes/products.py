from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from backend.app.database import get_db

router = APIRouter(prefix="/products", tags=["Products"])


@router.get("/")
async def list_products(
    store_id: Optional[str] = Query(None, description="Filter by store"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit:    int            = Query(50, ge=1, le=500),
    skip:     int            = Query(0, ge=0),
):
    db = get_db()
    query = {}
    if store_id:
        query["store_id"] = store_id
    if category:
        query["category"] = {"$regex": category, "$options": "i"}

    products = await db.products.find(query, {"_id": 0}).skip(skip).limit(limit).to_list(limit)
    total    = await db.products.count_documents(query)
    return {"total": total, "results": products}


@router.get("/{sku_id}")
async def get_product(sku_id: str):
    db  = get_db()
    doc = await db.products.find_one({"sku_id": sku_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Product {sku_id} not found")
    return doc


@router.get("/{sku_id}/inventory")
async def get_inventory(
    sku_id: str,
    limit:  int = Query(30, ge=1, le=365),
):
    db  = get_db()
    docs = await (
        db.inventory_snapshots
          .find({"sku_id": sku_id}, {"_id": 0})
          .sort("date", -1)
          .limit(limit)
          .to_list(limit)
    )
    if not docs:
        raise HTTPException(status_code=404, detail=f"No inventory data for {sku_id}")
    return {"sku_id": sku_id, "records": docs}
