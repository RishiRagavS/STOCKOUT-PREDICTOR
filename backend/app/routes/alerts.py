"""
alerts.py
---------
GET   /alerts              — list retailer alerts (filterable by store, status)
PATCH /alerts/{alert_id}/read — mark alert as read
GET   /alerts/summary      — unread count per store (dashboard widget)
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from bson import ObjectId
from datetime import datetime

from backend.app.database import get_db

router = APIRouter(prefix="/alerts", tags=["Alerts"])


def _serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.get("/")
async def list_alerts(
    store_id: Optional[str] = Query(None),
    status:   Optional[str] = Query(None, description="unread | read"),
    limit:    int            = Query(50, ge=1, le=200),
    skip:     int            = Query(0, ge=0),
):
    db    = get_db()
    query = {}
    if store_id:
        query["store_id"] = store_id
    if status:
        query["status"] = status

    docs  = await (
        db.alerts.find(query)
          .sort("triggered_at", -1)
          .skip(skip)
          .limit(limit)
          .to_list(limit)
    )
    total = await db.alerts.count_documents(query)
    return {"total": total, "alerts": [_serialize(d) for d in docs]}


@router.get("/summary")
async def alerts_summary():
    """Unread alert count grouped by store — for retailer dashboard."""
    db = get_db()
    pipeline = [
        {"$match": {"status": "unread"}},
        {"$group": {"_id": "$store_id", "unread_count": {"$sum": 1}}},
        {"$sort": {"unread_count": -1}},
    ]
    result = await db.alerts.aggregate(pipeline).to_list(100)
    return {"summary": [{"store_id": r["_id"], "unread_count": r["unread_count"]}
                        for r in result]}


@router.patch("/{alert_id}/read")
async def mark_alert_read(alert_id: str):
    db = get_db()
    try:
        oid = ObjectId(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert ID format")

    result = await db.alerts.update_one(
        {"_id": oid},
        {"$set": {"status": "read", "read_at": datetime.now().isoformat()}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return {"message": "Alert marked as read", "alert_id": alert_id}


@router.delete("/{alert_id}")
async def delete_alert(alert_id: str):
    db = get_db()
    try:
        oid = ObjectId(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert ID format")

    result = await db.alerts.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return {"message": "Alert deleted", "alert_id": alert_id}
