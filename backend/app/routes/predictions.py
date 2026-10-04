"""
predictions.py
--------------
POST /predictions  — run a full contextual prediction for a given SKU
GET  /predictions/{sku_id} — fetch prediction history for a SKU
"""

from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import Optional

from backend.app.database import get_db
from backend.app.models.prediction import PredictionRequest, PredictionResponse
from backend.app.services import predictor, message_generator

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/", response_model=PredictionResponse)
async def create_prediction(req: PredictionRequest):
    db = get_db()

    # 1. Fetch product
    product = await db.products.find_one({"sku_id": req.sku_id})
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {req.sku_id} not found")

    # 2. Fetch latest feature vector
    feat_doc = await db.features.find_one({"sku_id": req.sku_id})
    if not feat_doc:
        raise HTTPException(
            status_code=404,
            detail=f"No feature data for {req.sku_id}. Run seed_mongo.py."
        )

    # 3. Run model — predict_proba
    try:
        probability = predictor.predict_proba(feat_doc)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 4. Generate contextual message via Claude + weather + festival + temporal
    product_name = product.get("item_id", req.sku_id)
    result = await message_generator.generate_prediction_message(
        sku_id=req.sku_id,
        product_name=product_name,
        store_id=req.store_id,
        probability=probability,
        city=req.city,
    )

    # 5. Persist prediction to MongoDB
    await db.predictions.insert_one({**result})

    # 6. If probability >= threshold → write retailer alert
    if result["alert_triggered"]:
        await db.alerts.insert_one({
            "sku_id":           req.sku_id,
            "store_id":         req.store_id,
            "probability_pct":  result["probability_pct"],
            "retailer_alert":   result["retailer_alert"],
            "context_snapshot": result["context_snapshot"],
            "triggered_at":     datetime.now().isoformat(),
            "status":           "unread",
        })

    return PredictionResponse(**result)


@router.get("/{sku_id}")
async def get_predictions(
    sku_id: str,
    limit:  int           = Query(10, ge=1, le=100),
    store_id: Optional[str] = Query(None),
):
    db    = get_db()
    query = {"sku_id": sku_id}
    if store_id:
        query["store_id"] = store_id

    docs = await (
        db.predictions
          .find(query, {"_id": 0})
          .sort("generated_at", -1)
          .limit(limit)
          .to_list(limit)
    )
    if not docs:
        raise HTTPException(status_code=404,
                            detail=f"No predictions found for {sku_id}")
    return {"sku_id": sku_id, "predictions": docs}
