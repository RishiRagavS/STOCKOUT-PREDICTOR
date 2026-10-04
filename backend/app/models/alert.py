from pydantic import BaseModel
from typing import Optional, Dict, Any


class AlertResponse(BaseModel):
    id:               str
    sku_id:           str
    store_id:         str
    probability_pct:  int
    retailer_alert:   str
    context_snapshot: Dict[str, Any]
    triggered_at:     str
    status:           str  # "unread" | "read"


class AlertStatusUpdate(BaseModel):
    status: str  # "read"
