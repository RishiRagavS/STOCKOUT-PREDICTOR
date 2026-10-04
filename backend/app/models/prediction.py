from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    sku_id:   str = Field(..., example="FOODS_1_001_CA_1_CA_1")
    store_id: str = Field(..., example="CA_1")
    city:     str = Field("Hyderabad", example="Hyderabad")


class WeatherContext(BaseModel):
    condition:           str
    temp_c:              Optional[float]
    high_demand_products: list[str]
    boosts_qcommerce:    bool
    reason:              str


class FestivalContext(BaseModel):
    festival:             Optional[str]
    days_until:           Optional[int]
    affected_categories:  list[str]


class TemporalContext(BaseModel):
    day_of_week:       str
    is_weekend:        bool
    is_payday_period:  bool
    is_mid_month_slump:bool
    day_of_month:      int


class ContextSnapshot(BaseModel):
    weather:  WeatherContext
    festival: FestivalContext
    temporal: TemporalContext


class PredictionResponse(BaseModel):
    sku_id:           str
    store_id:         str
    probability:      float
    probability_pct:  int
    customer_message: str
    retailer_alert:   str
    context_snapshot: Dict[str, Any]
    generated_at:     str
    alert_triggered:  bool
