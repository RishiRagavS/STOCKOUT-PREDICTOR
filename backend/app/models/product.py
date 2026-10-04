from pydantic import BaseModel
from typing import Optional


class ProductResponse(BaseModel):
    sku_id:    str
    item_id:   str
    store_id:  str
    category:  str
