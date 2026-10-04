"""
message_generator.py
--------------------
Context-aware prediction narrative generator using Google Gemini API (free tier).

Flow:
  1. Fetch live weather from OpenWeatherMap
  2. Check Indian festival calendar (±3 days)
  3. Extract temporal demand signals (day-of-week, payday, etc.)
  4. Send all signals + probability to Gemini API
  5. Gemini returns two messages:
       - customer_message : shown on quick-commerce app to the shopper
       - retailer_alert   : shown to the warehouse/store manager

Falls back to rule-based messages if API key is missing or call fails.
"""

import os
import json
import httpx
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Gemini setup ─────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_key_here":
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False

# ── Indian Festival Calendar ──────────────────────────────────────────────────
FESTIVALS = [
    (1, 14,  "Pongal / Makar Sankranti",
             ["rice", "sugarcane", "sweets", "groceries", "household"]),
    (1, 26,  "Republic Day",
             ["snacks", "beverages", "cold drinks"]),
    (3, 20,  "Holi",
             ["colors", "sweets", "beverages", "snacks", "dairy"]),
    (4, 14,  "Tamil New Year / Ugadi",
             ["groceries", "sweets", "fruits", "clothing accessories"]),
    (8, 15,  "Independence Day",
             ["beverages", "snacks", "ice cream", "ready-to-eat"]),
    (8, 28,  "Onam",
             ["rice", "vegetables", "sweets", "payasam ingredients"]),
    (10, 2,  "Navratri / Dussehra",
             ["sweets", "fruits", "household", "puja items"]),
    (10, 20, "Diwali",
             ["sweets", "dry fruits", "snacks", "diyas", "household",
              "packaged foods", "beverages", "gifting"]),
    (11, 14, "Children's Day",
             ["chocolates", "snacks", "juices", "biscuits"]),
    (12, 24, "Christmas Eve",
             ["beverages", "bakery", "cakes", "snacks", "chocolates"]),
    (12, 31, "New Year's Eve",
             ["beverages", "snacks", "packaged foods", "ice cream"]),
]

# ── Weather → Demand Impact Map ───────────────────────────────────────────────
WEATHER_IMPACT = {
    "Rain": {
        "high_demand": ["umbrellas", "raincoats", "hot beverages",
                        "instant noodles", "ready-to-eat meals", "bread", "eggs"],
        "boosts_qcommerce": True,
        "reason": "people avoid stepping out in rain",
    },
    "Thunderstorm": {
        "high_demand": ["candles", "torch", "instant noodles", "snacks",
                        "power banks", "packaged water"],
        "boosts_qcommerce": True,
        "reason": "people stay strictly indoors during storms",
    },
    "Drizzle": {
        "high_demand": ["hot beverages", "snacks", "ready-to-eat"],
        "boosts_qcommerce": True,
        "reason": "light rain nudges people to order in",
    },
    "Clear": {
        "high_demand": ["cold drinks", "ice cream", "sunscreen", "juices"],
        "boosts_qcommerce": False,
        "reason": "clear weather means people prefer going out",
    },
    "Clouds": {
        "high_demand": ["hot beverages", "snacks"],
        "boosts_qcommerce": False,
        "reason": "mild weather has neutral impact on quick-commerce",
    },
    "Haze": {
        "high_demand": ["face masks", "air purifier refills", "eye drops", "water"],
        "boosts_qcommerce": True,
        "reason": "poor air quality keeps people indoors",
    },
    "Mist": {
        "high_demand": ["hot beverages", "soups"],
        "boosts_qcommerce": False,
        "reason": "mist has mild effect on outdoor activity",
    },
    "Snow": {
        "high_demand": ["hot beverages", "warm food", "soups", "blankets"],
        "boosts_qcommerce": True,
        "reason": "cold weather keeps people indoors",
    },
}

PRODUCT_CATEGORY_MAP = {
    "FOODS":     "food and grocery items",
    "HOBBIES":   "hobbies and leisure products",
    "HOUSEHOLD": "household essentials",
}


async def get_weather(city: str) -> dict:
    """Fetch current weather from OpenWeatherMap free tier."""
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key or api_key == "your_openweathermap_api_key_here":
        return {
            "condition": "Unknown",
            "temp_c": None,
            "high_demand_products": [],
            "boosts_qcommerce": False,
            "reason": "weather API key not configured",
        }
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?q={city}&appid={api_key}&units=metric")
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get(url)
            resp.raise_for_status()
            data = resp.json()
        condition = data["weather"][0]["main"]
        temp      = data["main"]["temp"]
        impact    = WEATHER_IMPACT.get(condition, {})
        return {
            "condition":            condition,
            "temp_c":               round(temp, 1),
            "high_demand_products": impact.get("high_demand", []),
            "boosts_qcommerce":     impact.get("boosts_qcommerce", False),
            "reason":               impact.get("reason", ""),
        }
    except Exception as e:
        return {
            "condition": "Unknown",
            "temp_c": None,
            "high_demand_products": [],
            "boosts_qcommerce": False,
            "reason": f"weather fetch failed: {e}",
        }


def get_festival_context(date: datetime) -> dict:
    """Return the nearest festival within ±3 days of the given date."""
    for days_ahead in range(-1, 4):
        check_month = date.month
        check_day   = date.day + days_ahead
        for month, day, name, categories in FESTIVALS:
            if month == check_month and day == check_day:
                return {
                    "festival":            name,
                    "days_until":          days_ahead,
                    "affected_categories": categories,
                }
    return {"festival": None, "days_until": None, "affected_categories": []}


def get_temporal_context(date: datetime) -> dict:
    """Extract demand-relevant temporal signals."""
    dow = date.weekday()
    dom = date.day
    return {
        "day_of_week":         date.strftime("%A"),
        "is_weekend":          dow >= 5,
        "is_payday_period":    dom in [1, 2, 30, 31],
        "is_mid_month_slump":  14 <= dom <= 16,
        "day_of_month":        dom,
    }


def get_product_category(sku_id: str) -> str:
    for key, label in PRODUCT_CATEGORY_MAP.items():
        if sku_id.upper().startswith(key):
            return label
    return "general retail products"


def _build_prompt(product_name, sku_id, store_id, category, pct,
                  now, weather, festival, temporal, festival_relevant) -> str:
    return f"""You are an intelligent inventory analyst for a quick-commerce platform in India (like Blinkit or Zepto).

PREDICTION DATA:
- Product: {product_name} (SKU: {sku_id})
- Category: {category}
- Store: {store_id}
- Stockout Probability: {pct}%
- Current Time: {now.strftime("%I:%M %p, %A, %d %B %Y")}

CONTEXTUAL SIGNALS:
Day & Demand:
  - Day: {temporal['day_of_week']}, {temporal['day_of_month']} of the month
  - Weekend: {temporal['is_weekend']} | Payday period: {temporal['is_payday_period']} | Mid-month slump: {temporal['is_mid_month_slump']}

Weather:
  - Condition: {weather['condition']}{f", {weather['temp_c']}°C" if weather['temp_c'] else ""}
  - Boosts quick-commerce demand: {weather['boosts_qcommerce']}
  - Reason: {weather['reason']}
  - High-demand products due to weather: {', '.join(weather['high_demand_products']) if weather['high_demand_products'] else 'none'}

Festival:
  - Upcoming: {festival['festival'] if festival['festival'] else 'None in next 3 days'}
  - Days until: {festival['days_until'] if festival['days_until'] is not None else 'N/A'}
  - Festival-affected categories: {', '.join(festival['affected_categories']) if festival['affected_categories'] else 'none'}
  - Product category is festival-relevant: {festival_relevant}

YOUR TASK:
Generate exactly two messages:

1. CUSTOMER_MESSAGE: What the shopper sees on the app.
   - 1-2 sentences, conversational, friendly, not robotic
   - State the probability in plain language
   - Briefly explain WHY (pick the 1-2 most relevant signals)
   - Nudge to order now if probability > 60%
   - Be reassuring if probability < 40%
   - Write for an Indian urban consumer

2. RETAILER_ALERT: What the warehouse manager sees.
   - Data-forward, professional, 2-3 sentences max
   - Include probability, key demand drivers, recommended action
   - Urgency: LOW (<40%), MEDIUM (40-70%), HIGH (>70%), CRITICAL (>85%)

Respond ONLY in this exact JSON format, no extra text, no markdown:
{{"customer_message": "...", "retailer_alert": "..."}}"""


async def generate_prediction_message(
    sku_id:       str,
    product_name: str,
    store_id:     str,
    probability:  float,
    city:         str = "Hyderabad",
) -> dict:
    now      = datetime.now()
    weather  = await get_weather(city)
    festival = get_festival_context(now)
    temporal = get_temporal_context(now)
    category = get_product_category(sku_id)
    pct      = round(probability * 100)

    festival_relevant = False
    if festival["festival"] and festival["affected_categories"]:
        cat_lower = category.lower()
        festival_relevant = any(
            fc.lower() in cat_lower or cat_lower in fc.lower()
            for fc in festival["affected_categories"]
        )

    messages = None

    # ── Try Gemini ────────────────────────────────────────────────────────────
    if GEMINI_AVAILABLE:
        try:
            prompt   = _build_prompt(product_name, sku_id, store_id, category,
                                     pct, now, weather, festival, temporal,
                                     festival_relevant)
            response = gemini_model.generate_content(prompt)
            raw      = response.text.strip()
            raw      = raw.replace("```json", "").replace("```", "").strip()
            messages = json.loads(raw)
        except Exception as e:
            print(f"[MessageGen] Gemini call failed: {e} — using fallback")

    # ── Fallback: rule-based ──────────────────────────────────────────────────
    if not messages:
        messages = _fallback_message(product_name, pct, weather, festival, temporal)

    return {
        "sku_id":            sku_id,
        "store_id":          store_id,
        "probability":       round(probability, 4),
        "probability_pct":   pct,
        "customer_message":  messages.get("customer_message", ""),
        "retailer_alert":    messages.get("retailer_alert", ""),
        "context_snapshot": {
            "weather":  weather,
            "festival": festival,
            "temporal": temporal,
        },
        "generated_at":    now.isoformat(),
        "alert_triggered": probability >= 0.70,
        "city":            city,
    }


def _fallback_message(product_name, pct, weather, festival, temporal) -> dict:
    """Rule-based message when no LLM API is available."""
    if pct >= 85:
        urgency_label = "CRITICAL"
        cta = " Order immediately — stock is almost gone."
    elif pct >= 70:
        urgency_label = "HIGH"
        cta = " Order now to avoid missing out."
    elif pct >= 40:
        urgency_label = "MEDIUM"
        cta = " Consider ordering soon."
    else:
        urgency_label = "LOW"
        cta = ""

    drivers = []
    if temporal["is_weekend"]:
        drivers.append("weekend demand spike")
    if temporal["is_payday_period"]:
        drivers.append("payday spending period")
    if weather["boosts_qcommerce"] and weather["condition"] != "Unknown":
        drivers.append(f"{weather['condition'].lower()} keeping people indoors")
    if festival["festival"] and festival["days_until"] is not None and festival["days_until"] <= 2:
        drivers.append(f"{festival['festival']} approaching")

    driver_str = " and ".join(drivers[:2]) if drivers else "current demand patterns"

    customer_msg = (
        f"{product_name} has a {pct}% chance of stocking out today "
        f"due to {driver_str}.{cta}"
    )
    retailer_msg = (
        f"[{urgency_label}] {product_name} — {pct}% stockout probability. "
        f"Key drivers: {driver_str}. "
        f"Recommend immediate stock check and reorder if below safety threshold."
    )
    return {"customer_message": customer_msg, "retailer_alert": retailer_msg}
