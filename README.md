# Stockout Predictor

**Context-aware stockout intelligence for quick-commerce, with one prediction delivered in two voices: one for the customer, one for the retailer.**

Most stockout systems stop at a number on a warehouse dashboard. This project goes further: it predicts the probability that a product runs out, reads the live situation around it (weather, festivals, payday and weekend patterns), and turns that into plain-language messages for the two people who act on it.

> Built with Python, XGBoost, LightGBM, FastAPI, MongoDB (time-series collections), Google Gemini and the OpenWeatherMap API.

<!-- TODO: add a Swagger UI screenshot or a short GIF of POST /predictions here -->

---

## One prediction, two audiences

The same stockout probability is framed differently depending on who needs to act.

**Customer app** (short, urgent, conversational):

> "Umbrellas near you have an 87% chance of running out by evening. Heavy rain is expected and demand has already spiked. Order now before stock runs out."

**Retailer / store manager** (data-forward, operational):

> "ALERT: SKU-4421 (Umbrellas), 87% stockout risk. Rain + weekend demand driving spike. Recommend immediate reorder of 200 units."

*These are illustrative examples of the output style, not captured API responses.*

---

## The problem

Quick-commerce runs on a 10 to 30 minute promise. A supermarket can forecast overnight and reorder tomorrow. A dark store cannot: if umbrellas run out at 4 pm on a rainy Friday, that revenue window is gone, and so is some of the customer's trust.

Two gaps follow from this:

1. **Retailers find out too late.** Inventory alone says what is on the shelf now, not what the next few hours of weather, festivals and pay-cycle behaviour will do to it.
2. **Customers are never told.** Stockout prediction is almost always inward-facing. Here it is also surfaced to the buyer as a real-time, data-backed signal, which sits at the intersection of inventory management and behavioural economics.

---

## How it works

```
M5 Sales Data (Kaggle)
        |
        v
Data Processing -> Feature Engineering
(lag, rolling, calendar, festival features)
        |
        v
XGBoost / LightGBM classifier
        |  predict_proba() -> stockout probability (e.g. 0.87)
        v
Context Aggregator
  |-- Day of week / day of month   (payday and weekend signals)
  |-- Indian festival calendar     (upcoming festivals + affected categories)
  `-- Live weather                 (OpenWeatherMap -> demand-impact mapping)
        |
        v
Google Gemini (turns probability + context into natural language)
        |
        v
Stored in MongoDB:
  |-- predictions  -> probability, customer_message, retailer_alert, context_snapshot
  `-- alerts       -> risk >= 70%, status: unread, for the retailer dashboard
```

### What "context-aware" means here

| Signal | Where it comes from | How it is used |
|--------|---------------------|----------------|
| Calendar and festival patterns | Engineered from the M5 calendar and the festival calendar | Part of the feature set the model is trained on |
| Payday / weekend / day-of-month | Context aggregator | Interpreted alongside the prediction and fed into the narrative |
| Live weather | OpenWeatherMap API at request time | Mapped to demand impact per product category and used to explain and escalate the prediction |
| Every one of the above | Saved with each prediction | Stored as a `context_snapshot`, so every alert can be audited later |

Historical weather is not part of the M5 training data, so weather acts at prediction time as live context rather than as a learned training feature. Making weather a learned feature is the first item under Future Work.

---

## Design decisions

- **Recall over precision.** Missing a real stockout costs more than a false alarm, so the classification threshold is tuned rather than left at 0.5. Stockouts are rare, so the tuning also handles heavy class imbalance.
- **A probability, not a yes/no.** The output is a percentage, so the same model can drive a soft nudge to a customer and a hard reorder alert to a manager.
- **Alert triage.** Only predictions at 70% risk or higher create a retailer alert, with read/unread status, so the dashboard shows what needs attention rather than everything.
- **Context stored with every prediction.** Each record keeps the weather, festival and temporal signals that produced it.
- **Two models, compared.** XGBoost and LightGBM are both trained and evaluated in `scripts/train_model.py`.

---

## Results

| Metric | XGBoost |
|--------|---------|
| F1 score | 0.9087 |
| ROC-AUC | 0.9988 |
| Recall | 1.0 |

**Dataset scale:** 500 products and 970,500 inventory rows seeded into MongoDB.

**How to read these numbers:** M5 contains sales but no stock levels, so inventory is simulated (see below). Because the simulated stock is derived from the same sales data, a model can partly learn the simulator's own rules, which is why the scores are so high. Treat them as proof that the pipeline works end to end, not as production accuracy. Validating on real inventory data is the most important next step.

---

## Why MongoDB

The data is document-shaped and does not fit cleanly into relational tables:

- **Variable context.** Each prediction carries a different mix of weather, festival and temporal signals.
- **Different schemas per document type.** Alerts and predictions look different.
- **Different drivers per category.** An umbrella and a packet of milk respond to different signals.
- **High write frequency.** Inventory snapshots arrive constantly and suit MongoDB's native time-series collections.

| Collection | Type | Purpose |
|------------|------|---------|
| `products` | Document | Product catalog |
| `sales_snapshots` / inventory | Time-series | High-frequency inventory and sales snapshots |
| `predictions` | Document | Probability, customer message, retailer alert, context snapshot |
| `alerts` | Document | High-risk alerts (>= 70%) with read/unread status |

---

## A note on synthetic inventory data

`scripts/generate_inventory.py` creates **simulated** inventory snapshots. The M5 Forecasting dataset (Walmart US sales) has daily sales quantities but no real stock levels, and no public quick-commerce inventory dataset exists. This is an acknowledged limitation.

In production this layer would be replaced by live POS or warehouse-management feeds from the dark store. The simulation lets the rest of the architecture be demonstrated end to end.

---

## Project structure

```
stockout-predictor/
|-- backend/                          # FastAPI application
|   `-- app/
|       |-- database.py               # Async MongoDB (Motor) with time-series setup
|       |-- main.py                   # FastAPI entry point
|       |-- models/                   # Pydantic schemas
|       |-- routes/                   # API endpoints
|       `-- services/
|           |-- message_generator.py  # Context aggregator + Gemini narrator
|           `-- predictor.py          # Model loader + predict_proba
|-- data/
|   |-- raw/                          # Download M5 files here
|   `-- processed/                    # Auto-generated by scripts
|-- notebooks/
|   `-- 01_exploration.ipynb          # EDA and model evaluation visualizations
|-- saved_models/                     # Trained models (auto-generated)
|-- scripts/
|   |-- process_data.py               # M5 -> long-format CSV
|   |-- generate_inventory.py         # Synthetic inventory simulation
|   |-- feature_engineering.py        # Lag, rolling, calendar, festival features
|   |-- train_model.py                # Train + evaluate XGBoost and LightGBM
|   `-- seed_mongo.py                 # Populate MongoDB collections
|-- .env.example
|-- requirements.txt
`-- README.md
```

---

## Getting started

### Prerequisites

- Python 3.10+
- MongoDB (local install or Atlas), on `localhost:27017` by default
- A free [Google Gemini API key](https://aistudio.google.com/)
- A free [OpenWeatherMap API key](https://openweathermap.org/api)
- The M5 files from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy): `sales_train_evaluation.csv` and `calendar.csv`, placed in `data/raw/`

> The M5 data and the trained models are **not** stored in this repository. The pipeline below regenerates everything.

### Setup (Windows PowerShell)

```powershell
# 1. Clone and create a virtual environment
git clone https://github.com/RishiRagavS/STOCKOUT-PREDICTOR.git
cd STOCKOUT-PREDICTOR
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env
# Open .env and fill in your keys (see below)

# 3. Make sure MongoDB is running (Windows service or mongod)

# 4. Place the M5 files in data/raw/, then run the pipeline in order
python scripts/process_data.py
python scripts/generate_inventory.py
python scripts/feature_engineering.py
python scripts/train_model.py
python scripts/seed_mongo.py

# 5. Start the API from the project root
uvicorn backend.app.main:app --reload
```

### Setup (macOS / Linux)

```bash
git clone https://github.com/RishiRagavS/STOCKOUT-PREDICTOR.git
cd STOCKOUT-PREDICTOR
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in the keys, then run the same pipeline scripts and the same uvicorn command
```

Once the server is up, open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

### Environment variables

API keys are deliberately **not** committed to this repository. Copy `.env.example` to `.env` and fill in:

```
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=stockout_predictor
GEMINI_API_KEY=your_gemini_key               # free key at aistudio.google.com
OPENWEATHER_API_KEY=your_openweathermap_key  # free tier works
CITY=Hyderabad                               # city used for the weather lookup
```

### Restarting later

```powershell
cd STOCKOUT-PREDICTOR
venv\Scripts\activate
uvicorn backend.app.main:app --reload
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/products` | List all products |
| GET | `/inventory/{sku_id}` | Inventory snapshots for a product |
| GET | `/predictions/{sku_id}` | Historical predictions for a product |
| POST | `/predictions` | Generate a new contextual prediction |
| GET | `/alerts` | All unread retailer alerts |
| PATCH | `/alerts/{alert_id}/read` | Mark an alert as read |

`POST /predictions` returns the stockout probability, the `customer_message`, the `retailer_alert`, and the context snapshot (weather, festival, temporal signals) that produced them. The Swagger UI at `/docs` shows the exact request and response schema.

---

## Tech stack

| Layer | Technology |
|-------|------------|
| ML | XGBoost, LightGBM, scikit-learn |
| Backend | FastAPI, Uvicorn |
| Database | MongoDB (Motor async driver), time-series collections |
| Context | OpenWeatherMap API, Indian festival calendar |
| Narration | Google Gemini API |
| Data | pandas, numpy |
| Evaluation | scikit-learn metrics, matplotlib, seaborn |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate the venv, then run `pip install -r requirements.txt` |
| MongoDB connection refused | Start the MongoDB service and check `MONGO_URI` in `.env` |
| `FileNotFoundError` on model or data | Run the pipeline scripts in order (step 4 above) |
| Gemini or weather errors | Check the API keys in `.env`, then restart the server |
| `uvicorn` can't find the app | Run it from the project root with `backend.app.main:app` |

---

## Limitations

- **Inventory is simulated**, so the reported metrics demonstrate the pipeline, not real-world accuracy.
- **Weather is live context, not a learned feature**, because the training data has no historical weather.
- **One city.** Weather is looked up for a single configured city; a real system would map each dark store to its own location.
- **Festival demand mapping is rule-based** rather than learned from historical festival sales.
- **Customer urgency messaging needs care.** A stated probability is only fair if it is well calibrated, and "order now" nudges can create the very demand spike they predict.

## Future work

- Train on historical weather so it becomes a learned feature, then compare the model with and without it.
- Calibrate predicted probabilities, so "87%" means 87% in practice.
- Add a time-to-zero forecast, so a prediction says both how likely and how soon.
- Evaluate against a simple rule-based baseline (stock vs. average daily sales) with a time-based train/test split.
- Retailer dashboard UI and automatic reorder suggestions.
- A/B testing of customer message wording, and scheduled batch predictions.

---

## License

MIT
