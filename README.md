# \# 🛒 Stockout Predictor: Context-Aware Inventory Intelligence for Quick-Commerce

# 

# A stockout prediction system for quick-commerce apps (Blinkit / Zepto style) that predicts which products are about to run out, explains \*\*why\*\* using live context, and delivers that prediction in two different voices to two different audiences.

# 

# \---

# 

# \## What This Project Does

# 

# The system takes one ML prediction and turns it into two messages:

# 

# \*\*Customer-facing message\*\* (shown on the app UI): a short, conversational nudge.

# 

# > "Umbrellas near you have an 87% chance of running out by evening. Heavy rain is expected and demand has already spiked. Order now before stock runs out."

# 

# \*\*Retailer / warehouse alert\*\* (shown to the store manager): a data-forward operations message.

# 

# > "ALERT: SKU-4421 (Umbrellas), 87% stockout risk. Rain + weekend demand driving spike. Recommend immediate reorder of 200 units."

# 

# Same prediction, two framings, two different decision-makers.

# 

# \*(Messages above are illustrative examples of the output style.)\*

# 

# \---

# 

# \## The Problem This Solves

# 

# Quick-commerce (10 to 30 minute delivery) runs on a much shorter timescale than traditional retail. A supermarket can run stockout predictions overnight and reorder tomorrow. A dark store cannot: if umbrellas run out at 4 pm on a rainy Friday, that revenue window is gone.

# 

# Most stockout systems are inward-facing only. They talk to inventory managers and never to customers. This project also surfaces the prediction to the end customer as a real-time nudge to purchase, which sits at the intersection of behavioral economics and inventory management.

# 

# \---

# 

# \## Results

# 

# | Metric | Value (XGBoost) |

# |--------|-----------------|

# | F1 score | 0.9087 |

# | ROC-AUC | 0.9988 |

# | Recall | 1.0 |

# 

# \- Classification threshold is \*\*tuned\*\* rather than left at the default 0.5, to handle heavy class imbalance (stockouts are rare).

# \- Recall is prioritized: missing a real stockout costs more than a false alarm.

# \- Both XGBoost and LightGBM are trained and compared in `scripts/train\_model.py`.

# \- Metrics are computed on \*\*synthetic inventory data\*\* (see the note below), so treat them as a demonstration of the pipeline, not production performance.

# 

# \*\*Dataset scale:\*\* 500 products, 970,500 inventory rows seeded into MongoDB.

# 

# \---

# 

# \## Why MongoDB (NoSQL)

# 

# The data is naturally document-shaped and does not fit cleanly into relational tables:

# 

# \- \*\*Variable context snapshots:\*\* each prediction carries a different mix of weather, festival, and temporal signals.

# \- \*\*Different schemas per document type:\*\* alert documents look different from prediction documents.

# \- \*\*Different feature sets per category:\*\* an umbrella and a packet of milk are driven by different signals.

# \- \*\*High write frequency:\*\* inventory snapshots arrive constantly and suit MongoDB's native \*\*time-series collections\*\*.

# 

# | Collection | Type | Purpose |

# |------------|------|---------|

# | `products` | Document | Product catalog |

# | `sales\_snapshots` / inventory | \*\*Time-series\*\* | High-frequency inventory and sales snapshots |

# | `predictions` | Document | Prediction + customer message + retailer alert + context snapshot |

# | `alerts` | Document | High-risk alerts (probability ≥ 70%) for the retailer dashboard, with read/unread status |

# 

# \---

# 

# \## Architecture

# 

# ```

# M5 Sales Data (Kaggle)

# &#x20;       ↓

# Data Processing → Feature Engineering

# &#x20;       ↓

# XGBoost / LightGBM Model

# &#x20;       ↓ predict\_proba() → stockout probability (e.g. 0.87)

# Context Aggregator

# &#x20; ├── Day of week / day of month (payday, weekend signals)

# &#x20; ├── Indian festival calendar (upcoming festivals + affected categories)

# &#x20; └── Live weather (OpenWeatherMap API → demand impact mapping)

# &#x20;       ↓

# Google Gemini (generates contextual narrative)

# &#x20;       ↓

# Two outputs stored in MongoDB:

# &#x20; ├── predictions collection → customer\_message + retailer\_alert + context\_snapshot

# &#x20; └── alerts collection (threshold ≥ 70%) → status: unread, for retailer dashboard

# ```

# 

# \---

# 

# \## Project Structure

# 

# ```

# stockout-predictor/

# ├── backend/                         # FastAPI application

# │   └── app/

# │       ├── database.py              # Async MongoDB (Motor) with time-series setup

# │       ├── main.py                  # FastAPI entry point

# │       ├── models/                  # Pydantic schemas

# │       ├── routes/                  # API endpoints

# │       └── services/

# │           ├── message\_generator.py # Context aggregator + Gemini narrator

# │           └── predictor.py         # Model loader + predict\_proba

# ├── data/

# │   ├── raw/                         # Download M5 files here (see Data Requirements)

# │   └── processed/                   # Auto-generated by scripts

# ├── notebooks/

# │   └── 01\_exploration.ipynb         # EDA and model evaluation visualizations

# ├── saved\_models/                    # Trained models (auto-generated)

# ├── scripts/

# │   ├── process\_data.py              # M5 → long format CSV

# │   ├── generate\_inventory.py        # Synthetic inventory simulation (see note)

# │   ├── feature\_engineering.py       # Lag, rolling, calendar, festival features

# │   ├── train\_model.py               # Train + evaluate XGBoost \& LightGBM

# │   └── seed\_mongo.py                # Populate MongoDB collections

# ├── .env.example

# ├── requirements.txt

# └── README.md

# ```

# 

# \---

# 

# \## A Note on Synthetic Inventory Data

# 

# `generate\_inventory.py` creates \*\*simulated\*\* inventory snapshots, because the M5 Forecasting dataset (Walmart US sales) contains sales quantities but no actual stock levels. This is an acknowledged limitation.

# 

# In production, this layer would be replaced by live POS inventory feeds from the dark store's warehouse management system. The synthetic data lets the rest of the architecture be demonstrated end to end.

# 

# \---

# 

# \## Data Requirements

# 

# Download from the \[M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition on Kaggle:

# 

# \- `sales\_train\_evaluation.csv`: daily sales for 30,490 products across 10 stores

# \- `calendar.csv`: dates, holidays, SNAP days, promotional events

# 

# Place both files in `data/raw/`.

# 

# > These files and the trained model (`saved\_models/`) are \*\*not\*\* stored in this repository. The pipeline below regenerates everything.

# 

# \---

# 

# \## Prerequisites

# 

# \- Python 3.10+

# \- MongoDB (local install or Atlas), running on `localhost:27017` by default

# \- A free \[Google Gemini API key](https://aistudio.google.com/) (Google AI Studio)

# \- A free \[OpenWeatherMap API key](https://openweathermap.org/api)

# \- The M5 dataset files from Kaggle (see above)

# 

# \---

# 

# \## Installation \& Setup

# 

# \*\*Windows (PowerShell)\*\*

# 

# ```powershell

# \# 1. Clone and create a virtual environment

# git clone https://github.com/RishiRagavS/STOCKOUT-PREDICTOR.git

# cd STOCKOUT-PREDICTOR

# python -m venv venv

# venv\\Scripts\\activate

# pip install -r requirements.txt

# 

# \# 2. Configure environment

# copy .env.example .env

# \# Open .env and fill in MONGO\_URI, GEMINI\_API\_KEY, OPENWEATHER\_API\_KEY

# 

# \# 3. Make sure MongoDB is running (Windows service or mongod)

# 

# \# 4. Place the M5 files in data/raw/, then run the pipeline in order

# python scripts/process\_data.py

# python scripts/generate\_inventory.py

# python scripts/feature\_engineering.py

# python scripts/train\_model.py

# python scripts/seed\_mongo.py

# 

# \# 5. Start the API from the project root

# uvicorn backend.app.main:app --reload

# ```

# 

# \*\*macOS / Linux\*\*

# 

# ```bash

# git clone https://github.com/RishiRagavS/STOCKOUT-PREDICTOR.git

# cd STOCKOUT-PREDICTOR

# python3 -m venv venv

# source venv/bin/activate

# pip install -r requirements.txt

# cp .env.example .env

# \# fill in the keys, then run the same pipeline scripts and the same uvicorn command

# ```

# 

# Once the server is up, open \*\*http://127.0.0.1:8000/docs\*\* for the interactive Swagger UI, where you can try every endpoint.

# 

# \### Restarting later

# 

# ```powershell

# cd STOCKOUT-PREDICTOR

# venv\\Scripts\\activate

# uvicorn backend.app.main:app --reload

# ```

# 

# \---

# 

# \## API Endpoints

# 

# | Method | Endpoint | Description |

# |--------|----------|-------------|

# | GET | `/products` | List all products |

# | GET | `/inventory/{sku\_id}` | Inventory snapshots for a product |

# | GET | `/predictions/{sku\_id}` | Historical predictions for a product |

# | POST | `/predictions` | Generate a new contextual prediction |

# | GET | `/alerts` | All unread retailer alerts |

# | PATCH | `/alerts/{alert\_id}/read` | Mark an alert as read |

# 

# `POST /predictions` returns the stockout probability, the `customer\_message`, the `retailer\_alert`, and the context snapshot (weather, festival, temporal signals) that produced them. Use the Swagger UI at `/docs` to see the exact request and response schema.

# 

# \---

# 

# \## Environment Variables

# 

# Copy `.env.example` to `.env` and fill in:

# 

# ```

# MONGO\_URI=mongodb://localhost:27017

# MONGO\_DB\_NAME=stockout\_predictor

# GEMINI\_API\_KEY=your\_gemini\_key               # free key at aistudio.google.com

# OPENWEATHER\_API\_KEY=your\_openweathermap\_key  # free tier works

# CITY=Hyderabad                               # city for weather lookup

# ```

# 

# \*\*Never commit your `.env` file.\*\* It is already listed in `.gitignore`.

# 

# \---

# 

# \## Technologies

# 

# | Layer | Technology |

# |-------|------------|

# | ML | XGBoost, LightGBM, scikit-learn |

# | Backend | FastAPI, Uvicorn |

# | Database | MongoDB (Motor async driver), time-series collections |

# | Context | OpenWeatherMap API, Indian festival calendar |

# | Narration | Google Gemini API |

# | Data | pandas, numpy |

# | Evaluation | scikit-learn metrics, matplotlib, seaborn |

# 

# \---

# 

# \## Troubleshooting

# 

# | Problem | Fix |

# |---------|-----|

# | `ModuleNotFoundError` | Activate the venv, then run `pip install -r requirements.txt` |

# | MongoDB connection refused | Start the MongoDB service, and check `MONGO\_URI` in `.env` |

# | `FileNotFoundError` on model or data | Run the pipeline scripts in order (step 4 above) |

# | Gemini or weather errors | Check the API keys in `.env`, then restart the server |

# | `uvicorn` can't find the app | Run it from the \*\*project root\*\* with `backend.app.main:app` |

# 

# \---

# 

# \## Limitations \& Future Work

# 

# \- Inventory levels are simulated; a real deployment needs live warehouse feeds.

# \- Weather is looked up for a single configured city; a real system would map each dark store to its own location.

# \- Festival demand mapping is rule-based and could be learned from historical festival sales.

# \- Possible next steps: a retailer dashboard UI, automatic reorder suggestions, A/B testing of customer message wording, and scheduled batch predictions.

# 

# \---

# 

# \## License

# 

# MIT

