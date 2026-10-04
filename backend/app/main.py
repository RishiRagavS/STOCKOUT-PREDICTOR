"""
main.py
-------
FastAPI application entry point.
Registers all routers, handles startup/shutdown lifecycle.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.database import connect_db, close_db
from backend.app.services import predictor
from backend.app.routes import products, predictions, alerts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_db()
    try:
        predictor.load_model()
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] Predictions will fail until you run scripts/train_model.py")
    yield
    # Shutdown
    await close_db()


app = FastAPI(
    title="Stockout Predictor API",
    description=(
        "Context-aware stockout prediction for quick-commerce. "
        "Predicts which products are at risk of stocking out and explains why "
        "using live weather, festival calendar, and temporal demand signals. "
        "Serves two audiences: customers (probability nudge) and retailers (ops alert)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(products.router)
app.include_router(predictions.router)
app.include_router(alerts.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Stockout Predictor",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    from backend.app.database import get_db
    db = get_db()
    try:
        await db.command("ping")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    model_loaded = predictor._model is not None
    return {
        "database": db_status,
        "model_loaded": model_loaded,
        "model_threshold": predictor.get_threshold() if model_loaded else None,
    }
