"""
predictor.py
------------
Loads the trained model from disk and exposes predict_proba().
Handles feature preparation from a MongoDB features document.
"""

import os, json
import numpy as np
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "saved_models")

_model         = None
_feature_names = None
_threshold     = 0.5


def load_model():
    global _model, _feature_names, _threshold

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    feat_path  = os.path.join(MODELS_DIR, "feature_names.txt")
    meta_path  = os.path.join(MODELS_DIR, "best_model_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run scripts/train_model.py first."
        )

    _model = joblib.load(model_path)

    with open(feat_path) as f:
        _feature_names = [l.strip() for l in f.readlines() if l.strip()]

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        _threshold = meta.get("threshold", 0.5)

    print(f"[Predictor] Loaded model | features={len(_feature_names)} | threshold={_threshold}")


def prepare_features(feature_doc: dict) -> np.ndarray:
    """Convert a MongoDB features document into the numpy array the model expects."""
    if _feature_names is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    row = [float(feature_doc.get(f, 0) or 0) for f in _feature_names]
    return np.array(row).reshape(1, -1)


def predict_proba(feature_doc: dict) -> float:
    """Return stockout probability (float 0-1) for a feature document."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    X = prepare_features(feature_doc)
    prob = float(_model.predict_proba(X)[0][1])
    return prob


def get_threshold() -> float:
    return _threshold
