"""Flask API for the purchase-intent model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "feature_metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model artifact not found. Run `python notebooks\\analysis.py` first.")

model = joblib.load(MODEL_PATH)
metrics_report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
FEATURES = metadata["features"]
FEATURE_SUMMARY = metadata["feature_summary"]

app = Flask(__name__)
CORS(app)


def _validate_payload(payload: object) -> tuple[dict[str, float] | None, list[str]]:
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]
    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        return None, [f"Missing required field: {feature}" for feature in missing]

    values: dict[str, float] = {}
    errors: list[str] = []
    for feature in FEATURES:
        value = payload[feature]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"{feature} must be a finite number.")
            continue
        bounds = FEATURE_SUMMARY[feature]
        if value < bounds["min"] or value > bounds["max"]:
            errors.append(f"{feature} must be between {bounds['min']} and {bounds['max']}.")
            continue
        values[feature] = float(value)
    return (values if not errors else None), errors


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "Logistic Regression", "features": FEATURES})


@app.get("/analytics")
def analytics():
    """Return model quality, data summary, and ranked feature effects."""
    return jsonify(
        {
            "metrics": metrics_report["metrics"],
            "confusion_matrix": metrics_report["confusion_matrix"],
            "roc_curve": metrics_report["roc_curve"],
            "feature_importance": metrics_report["feature_importance"],
            "dataset": metrics_report["dataset"],
            "training": metrics_report["training"],
        }
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    values, errors = _validate_payload(payload)
    if errors:
        return jsonify({"error": "Invalid input", "details": errors}), 400

    row = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return jsonify(
        {
            "prediction": "Purchase Likely" if prediction else "No Purchase Likely",
            "purchase_probability": round(probability, 4),
            "no_purchase_probability": round(1 - probability, 4),
            "prediction_code": prediction,
            "input": values,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
