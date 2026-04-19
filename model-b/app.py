"""Flask API for house price predictor."""

import os
import pickle
from flask import Flask, jsonify
from shared.metrics import init_metrics, get_metrics
from shared.middleware import instrument_predict

app = Flask(__name__)

MODEL_PATH = "/app/models/model.pkl"
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_names = bundle["feature_names"]

init_metrics()


@app.route("/predict", methods=["POST"])
@instrument_predict
def predict(input_data):
    """Predict house price from 8 features."""
    features = input_data["features"]

    if len(features) != 8:
        raise ValueError(f"Expected 8 features, got {len(features)}. Required: {feature_names}")

    prediction = model.predict([features])

    return {
        "predicted_price": round(float(prediction[0]), 4),
        "unit": "100k_usd",
        "features_used": {name: val for name, val in zip(feature_names, features)},
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": os.environ.get("MODEL_NAME"),
        "version": os.environ.get("MODEL_VERSION"),
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    data, content_type = get_metrics()
    return data, 200, {"Content-Type": content_type}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))
    app.run(host="0.0.0.0", port=port)
