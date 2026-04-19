"""Flask API for sentiment analyzer."""

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
vectorizer = bundle["vectorizer"]

LABELS = ["negative", "positive"]

init_metrics()


@app.route("/predict", methods=["POST"])
@instrument_predict
def predict(input_data):
    """Predict sentiment from text input."""
    text = input_data["text"]

    if not text or not isinstance(text, str):
        raise ValueError("Field 'text' must be a non-empty string")

    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)
    probabilities = model.predict_proba(vectorized)[0]

    return {
        "sentiment": LABELS[int(prediction[0])],
        "confidence": round(float(max(probabilities)), 4),
        "probabilities": {
            LABELS[i]: round(float(p), 4) for i, p in enumerate(probabilities)
        },
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
    port = int(os.environ.get("PORT", 8003))
    app.run(host="0.0.0.0", port=port)
