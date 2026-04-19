"""Flask API for Iris classifier."""

import os
import pickle
from flask import Flask, jsonify
from shared.metrics import init_metrics, get_metrics
from shared.middleware import instrument_predict

app = Flask(__name__)

MODEL_PATH = "/app/models/model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

SPECIES = ["setosa", "versicolor", "virginica"]

init_metrics()


@app.route("/predict", methods=["POST"])
@instrument_predict
def predict(input_data):
    """Predict iris species from 4 features."""
    features = input_data["features"]
    prediction = model.predict([features])
    probabilities = model.predict_proba([features])[0]

    return {
        "species": SPECIES[int(prediction[0])],
        "class": int(prediction[0]),
        "probabilities": {
            SPECIES[i]: round(float(p), 4) for i, p in enumerate(probabilities)
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
    port = int(os.environ.get("PORT", 8001))
    app.run(host="0.0.0.0", port=port)
