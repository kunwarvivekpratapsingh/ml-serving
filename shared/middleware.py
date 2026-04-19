"""Flask middleware that instruments every /predict call with metrics and logging."""

import os
import time
import traceback
from functools import wraps

from flask import jsonify, request

from shared.metrics import REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT
from shared.logger import log_prediction


def instrument_predict(predict_fn):
    """Decorator that wraps a predict function with metrics and logging."""
    model_name = os.environ.get("MODEL_NAME", "unknown")
    model_version = os.environ.get("MODEL_VERSION", "0.0")
    scientist = os.environ.get("SCIENTIST_NAME", "unknown")

    @wraps(predict_fn)
    def wrapper():
        start = time.perf_counter()
        input_data = request.get_json(force=True)

        try:
            result = predict_fn(input_data)
            latency = time.perf_counter() - start

            REQUEST_COUNT.labels(
                model_name=model_name,
                model_version=model_version,
                status="success",
            ).inc()
            REQUEST_LATENCY.labels(model_name=model_name).observe(latency)

            response = {
                "model": model_name,
                "version": model_version,
                "prediction": result,
                "latency_ms": round(latency * 1000, 2),
            }

            log_prediction(model_name, model_version, scientist, input_data, result, latency)

            return jsonify(response)

        except Exception as e:
            latency = time.perf_counter() - start
            error_type = type(e).__name__

            REQUEST_COUNT.labels(
                model_name=model_name,
                model_version=model_version,
                status="error",
            ).inc()
            ERROR_COUNT.labels(model_name=model_name, error_type=error_type).inc()

            log_prediction(
                model_name, model_version, scientist, input_data,
                {"error": str(e)}, latency, status="error",
            )

            return jsonify({"error": str(e), "type": error_type}), 500

    return wrapper
