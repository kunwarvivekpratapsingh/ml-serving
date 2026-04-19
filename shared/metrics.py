"""Prometheus metrics shared by all model containers."""

import os
from prometheus_client import Counter, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["model_name", "model_version", "status"],
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction requests",
    ["model_name"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Total number of failed predictions",
    ["model_name", "error_type"],
)

MODEL_INFO = Info(
    "model_deployment",
    "Metadata about the currently deployed model",
)


def init_metrics():
    """Initialize model info metric from environment variables."""
    MODEL_INFO.info(
        {
            "model_name": os.environ.get("MODEL_NAME", "unknown"),
            "model_version": os.environ.get("MODEL_VERSION", "0.0"),
            "model_type": os.environ.get("MODEL_TYPE", "unknown"),
            "scientist": os.environ.get("SCIENTIST_NAME", "unknown"),
        }
    )


def get_metrics():
    """Return Prometheus metrics as bytes."""
    return generate_latest(), CONTENT_TYPE_LATEST
