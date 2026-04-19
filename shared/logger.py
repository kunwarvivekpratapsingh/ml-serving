"""Prediction logger — writes every prediction to SQLite for audit and drift detection."""

import json
import os
import sqlite3
import time
import threading

DB_PATH = "/data/logs/predictions.db"
_local = threading.local()


def _get_conn():
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn"):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                model_name TEXT,
                model_version TEXT,
                scientist TEXT,
                input_data TEXT,
                output_data TEXT,
                latency_ms REAL,
                status TEXT
            )"""
        )
        _local.conn.commit()
    return _local.conn


def log_prediction(model_name, model_version, scientist, input_data, output_data, latency_s, status="success"):
    """Log a single prediction to the database."""
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO predictions
               (timestamp, model_name, model_version, scientist, input_data, output_data, latency_ms, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                model_name,
                model_version,
                scientist,
                json.dumps(input_data),
                json.dumps(output_data),
                round(latency_s * 1000, 2),
                status,
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"[logger] Failed to log prediction: {e}")
