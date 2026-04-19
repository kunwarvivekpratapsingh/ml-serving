"""Train the house price predictor and save as pickle."""

import pickle
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


def train():
    print("[train] Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[train] Training XGBRegressor...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    print(f"[train] Test RMSE: {rmse:.4f}, R2: {r2:.4f}")

    os.makedirs("/app/models", exist_ok=True)
    model_path = "/app/models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_names": list(data.feature_names)}, f)

    print(f"[train] Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    train()
