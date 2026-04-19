"""Train the Iris classifier and save as pickle."""

import pickle
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train():
    print("[train] Loading Iris dataset...")
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[train] Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"[train] Test accuracy: {accuracy:.4f}")

    os.makedirs("/app/models", exist_ok=True)
    model_path = "/app/models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[train] Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    train()
