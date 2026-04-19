"""Train the sentiment analyzer and save as pickle."""

import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

REVIEWS = [
    ("This movie was absolutely wonderful and amazing", 1),
    ("Great film with excellent acting and storyline", 1),
    ("I loved every minute of this beautiful movie", 1),
    ("Outstanding performance by the entire cast", 1),
    ("A masterpiece of modern cinema", 1),
    ("Brilliant writing and superb direction", 1),
    ("One of the best films I have ever seen", 1),
    ("Highly recommended for everyone", 1),
    ("Fantastic movie with great special effects", 1),
    ("A truly enjoyable and entertaining experience", 1),
    ("The acting was top notch and very convincing", 1),
    ("Beautiful cinematography and wonderful score", 1),
    ("I was thoroughly impressed by this film", 1),
    ("An incredible journey from start to finish", 1),
    ("Perfect blend of humor and drama", 1),
    ("This movie was terrible and boring", 0),
    ("Worst film I have ever watched", 0),
    ("Complete waste of time and money", 0),
    ("The acting was awful and unconvincing", 0),
    ("I hated this movie so much", 0),
    ("Terrible plot with no real story", 0),
    ("Extremely disappointing and dull", 0),
    ("A horrible movie that makes no sense", 0),
    ("Bad acting and worse direction", 0),
    ("I would not recommend this to anyone", 0),
    ("The worst movie of the year by far", 0),
    ("Painfully slow and incredibly boring", 0),
    ("A complete disaster from beginning to end", 0),
    ("Absolutely dreadful and unwatchable", 0),
    ("Nothing good about this film at all", 0),
]


def train():
    print("[train] Preparing sentiment dataset...")
    texts = [r[0] for r in REVIEWS]
    labels = [r[1] for r in REVIEWS]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print("[train] Training TF-IDF + LogisticRegression pipeline...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    print(f"[train] Test accuracy: {accuracy:.4f}")

    os.makedirs("/app/models", exist_ok=True)
    model_path = "/app/models/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)

    print(f"[train] Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    train()
