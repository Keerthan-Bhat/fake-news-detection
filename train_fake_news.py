# train_fake_news.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from joblib import dump
import re
import os

DATA_PATH = r"C:\Users\User\Documents\fake-news\data\fake_and_real_news.csv"           # your dataset
MODEL_PATH = "models/fake_news_model.joblib"


def ensure_dirs():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)         # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)           # keep only letters & spaces
    text = re.sub(r"\s+", " ", text).strip()           # collapse spaces
    return text


def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Your exact columns:
    text_col = "Text"
    label_col = "label"

    # Drop rows with missing text/label
    df = df.dropna(subset=[text_col, label_col])

    # Clean text
    df["clean_text"] = df[text_col].astype(str).apply(basic_clean)

    print("Unique labels in dataset:", df[label_col].unique())

    # Map labels to 0/1
    # 0 -> Real, 1 -> Fake (you can flip if you want)
    def map_label(val):
        v = str(val).strip().lower()
        if v == "fake":
            return 1
        elif v == "real":
            return 0
        else:
            raise ValueError(f"Unknown label value: {val}")

    df["label_num"] = df[label_col].apply(map_label)
    return df


def train_baseline(df: pd.DataFrame):
    X = df["clean_text"].values
    y = df["label_num"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF + Logistic Regression pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="liblinear"
        ))
    ])

    print("🚀 Training model...")
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc:.4f}\n")
    print("📊 Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    # Save model
    ensure_dirs()
    dump(pipe, MODEL_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")


def main():
    print("📥 Loading data...")
    df = load_and_prepare_data(DATA_PATH)
    print(f"Loaded {len(df)} samples.")
    train_baseline(df)


if __name__ == "__main__":
    main()
