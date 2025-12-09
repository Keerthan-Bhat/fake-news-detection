# news_verification_agent.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from explain_utils import get_top_words_for_text
import re

DATA_PATH = "data/news.csv"
MODEL_PATH = "models/fake_news_model.joblib"


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_url(url: str) -> str:
    """Very simple scraper to get text from a news page."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    article = soup.find("article")
    if article:
        text = article.get_text(separator=" ")
    else:
        paragraphs = [p.get_text(separator=" ") for p in soup.find_all("p")]
        text = " ".join(paragraphs)

    return text.strip()


def load_real_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    From your dataset (Text,label), build a DataFrame of only REAL articles
    for similarity suggestions.
    """
    label_col = "label"
    text_col = "Text"

    # Filter rows where label is Real (case-insensitive)
    real_df = df[df[label_col].astype(str).str.strip().str.lower() == "real"].copy()
    real_df["clean_text"] = real_df[text_col].astype(str).apply(basic_clean)
    return real_df


def suggest_similar_real_articles(pipeline, query_text: str, real_df: pd.DataFrame, top_k: int = 3):
    tfidf = pipeline.named_steps["tfidf"]

    query_vec = tfidf.transform([query_text])
    real_vecs = tfidf.transform(real_df["clean_text"].values)

    sims = cosine_similarity(query_vec, real_vecs)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    suggestions = []
    for idx in top_idx:
        row = real_df.iloc[idx]
        suggestions.append({
            "snippet": (row["Text"][:250] + "...") if isinstance(row["Text"], str) else "",
            "similarity": float(sims[idx])
        })
    return suggestions


def predict_article(pipeline, text: str):
    clean_text = basic_clean(text)
    proba = pipeline.predict_proba([clean_text])[0]
    pred = pipeline.predict([clean_text])[0]

    # Make sure this matches train mapping:
    # 0 -> Real, 1 -> Fake
    label = "Fake" if pred == 1 else "Real"
    confidence = proba[pred] * 100.0
    return label, confidence, clean_text


def main():
    print("🔍 Loading model...")
    pipeline = load(r"C:\Users\User\Documents\fake-news\models\fake_news_model.joblib")

    print("📥 Loading dataset for similarity search...")
    df = pd.read_csv(r"C:\Users\User\Documents\fake-news\data\fake_and_real_news.csv")
    real_df = load_real_articles(df)
    print(f"Using {len(real_df)} REAL articles for similarity suggestions.")

    while True:
        print("\n==============================")
        print(" News Verification Assistant")
        print("==============================")
        print("1. Paste article URL")
        print("2. Paste article text")
        print("3. Exit")
        choice = input("Choose an option (1/2/3): ").strip()

        if choice == "3":
            print("👋 Goodbye!")
            break

        if choice == "1":
            url = input("Paste news URL: ").strip()
            article_text = extract_text_from_url(url)
            if not article_text:
                print("❌ Could not extract text from URL.")
                continue
        elif choice == "2":
            print("Paste the article text (finish with an empty line):")
            lines = []
            while True:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            article_text = "\n".join(lines)
        else:
            print("Invalid choice.")
            continue

        if not article_text.strip():
            print("No text provided.")
            continue

        # 🔮 Prediction
        label, confidence, clean_text = predict_article(pipeline, article_text)
        print(f"\n🧾 Verdict: {label}")
        print(f"📈 Confidence: {confidence:.2f}%")

        # 🔍 Explainability
        print("\n📌 Top words contributing to prediction:")
        fake_words, real_words = get_top_words_for_text(pipeline, clean_text, top_k=8)

        print("\n  Words pushing towards FAKE:")
        if fake_words:
            for w, c in fake_words:
                print(f"   - {w:20s} ({c:+.4f})")
        else:
            print("   (None strong)")

        print("\n  Words pushing towards REAL:")
        if real_words:
            for w, c in real_words:
                print(f"   - {w:20s} ({c:+.4f})")
        else:
            print("   (None strong)")

        # 🔗 Similar verified (REAL) articles
        print("\n🔎 Similar verified (REAL) articles from dataset:")
        suggestions = suggest_similar_real_articles(pipeline, clean_text, real_df, top_k=3)
        for i, s in enumerate(suggestions, 1):
            print(f"\n  [{i}] similarity: {s['similarity']:.3f}")
            print(f"      snippet: {s['snippet']}")


if __name__ == "__main__":
    main()