import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load
import re
from sklearn.metrics.pairwise import cosine_similarity
from newsapi import NewsApiClient

# ---------------- CONFIG ----------------
MODEL_PATH = "models/fake_news_model.joblib"
DATA_PATH = "data/fake_and_real_news.csv"

# ✅ ADD YOUR API KEY HERE LATER
NEWS_API_KEY = "YOUR_API_KEY"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return load(MODEL_PATH)

pipeline = load_model()

# ---------------- CLEANING ----------------
def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- PREDICTION ----------------
def predict(text):
    clean_text = basic_clean(text)
    proba = pipeline.predict_proba([clean_text])[0]
    pred = pipeline.predict([clean_text])[0]
    label = "Fake" if pred == 1 else "Real"
    confidence = proba[pred] * 100
    return label, confidence, clean_text

# ---------------- SHAP EXPLAINABILITY ----------------
def shap_explain(text):
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X = tfidf.transform([text])
    explainer = shap.LinearExplainer(clf, tfidf.transform(["sample"]), feature_names=tfidf.get_feature_names_out())

    shap_values = explainer(X)

    plt.figure()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(plt)

# ---------------- LOAD REAL NEWS FOR SIMILARITY ----------------
df = pd.read_csv(DATA_PATH)
real_df = df[df["label"].str.lower() == "real"].copy()
real_df["clean_text"] = real_df["Text"].apply(basic_clean)

def suggest_similar(clean_text, top_k=3):
    tfidf = pipeline.named_steps["tfidf"]
    query_vec = tfidf.transform([clean_text])
    real_vecs = tfidf.transform(real_df["clean_text"].values)

    sims = cosine_similarity(query_vec, real_vecs)[0]
    top_idx = sims.argsort()[-top_k:][::-1]

    results = []
    for idx in top_idx:
        results.append(real_df.iloc[idx]["Text"][:250] + "...")
    return results

# ---------------- STREAMLIT UI ----------------
st.title("📰 Fake News Verification Assistant")
st.write("Enter text OR fetch live news using API.")

mode = st.radio("Choose Input Mode:", ["Paste Text", "Live News API"])

if mode == "Paste Text":
    user_text = st.text_area("Paste News Article Text")

    if st.button("Verify"):
        label, confidence, clean_text = predict(user_text)

        st.subheader(f"🧾 Verdict: {label}")
        st.subheader(f"📈 Confidence: {confidence:.2f}%")

        st.subheader("🔎 Similar Verified Real News:")
        sims = suggest_similar(clean_text)
        for s in sims:
            st.write("-", s)

        st.subheader("📊 SHAP Explainability")
        shap_explain(clean_text)

# ---------------- LIVE NEWS API ----------------
else:
    topic = st.text_input("Enter topic (e.g., economy, india, tech)")

    if st.button("Fetch Live News"):
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(q=topic, language='en', page_size=5)

        for art in articles["articles"]:
            st.write("###", art["title"])
            st.write(art["description"])

            if st.button(f"Verify - {art['title']}"):
                label, confidence, clean_text = predict(art["description"])

                st.subheader(f"🧾 Verdict: {label}")
                st.subheader(f"📈 Confidence: {confidence:.2f}%")
                shap_explain(clean_text)
