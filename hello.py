# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import yaml
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import networkx as nx
import io
import os
from datetime import datetime
from collections import Counter

# ============ SETTINGS ============
st.set_page_config(page_title="AI Threat Scanner", layout="wide")
st.markdown(
    """
    <style>
    body {background-color: #0e1117; color: #fafafa;}
    .stTextInput > div > div > input {background-color: #1e222a; color: #fafafa;}
    .stTextArea textarea {background-color: #1e222a; color: #fafafa;}
    .stButton button {background-color: #6c63ff; color: white; border-radius: 8px; font-weight: bold;}
    .stDataFrame {background-color: #0e1117; color: #fafafa;}
    .stMarkdown {color: #fafafa;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============ FILE STORAGE ============
KEYWORD_FILE = "keywords.yaml"
DEFAULT_KEYWORDS = [
    {"term": "attack", "type": "phrase", "language": "en", "weight": 3},
    {"term": "exploit", "type": "phrase", "language": "en", "weight": 3},
    {"term": "phishing", "type": "phrase", "language": "en", "weight": 3},
    {"term": "malware", "type": "phrase", "language": "en", "weight": 3},
    {"term": "hack", "type": "phrase", "language": "en", "weight": 3},
    {"term": "breach", "type": "phrase", "language": "en", "weight": 3},
]

def ensure_keyword_file():
    """Create file with defaults only if keywords.yaml does not exist."""
    if not os.path.exists(KEYWORD_FILE):
        with open(KEYWORD_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_KEYWORDS, f, allow_unicode=True)

def detect_language(term: str) -> str:
    """Very simple language detector placeholder."""
    return "en"

def load_keywords():
    """Always load all keywords from YAML file without overwriting user content."""
    try:
        if os.path.exists(KEYWORD_FILE):
            with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
                kws = yaml.safe_load(f) or []
        else:
            kws = DEFAULT_KEYWORDS

        migrated = []
        for kw in kws:
            if isinstance(kw, str):  # in case plain terms are added
                kw = {"term": kw, "type": "phrase", "language": detect_language(kw), "weight": 3}
            if "language" not in kw:
                kw["language"] = detect_language(kw["term"])
            if "type" not in kw:
                kw["type"] = "phrase"
            if "weight" not in kw:
                kw["weight"] = 3
            migrated.append(kw)

        return migrated

    except Exception as e:
        st.error(f"⚠️ Error loading keywords: {e}")
        return DEFAULT_KEYWORDS

# ============ ANALYSIS UTILS ============
def analyze_text(text, keywords):
    """Simple keyword + sentiment analysis."""
    results = []
    for kw in keywords:
        term = kw["term"]
        count = len(re.findall(rf"\b{re.escape(term)}\b", text, re.IGNORECASE))
        if count > 0:
            results.append({"keyword": term, "count": count, "weight": kw["weight"]})
    sentiment = TextBlob(text).sentiment
    return results, sentiment

def scrape_url(url):
    """Fetch website text content."""
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            return soup.get_text(" ", strip=True)
        else:
            return ""
    except Exception:
        return ""

# ============ MAIN UI ============
st.title("🛡️ AI Threat Scanner")

menu = st.sidebar.radio("📌 Navigation", ["📊 Dashboard", "🔎 URL Scanner", "📂 File Analyzer", "🗂️ Keyword DB"])

keywords = load_keywords()

# ---------- Dashboard ----------
if menu == "📊 Dashboard":
    st.subheader("📊 Threat Intelligence Dashboard")
    st.info("Monitor extracted insights, keyword trends, and sentiment from sources.")

    # Wordcloud
    all_terms = [kw["term"] for kw in keywords]
    wordcloud = WordCloud(width=800, height=400, background_color="black").generate(" ".join(all_terms))
    st.image(wordcloud.to_array(), caption="Keyword Cloud")

    # Show keyword DB
    df_kw = pd.DataFrame(keywords)
    st.dataframe(df_kw, use_container_width=True)

# ---------- URL Scanner ----------
elif menu == "🔎 URL Scanner":
    st.subheader("🔎 Scan URLs with AI (Text + Optional Images)")
    urls = st.text_area("Enter URL(s), comma-separated").split(",")
    include_img = st.checkbox("🔍 Include images in scan")

    if st.button("Scan"):
        for url in urls:
            url = url.strip()
            if not url:
                continue
            st.write(f"🌐 Scanning {url} ...")
            text = scrape_url(url)
            if text:
                results, sentiment = analyze_text(text, keywords)
                st.write(f"Sentiment → Polarity: {sentiment.polarity:.2f}, Subjectivity: {sentiment.subjectivity:.2f}")
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                else:
                    st.warning("No keywords detected.")
            else:
                st.error("❌ Could not fetch content.")

# ---------- File Analyzer ----------
elif menu == "📂 File Analyzer":
    st.subheader("📂 Upload & Analyze Files")
    uploaded = st.file_uploader("Upload text file", type=["txt", "md"])
    if uploaded:
        text = uploaded.read().decode("utf-8", errors="ignore")
        results, sentiment = analyze_text(text, keywords)
        st.write(f"Sentiment → Polarity: {sentiment.polarity:.2f}, Subjectivity: {sentiment.subjectivity:.2f}")
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
        else:
            st.warning("No keywords detected.")

# ---------- Keyword DB ----------
elif menu == "🗂️ Keyword DB":
    st.subheader("🗂️ Keyword Database")
    df_kw = pd.DataFrame(load_keywords())
    st.dataframe(df_kw, use_container_width=True)
    st.info("Edit `keywords.yaml` manually to add or modify keywords. They will auto-refresh here.")
