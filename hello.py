# hello.py
"""
Anti-India Campaign Detection — Exhibition Safe Version
(Same UI, Rule-Based Intelligence)
"""

import os
import re
import yaml
import requests
import json
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from langdetect import detect

# -----------------------------
# Streamlit page config & CSS (UNCHANGED UI)
# -----------------------------
st.set_page_config(
    page_title="Anti-India Campaign Detector",
    page_icon="🛡️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f1720; color: #e6eef6; }
    .metric-card {
        padding:20px; border-radius:15px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color:white; text-align:center; font-weight:bold;
        box-shadow:0 4px 15px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Keyword DB
# -----------------------------
KEYWORD_FILE = "keywords.yaml"

DEFAULT_KEYWORDS = [
    {"term": "boycott india", "type": "phrase", "language": "English", "weight": 4},
    {"term": "#freekashmir", "type": "hashtag", "language": "English", "weight": 5},
    {"term": "down with india", "type": "phrase", "language": "English", "weight": 4},
    {"term": "anti india", "type": "phrase", "language": "English", "weight": 3},
]

def ensure_keyword_file():
    if not os.path.exists(KEYWORD_FILE):
        with open(KEYWORD_FILE, "w") as f:
            yaml.safe_dump(DEFAULT_KEYWORDS, f)

def load_keywords():
    with open(KEYWORD_FILE, "r") as f:
        return yaml.safe_load(f) or DEFAULT_KEYWORDS

def save_keywords(kws):
    with open(KEYWORD_FILE, "w") as f:
        yaml.safe_dump(kws, f)

ensure_keyword_file()
keywords = load_keywords()

# -----------------------------
# Helpers
# -----------------------------
HEADERS = {"User-Agent": "Mozilla/5.0"}

@st.cache_data(show_spinner=False)
def extract_content_from_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        text = " ".join(
            p.get_text()
            for p in soup.find_all(["p", "h1", "h2", "li"])
        )
        return text
    except:
        return ""

def detect_language(word):
    try:
        return detect(word)
    except:
        return "en"

def rule_based_ai_analysis(text):
    text_l = text.lower()
    score = 0
    matched = []

    for kw in keywords:
        if kw["term"].lower() in text_l:
            score += kw["weight"]
            matched.append(kw["term"])

    if score >= 7:
        label = "🚨 Anti-India Campaign Detected"
        explanation = f"High confidence detection. Keywords found: {', '.join(matched)}"
    elif score >= 3:
        label = "⚠️ Suspicious Content"
        explanation = f"Potential propaganda indicators found: {', '.join(matched)}"
    else:
        label = "✅ No Anti-India Content Detected"
        explanation = "Content does not match known propaganda patterns."

    return label, explanation, score

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<h1 style='text-align:center'>🛡️ Anti-India Campaign Detection Dashboard</h1>",
    unsafe_allow_html=True
)
st.write("---")

tabs = st.tabs(["📊 Dashboard","🔍 URL Scanner","📂 File Analysis","📝 Keyword DB"])

# -----------------------------
# Dashboard
# -----------------------------
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div class='metric-card'><h3>{len(keywords)}</h3><p>Keywords</p></div>",
        unsafe_allow_html=True
    )
    col2.markdown(
        "<div class='metric-card'><h3>Offline</h3><p>AI Monitoring</p></div>",
        unsafe_allow_html=True
    )
    col3.markdown(
        "<div class='metric-card'><h3>Cyber</h3><p>Threat Analysis</p></div>",
        unsafe_allow_html=True
    )

# -----------------------------
# URL Scanner
# -----------------------------
with tabs[1]:
    st.subheader("Scan URLs with Intelligence Engine")
    url = st.text_input("Enter URL")

    if st.button("Scan URL"):
        if not url:
            st.warning("Please enter a URL")
        else:
            with st.spinner("Analyzing content..."):
                text = extract_content_from_url(url)

            label, expl, score = rule_based_ai_analysis(text)

            st.metric("Threat Score", score)

            if "Detected" in label:
                st.error(f"{label}\n\n{expl}")
            elif "Suspicious" in label:
                st.warning(f"{label}\n\n{expl}")
            else:
                st.success(f"{label}\n\n{expl}")

# -----------------------------
# File Analysis
# -----------------------------
with tabs[2]:
    st.subheader("Analyze Text / CSV")
    f = st.file_uploader("Upload file", type=["txt","csv"])

    if f and st.button("Analyze"):
        texts = []

        if f.name.endswith(".csv"):
            df = pd.read_csv(f)
            texts = df.iloc[:,0].astype(str).tolist()
        else:
            texts = [f.read().decode("utf-8", errors="ignore")]

        results = []
        for t in texts:
            lbl, expl, score = rule_based_ai_analysis(t)
            results.append({
                "Text": t[:80],
                "Label": lbl,
                "Score": score,
                "Explanation": expl
            })

        st.dataframe(pd.DataFrame(results))

# -----------------------------
# Keyword DB
# -----------------------------
with tabs[3]:
    st.subheader("Manage Keywords")
    st.dataframe(pd.DataFrame(keywords))

    new_kw = st.text_input("Add new keyword")
    if st.button("Add"):
        if new_kw:
            keywords.append({
                "term": new_kw,
                "type": "phrase",
                "language": detect_language(new_kw),
                "weight": 3
            })
            save_keywords(keywords)
            st.success("Keyword added! Refresh app.")
        else:
            st.warning("Keyword cannot be empty")
