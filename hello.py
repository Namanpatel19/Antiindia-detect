# hello.py
"""
Anti-India Campaign Detection — AI Version (Groq API)
"""

import os, re, yaml, requests, json
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from langdetect import detect
from groq import Groq

# -----------------------------
# CONFIG
# -----------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.1-8b-instant"

# -----------------------------
# Streamlit page config & CSS
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
    {"term": "boycott india", "weight": 4},
    {"term": "#freekashmir", "weight": 5},
    {"term": "down with india", "weight": 4},
]

if not os.path.exists(KEYWORD_FILE):
    with open(KEYWORD_FILE, "w") as f:
        yaml.safe_dump(DEFAULT_KEYWORDS, f)

with open(KEYWORD_FILE) as f:
    keywords = yaml.safe_load(f)

# -----------------------------
# Helpers
# -----------------------------
HEADERS = {"User-Agent": "Mozilla/5.0"}

def extract_content_from_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        return " ".join(p.get_text() for p in soup.find_all(["p","h1","h2","li"]))
    except:
        return ""

def ai_analyze(text):
    prompt = f"""
You are a cybersecurity AI.

Analyze the content below and detect:
1. Anti-India propaganda
2. Hate speech
3. Coordinated influence campaigns

Respond strictly in JSON:
{{
  "label": "...",
  "confidence": "...",
  "explanation": "..."
}}

Content:
{text[:3500]}
"""

    try:
        chat = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return json.loads(chat.choices[0].message.content)
    except Exception as e:
        return {
            "label": "Error",
            "confidence": "0%",
            "explanation": str(e)
        }

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
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><h3>{len(keywords)}</h3><p>Keywords</p></div>",unsafe_allow_html=True)
    c2.markdown("<div class='metric-card'><h3>AI</h3><p>Groq LLaMA 3.1</p></div>",unsafe_allow_html=True)
    c3.markdown("<div class='metric-card'><h3>Live</h3><p>Threat Detection</p></div>",unsafe_allow_html=True)

# -----------------------------
# URL Scanner
# -----------------------------
with tabs[1]:
    st.subheader("Scan URL with AI")
    url = st.text_input("Enter URL")

    if st.button("Scan"):
        with st.spinner("AI analyzing content..."):
            text = extract_content_from_url(url)
            result = ai_analyze(text)

        if "Anti" in result["label"]:
            st.error(f"🚨 {result['label']}")
        else:
            st.success(f"✅ {result['label']}")

        st.write("**Confidence:**", result["confidence"])
        st.write("**Explanation:**", result["explanation"])

# -----------------------------
# File Analysis
# -----------------------------
with tabs[2]:
    f = st.file_uploader("Upload TXT or CSV", type=["txt","csv"])

    if f and st.button("Analyze"):
        texts = []
        if f.name.endswith(".csv"):
            df = pd.read_csv(f)
            texts = df.iloc[:,0].astype(str).tolist()
        else:
            texts = [f.read().decode()]

        rows = []
        for t in texts:
            r = ai_analyze(t)
            rows.append({
                "Text": t[:80],
                "Label": r["label"],
                "Confidence": r["confidence"],
                "Explanation": r["explanation"]
            })

        st.dataframe(pd.DataFrame(rows))

# -----------------------------
# Keyword DB
# -----------------------------
with tabs[3]:
    st.dataframe(pd.DataFrame(keywords))
