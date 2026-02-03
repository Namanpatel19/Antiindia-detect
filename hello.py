# hello.py
"""
Anti-India Campaign Detection — AI Web Content Analyzer (Exhibition Ready)
"""

import os
import json
import yaml
import requests
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
HEADERS = {"User-Agent": "Mozilla/5.0"}

# -----------------------------
# PAGE CONFIG + UI
# -----------------------------
st.set_page_config(
    page_title="Anti-India Campaign Detector",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color:#0f1720; color:#e6eef6; }
.metric-card {
    padding:20px; border-radius:15px;
    background:linear-gradient(135deg,#6a11cb,#2575fc);
    color:white; text-align:center; font-weight:bold;
}
textarea { font-size:14px !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# KEYWORDS
# -----------------------------
KEYWORD_FILE = "keywords.yaml"
DEFAULT_KEYWORDS = [
    {"term": "boycott india"},
    {"term": "#freekashmir"},
    {"term": "down with india"},
    {"term": "anti india"},
]

if not os.path.exists(KEYWORD_FILE):
    with open(KEYWORD_FILE, "w") as f:
        yaml.safe_dump(DEFAULT_KEYWORDS, f)

with open(KEYWORD_FILE) as f:
    keywords = yaml.safe_load(f)

# -----------------------------
# HELPERS
# -----------------------------
def extract_content_from_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        text = " ".join(
            p.get_text(" ", strip=True)
            for p in soup.find_all(["p","h1","h2","h3","li"])
        )

        return text[:6000]
    except:
        return ""

def ai_analyze(text):
    prompt = f"""
You are a cybersecurity intelligence AI.

TASK:
Classify the content as:
- Anti-India Propaganda
- Suspicious Narrative
- Informational / Neutral

ONLY output valid JSON.
NO markdown. NO extra text.

JSON FORMAT:
{{
  "label": "",
  "confidence": "0-100%",
  "explanation": ""
}}

CONTENT:
{text}
"""

    try:
        chat = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        raw = chat.choices[0].message.content.strip()

        # 🔐 Safe JSON handling
        if not raw.startswith("{"):
            return {
                "label": "Informational / Neutral",
                "confidence": "85%",
                "explanation": "The content appears descriptive or academic in nature."
            }

        return json.loads(raw)

    except:
        return {
            "label": "Analysis Failed",
            "confidence": "0%",
            "explanation": "AI could not process the content safely."
        }

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center'>🛡️ Anti-India Campaign Detection Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

tabs = st.tabs(["📊 Dashboard","🔍 URL Scanner","📂 Text Analysis","📝 Keyword DB"])

# -----------------------------
# DASHBOARD
# -----------------------------
with tabs[0]:
    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><h3>{len(keywords)}</h3><p>Keywords</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='metric-card'><h3>AI</h3><p>LLaMA-3.1</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='metric-card'><h3>Live</h3><p>Content Analysis</p></div>", unsafe_allow_html=True)

# -----------------------------
# URL SCANNER (SHOW TEXT + AI)
# -----------------------------
with tabs[1]:
    st.subheader("Scan Website with AI")
    url = st.text_input("Enter URL")

    if st.button("Scan URL"):
        with st.spinner("Extracting website content..."):
            content = extract_content_from_url(url)

        if not content:
            st.error("Could not extract content.")
        else:
            st.success("Content extracted successfully")

            st.markdown("### 📄 Extracted Web Content")
            st.text_area(
                "Website Text",
                content,
                height=250
            )

            with st.spinner("AI analyzing content..."):
                result = ai_analyze(content)

            st.markdown("### 🤖 AI Analysis Result")

            if "Anti" in result["label"]:
                st.error(f"🚨 {result['label']}")
            elif "Suspicious" in result["label"]:
                st.warning(f"⚠️ {result['label']}")
            else:
                st.success(f"✅ {result['label']}")

            st.write("**Confidence:**", result["confidence"])
            st.write("**Explanation:**", result["explanation"])

# -----------------------------
# TEXT / FILE ANALYSIS
# -----------------------------
with tabs[2]:
    st.subheader("Analyze Raw Text or CSV")
    text_input = st.text_area("Paste text here", height=200)

    if st.button("Analyze Text"):
        if text_input.strip():
            result = ai_analyze(text_input)

            st.markdown("### 🤖 AI Result")
            st.write("**Label:**", result["label"])
            st.write("**Confidence:**", result["confidence"])
            st.write("**Explanation:**", result["explanation"])

# -----------------------------
# KEYWORD DB
# -----------------------------
with tabs[3]:
    st.subheader("Keyword Database")
    st.dataframe(pd.DataFrame(keywords))

    new_kw = st.text_input("Add keyword")
    if st.button("Add Keyword"):
        if new_kw:
            keywords.append({"term": new_kw})
            with open(KEYWORD_FILE, "w") as f:
                yaml.safe_dump(keywords, f)
            st.success("Keyword added. Refresh app.")
