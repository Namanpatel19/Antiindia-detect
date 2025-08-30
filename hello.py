# app.py
"""
Anti-India Campaign Detection — Cybersecurity Dashboard Style
"""

import os, re, time, json, yaml, requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime
from collections import Counter
import plotly.express as px

# -----------------------------
# CONFIG - paste your API key here
# -----------------------------
GEMINI_API_KEY = st.secrets["gemini"]

# Gemini REST endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# -----------------------------
# Streamlit page config & CSS
# -----------------------------
st.set_page_config(page_title="Anti-India Campaign Detector", page_icon="🛡️", layout="wide")

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
    .stTabs [role="tablist"] {
        display:flex; justify-content:center; gap:10px;
        background:#1E1E1E; padding:10px; border-radius:12px;
    }
    .stTabs [role="tab"] {
        background:#2C2F38; color:white; padding:10px 20px;
        border-radius:10px; font-weight:bold;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background:linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color:white; box-shadow:0 4px 10px rgba(0,0,0,0.4);
    }
    .badge { padding:6px 10px; border-radius:999px; font-weight:600; }
    .low { background:#E8FFF1; color:#0B7B37; }
    .med { background:#FFF8E6; color:#8A6100; }
    .high{ background:#FFE8E8; color:#8E0000; }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------------
# Keyword DB
# -----------------------------
KEYWORD_FILE = "keywords.yaml"
DEFAULT_KEYWORDS = [
    {"term": "boycott india", "type": "phrase", "lang": "en", "weight": 4},
    {"term": "#freekashmir", "type": "hashtag", "lang": "en", "weight": 5},
    {"term": "down with india", "type": "phrase", "lang": "en", "weight": 4},
    {"term": "anti-india", "type": "keyword", "lang": "en", "weight": 3},
    {"term": "destroy india", "type": "phrase", "lang": "en", "weight": 5},
    {"term": "traitor india", "type": "phrase", "lang": "en", "weight": 3},
]
def ensure_keyword_file():
    if not os.path.exists(KEYWORD_FILE):
        with open(KEYWORD_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_KEYWORDS, f, allow_unicode=True)
def load_keywords():
    if os.path.exists(KEYWORD_FILE):
        with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or DEFAULT_KEYWORDS
    return DEFAULT_KEYWORDS
def save_keywords(kws):
    with open(KEYWORD_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(kws, f, allow_unicode=True)
    return True
ensure_keyword_file()
keywords = load_keywords()

# -----------------------------
# Helpers
# -----------------------------
HEADERS = {"User-Agent": "Mozilla/5.0"}
@st.cache_data(show_spinner=False)
def extract_text_from_url(url, timeout=10):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        return " ".join([p.get_text(" ", strip=True) for p in soup.find_all(["p","h1","h2","h3","li"])])
    except: return ""
def keyword_hits(text, kw_list):
    hits, strength = [], 0; text_l = text.lower()
    for k in kw_list:
        term = str(k.get("term","")).lower().strip(); weight=int(k.get("weight",1))
        if not term: continue
        if term.startswith("#"):
            if term in re.findall(r"\B#\w+", text_l): hits.append(term); strength += weight
        else:
            if re.search(rf"\b{re.escape(term)}\b", text_l): hits.append(term); strength += weight
    return sorted(set(hits)), strength
def sentiment_score(text): return TextBlob(text).sentiment.polarity
def compute_risk(keyword_strength, sentiment): 
    return min(1.0, 0.45*min(1,keyword_strength/8.0) + 0.2*0 + 0.2*max(0,-sentiment))
def highlight_sentences(text, hits):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if any(h in s.lower() for h in hits)]
def badge_html(risk):
    if risk < 0.2: return f'<span class="badge low">Low · {risk*100:.1f}%</span>'
    if risk < 0.6: return f'<span class="badge med">Medium · {risk*100:.1f}%</span>'
    return f'<span class="badge high">High · {risk*100:.1f}%</span>'

# -----------------------------
# Gemini AI call
# -----------------------------
def call_gemini_classify(text):
    if not GEMINI_API_KEY: return ("NoKey","Gemini API key not set.")
    payload = {
        "contents": [{"parts": [{"text": f"Classify this text: {text[:4000]}"}]}]
    }
    headers = {"Content-Type":"application/json","X-goog-api-key":GEMINI_API_KEY}
    try:
        r = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=15)
        if r.status_code!=200: return ("Error", r.text[:200])
        j=r.json(); ai_text=j.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
        return ("Anti-India Detected","Suspicious content") if "india" in ai_text.lower() else ("Safe", ai_text)
    except Exception as e: return ("Error", str(e))

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center'>🛡️ Anti-India Campaign Detection Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

# -----------------------------
# TABS
# -----------------------------
tabs = st.tabs(["📊 Dashboard","🔍 URL Scanner","📂 File Analysis","📝 Keyword DB","⚙ Utilities"])

# -----------------------------
# TAB: Dashboard
# -----------------------------
with tabs[0]:
    col1,col2,col3,col4=st.columns(4)
    col1.markdown(f"<div class='metric-card'><h3>{len(keywords)}</h3><p>Keywords</p></div>",unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>29</h3><p>High-Risk Events</p></div>",unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3>08</h3><p>Risky Activities</p></div>",unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><h3>06</h3><p>High-Risk Users</p></div>",unsafe_allow_html=True)

    st.markdown("### 📈 Activity Over Time")
    df=pd.DataFrame({"Month":["Jan","Feb","Mar","Apr","May","Jun","Jul"],"Events":[200,300,400,600,500,700,900]})
    fig=px.line(df,x="Month",y="Events",markers=True)
    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# TAB: URL Scanner
# -----------------------------
with tabs[1]:
    st.subheader("Scan URLs")
    url_input=st.text_input("Enter URL(s), comma-separated")
    if st.button("Scan"):
        for u in [x.strip() for x in url_input.split(",") if x.strip()]:
            txt=extract_text_from_url(u)
            hits,ks=keyword_hits(txt,keywords); sent=sentiment_score(txt); risk=compute_risk(ks,sent)
            ai_label,ai_expl=call_gemini_classify(txt)
            st.markdown("---")
            st.markdown(f"**{u}** {badge_html(risk)}",unsafe_allow_html=True)
            st.write("Hits:",hits); st.write("Sentiment:",sent)
            if ai_label.startswith("Anti"): st.error(ai_label)
            else: st.success(ai_label)
            if txt:
                wc=WordCloud(width=600,height=250,background_color="black").generate(txt)
                fig,ax=plt.subplots(figsize=(6,3)); ax.imshow(wc); ax.axis("off"); st.pyplot(fig)

# -----------------------------
# TAB: File Analysis
# -----------------------------
with tabs[2]:
    st.subheader("Analyze CSV/TXT")
    f=st.file_uploader("Upload file",type=["csv","txt"])
    if f and st.button("Analyze"):
        texts=[]
        if f.name.endswith(".csv"):
            df=pd.read_csv(f); texts=df["text"].astype(str).tolist()
        else: texts=[f.read().decode()]
        recs=[]
        for t in texts:
            hits,ks=keyword_hits(t,keywords); s=sentiment_score(t); risk=compute_risk(ks,s); lbl,expl=call_gemini_classify(t)
            recs.append({"text":t[:80],"hits":hits,"risk":risk,"ai":lbl})
        st.dataframe(pd.DataFrame(recs))

# -----------------------------
# TAB: Keyword DB
# -----------------------------
with tabs[3]:
    st.subheader("Manage Keywords")
    st.dataframe(pd.DataFrame(keywords))
    new=st.text_input("New keyword"); 
    if st.button("Add") and "india" in new.lower():
        keywords.append({"term":new,"type":"phrase","lang":"en","weight":3}); save_keywords(keywords); st.success("Added")

# -----------------------------
# TAB: Utilities
# -----------------------------
with tabs[4]:
    txt=st.text_area("Enter text")
    if st.button("Sentiment"): st.write("Score:", sentiment_score(txt))
    if st.button("Wordcloud"): 
        wc=WordCloud(width=600,height=250,background_color="black").generate(txt or " ".join([k["term"] for k in keywords]))
        fig,ax=plt.subplots(); ax.imshow(wc); ax.axis("off"); st.pyplot(fig)
    if st.button("Test AI"): st.write(call_gemini_classify(txt))
