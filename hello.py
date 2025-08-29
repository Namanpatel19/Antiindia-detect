# upgraded_app.py

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
import base64
from datetime import datetime
from collections import Counter
import os
import time
import json

st.set_page_config(page_title="🛡️ Anti-India Campaign Detector", layout="wide")

# -----------------------------
# Constants & default keywords
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

# -----------------------------
# Keyword DB helpers
# -----------------------------
def ensure_keyword_file():
    if not os.path.exists(KEYWORD_FILE):
        save_keywords(DEFAULT_KEYWORDS)

def load_keywords():
    try:
        if os.path.exists(KEYWORD_FILE):
            with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or []
                return data if isinstance(data, list) else []
        else:
            return DEFAULT_KEYWORDS.copy()
    except Exception:
        return DEFAULT_KEYWORDS.copy()

def save_keywords(keywords):
    try:
        with open(KEYWORD_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(keywords, f, allow_unicode=True)
        return True
    except Exception as e:
        st.error(f"Error saving keywords: {e}")
        return False

ensure_keyword_file()
keywords = load_keywords()

# -----------------------------
# Gemini AI helpers
# -----------------------------
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def get_gemini_api_key():
    try:
        return st.secrets["gemini"]["api_key"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY", None)

def call_gemini_classify(text, timeout=20):
    api_key = get_gemini_api_key()
    if not api_key:
        return ("NoKey", "Gemini API key not configured.")

    prompt = (
        "Classify the following text for whether it contains anti-India propaganda, calls for boycott/violence "
        "or coordinated disinformation targeted at India. Return JSON: {label, confidence, explanation}.\n\n"
        f"Text:\n{text}"
    )

    payload = {"contents":[{"parts":[{"text": prompt}]}]}
    headers = {"Content-Type":"application/json","X-goog-api-key": api_key}

    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        return ("Error", f"Request failed: {e}")

    if resp.status_code != 200:
        return ("Error", f"API {resp.status_code}: {resp.text[:400]}")

    try:
        j = resp.json()
        ai_text = j.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","").strip()
        try:
            parsed = json.loads(ai_text)
            label = parsed.get("label","").strip() or "Unknown"
            explanation = parsed.get("explanation","").strip() or ai_text
            confidence = parsed.get("confidence", None)
            if confidence is None:
                return (label, explanation)
            else:
                return (f"{label} ({confidence})", explanation)
        except Exception:
            txtlower = ai_text.lower()
            if "anti-india" in txtlower or "boycott" in txtlower:
                return ("Anti-India Detected", ai_text)
            if "safe" in txtlower and "anti" not in txtlower:
                return ("Safe", ai_text)
            return ("Unknown", ai_text)
    except Exception as e:
        return ("Error", f"Parsing response failed: {e}")

# -----------------------------
# Analysis helpers
# -----------------------------
HEADERS = {"User-Agent":"Mozilla/5.0"}

def extract_text_from_url(url, timeout=10):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p","h1","h2","h3"])]
        return " ".join([t for t in paragraphs if t])
    except Exception:
        return ""

def keyword_hits(text, keywords):
    text_l = text.lower()
    hits, strength = [], 0
    for k in keywords:
        term = str(k.get("term","")).lower()
        weight = k.get("weight",1)
        if not term: continue
        if term.startswith("#"):
            words = re.findall(r"\B#\w+", text_l)
            if term in words:
                hits.append(term); strength += weight
        else:
            if re.search(rf"\b{re.escape(term)}\b", text_l):
                hits.append(term); strength += weight
    return list(set(hits)), strength

def sentiment_score(text):
    try: return TextBlob(text).sentiment.polarity
    except: return 0.0

def compute_risk(keyword_strength, sentiment, engagement_norm=0.0, account_suspicion=0.0):
    k_norm = min(1.0, keyword_strength / 8.0)
    neg = max(0.0, -sentiment)
    w_k, w_e, w_t, w_a = 0.45, 0.2, 0.2, 0.15
    return min(1.0, w_k*k_norm + w_e*engagement_norm + w_t*neg + w_a*account_suspicion)

def highlight_sentences(text, hits):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    lower_hits = [h.lower() for h in hits]
    return [s.strip() for s in sentences if any(h in s.lower() for h in lower_hits)]

def account_suspicion_from_row(row):
    score = 0.0
    try:
        followers = float(row.get("followers", 0) or 0)
        if followers < 50: score += 0.5
        elif followers < 300: score += 0.2
    except: pass
    try:
        created = row.get("created_at", None)
        if created:
            dt = datetime.strptime(str(created).split("T")[0], "%Y-%m-%d")
            if (datetime.now()-dt).days < 365: score += 0.3
    except: pass
    return min(1.0, score)

def process_single_text(source_label, text, keywords, ai_enabled=False):
    hits, k_strength = keyword_hits(text, keywords)
    sent = sentiment_score(text)
    highlights = highlight_sentences(text, hits)
    risk = compute_risk(k_strength, sent)
    ai_label, ai_expl = None, None
    if ai_enabled and (k_strength>0 or risk>=0.25):
        ai_label, ai_expl = call_gemini_classify(text)
        time.sleep(0.2)
    return {
        "source": source_label, "keyword_hits": hits, "keyword_strength": k_strength,
        "sentiment": sent, "highlights": highlights, "risk": risk,
        "raw_text": text, "ai_label": ai_label, "ai_explanation": ai_expl
    }

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🛡️ Anti-India Campaign Detection — Prototype")

tabs = st.tabs(["📊 Dashboard", "🔗 URL Scanner", "📁 File Analysis", "⚙️ Keyword Manager"])

# --- Dashboard ---
with tabs[0]:
    st.subheader("📊 Quick Dashboard")
    col1,col2,col3 = st.columns(3)
    col1.metric("Total Keywords", len(keywords))
    col2.metric("Default Keywords", len(DEFAULT_KEYWORDS))
    col3.metric("AI Classification", "Enabled" if st.session_state.get("use_ai", False) else "Disabled")
    st.info("👉 Run scans from other tabs to populate more data.")

# --- URL Scanner ---
with tabs[1]:
    st.subheader("🔗 Scan Website(s)")
    url_input = st.text_input("Enter URLs (comma-separated):")
    use_ai = st.checkbox("Enable Gemini AI classification", value=False, key="use_ai")
    if st.button("🚀 Scan URLs"):
        urls = [u.strip() for u in url_input.split(",") if u.strip()]
        results = []
        for u in urls:
            text = extract_text_from_url(u)
            if not text:
                st.warning(f"Could not extract text from: {u}")
                continue
            res = process_single_text(u, text, keywords, ai_enabled=use_ai)
            results.append(res)

        for r in results:
            with st.expander(f"📌 Results for {r['source']}"):
                st.metric("Risk Score", f"{r['risk']*100:.1f}%")
                st.write("**Keyword Hits:**", r['keyword_hits'] or "None")
                st.write("**Sentiment:**", f"{r['sentiment']:.3f}")
                if r['ai_label']:
                    st.info(f"AI: {r['ai_label']}")
                    st.caption(r['ai_explanation'])
                if r['highlights']:
                    st.markdown("**⚠️ Suspicious Sentences:**")
                    for sent in r['highlights']:
                        st.error(sent)

# --- File Analysis ---
with tabs[2]:
    st.subheader("📁 Upload File for Analysis")
    uploaded_file = st.file_uploader("Upload CSV or JSON dataset", type=["csv","json"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_json(uploaded_file)
        st.success(f"Loaded {len(df)} records")
        if "text" not in df.columns:
            st.error("Dataset must contain a `text` column.")
        else:
            df['text'] = df['text'].astype(str).fillna("")
            keyword_hits_list, risks, sentiments = [], [], []
            for _, row in df.iterrows():
                hits, ks = keyword_hits(row['text'], keywords)
                sent = sentiment_score(row['text'])
                risk = compute_risk(ks, sent)
                keyword_hits_list.append(hits); sentiments.append(sent); risks.append(risk)
            df['keyword_hits'] = keyword_hits_list
            df['sentiment'] = sentiments
            df['risk'] = risks

            st.subheader("🔥 Top Risky Posts")
            st.dataframe(df.sort_values("risk", ascending=False).head(10)[['text','keyword_hits','risk']])

            c1,c2 = st.columns([1,1])
            with c1:
                counts = [
                    sum(df['risk']<0.2),
                    sum((df['risk']>=0.2)&(df['risk']<0.6)),
                    sum(df['risk']>=0.6)
                ]
                plt.figure(figsize=(4,3))
                plt.pie(counts, labels=["Low","Medium","High"], autopct="%1.1f%%")
                plt.title("Risk Levels")
                st.pyplot(plt)
            with c2:
                plt.hist(df['sentiment'], bins=20)
                plt.title("Sentiment Distribution")
                st.pyplot(plt)

            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv, "analysis_results.csv", "text/csv")

# --- Keyword Manager ---
with tabs[3]:
    st.subheader("⚙️ Manage Keywords")
    st.dataframe(pd.DataFrame(keywords))
    st.markdown("### ➕ Add New Keyword")
    new_term = st.text_input("New keyword (must contain 'india')", key="new_term")
    new_type = st.selectbox("Type", ["phrase","hashtag","keyword"], key="new_type")
    if st.button("Add Keyword"):
        if "india" not in new_term.lower():
            st.error("Only keywords containing 'india' allowed.")
        else:
            keywords.append({"term": new_term, "type": new_type, "lang": "en", "weight": 3})
            save_keywords(keywords)
            st.success(f"Added {new_term}")

    st.markdown("### ❌ Delete Keywords")
    to_del = st.multiselect("Select to delete", [k['term'] for k in keywords])
    if st.button("Delete Selected"):
        keywords[:] = [k for k in keywords if k['term'] not in to_del]
        save_keywords(keywords)
        st.success("Deleted selected keywords.")
