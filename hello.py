# app.py
"""
Anti-India Campaign Detection — Single-file Streamlit app
- Hard-coded Gemini API key (replace placeholder)
- Tabs: Dashboard | URL Scanner | File Analysis | Keyword DB | Utilities
- Always-on AI classification via Gemini REST API
- Keyword DB (local YAML), wordcloud, sentiment, basic risk scoring
"""

import os
import re
import time
import json
import yaml
import base64
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime
from collections import Counter
import streamlit as st
import google.generativeai as genai


# -----------------------------
# CONFIG - paste your API key here
# -----------------------------
# >>> REPLACE the placeholder below with your real Gemini API key <<<
GEMINI_API_KEY = st.secrets["gemini"]



# Gemini REST endpoint used in this app (model: gemini-2.0-flash)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# -----------------------------
# Streamlit page config & CSS
# -----------------------------
st.set_page_config(page_title="Anti-India Campaign Detector", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1720; color: #e6eef6; }
    .card { padding:12px; border-radius:8px; background:#0b1220; box-shadow: 0 2px 10px rgba(0,0,0,0.6); }
    .metric-card { padding:10px; border-radius:8px; background:#071022; text-align:left }
    .small { font-size:0.9rem; color:#aab7c7; }
    .badge { padding:6px 10px; border-radius:999px; font-weight:600; }
    .low { background:#E8FFF1; color:#0B7B37; }
    .med { background:#FFF8E6; color:#8A6100; }
    .high{ background:#FFE8E8; color:#8E0000; }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------------
# Keyword DB (file-backed)
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
    try:
        if os.path.exists(KEYWORD_FILE):
            with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or []
                return data if isinstance(data, list) else DEFAULT_KEYWORDS.copy()
        return DEFAULT_KEYWORDS.copy()
    except Exception:
        return DEFAULT_KEYWORDS.copy()

def save_keywords(kws):
    try:
        with open(KEYWORD_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(kws, f, allow_unicode=True)
        return True
    except Exception:
        st.error("Failed to save keywords.")
        return False

ensure_keyword_file()
keywords = load_keywords()

# -----------------------------
# Helpers: extraction, scoring, AI call
# -----------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

@st.cache_data(show_spinner=False)
def extract_text_from_url(url, timeout=10):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p","h1","h2","h3","li"])]
        return " ".join([t for t in paragraphs if t])
    except Exception:
        return ""

def keyword_hits(text, kw_list):
    text_l = text.lower()
    hits = []
    strength = 0
    for k in kw_list:
        term = str(k.get("term","")).lower().strip()
        if not term: continue
        weight = int(k.get("weight",1) or 1)
        if term.startswith("#"):
            tags = re.findall(r"\B#\w+", text_l)
            if term in tags:
                hits.append(term); strength += weight
        else:
            if re.search(rf"\b{re.escape(term)}\b", text_l):
                hits.append(term); strength += weight
    return sorted(set(hits)), strength

def sentiment_score(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

def account_suspicion_from_row(row):
    score = 0.0
    try:
        followers = float(row.get("followers", 0) or 0)
        if followers < 50: score += 0.5
        elif followers < 300: score += 0.2
    except Exception:
        pass
    try:
        created = row.get("created_at", None)
        if created:
            dt = datetime.strptime(str(created).split("T")[0], "%Y-%m-%d")
            if (datetime.now() - dt).days < 365: score += 0.3
    except Exception:
        pass
    return min(1.0, score)

def compute_risk(keyword_strength, sentiment, engagement_norm=0.0, account_suspicion=0.0):
    k_norm = min(1.0, keyword_strength / 8.0)
    neg = max(0.0, -sentiment)
    w_k, w_e, w_t, w_a = 0.45, 0.2, 0.2, 0.15
    return float(min(1.0, w_k*k_norm + w_e*engagement_norm + w_t*neg + w_a*account_suspicion))

def highlight_sentences(text, hits):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    lh = [h.lower() for h in hits]
    return [s.strip() for s in sentences if any(h in s.lower() for h in lh)]

def call_gemini_classify(text, timeout=18):
    """Call Gemini API and attempt to parse JSON response. Returns (label, explanation)."""
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip().startswith("YOUR_"):
        return ("NoKey", "Gemini API key is not set in code. Replace GEMINI_API_KEY in app.py.")

    payload = {
        "contents": [
            {"parts": [{"text": (
                "Classify the following text for whether it contains anti-India propaganda, calls for boycott/violence, "
                "or coordinated disinformation targeted at India. Return a short JSON object with keys: "
                "label (Safe or Anti-India Detected), confidence (0-1), explanation (1-2 sentences).\n\n"
                f"Text:\n{text[:4000]}"
            )}]}]
    }
    headers = {"Content-Type":"application/json", "X-goog-api-key": GEMINI_API_KEY}

    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        return ("Error", f"Request failed: {e}")

    if resp.status_code != 200:
        # show trimmed message
        return ("Error", f"API {resp.status_code}: {resp.text[:400]}")

    try:
        j = resp.json()
        ai_text = j.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","").strip()
        try:
            parsed = json.loads(ai_text)
            label = parsed.get("label","Unknown")
            explanation = parsed.get("explanation","").strip() or ai_text
            confidence = parsed.get("confidence", None)
            if confidence is None:
                return (label, explanation)
            else:
                return (f"{label} ({confidence})", explanation)
        except Exception:
            lower = ai_text.lower()
            if "anti-india" in lower or "boycott" in lower:
                return ("Anti-India Detected", ai_text)
            if "safe" in lower and "anti" not in lower:
                return ("Safe", ai_text)
            return ("Unknown", ai_text)
    except Exception as e:
        return ("Error", f"Parsing response failed: {e}")

def badge_html(risk):
    if risk < 0.2:
        return f'<span class="badge low">Low · {risk*100:.1f}%</span>'
    if risk < 0.6:
        return f'<span class="badge med">Medium · {risk*100:.1f}%</span>'
    return f'<span class="badge high">High · {risk*100:.1f}%</span>'

# -----------------------------
# Main UI: top header & tabs
# -----------------------------
st.markdown("<h1 style='margin:0 0 6px 0'>🛡️ Anti-India Campaign Detection — Prototype</h1>", unsafe_allow_html=True)
st.markdown("<div class='small'>Prototype: keyword + sentiment + optional Gemini classification (key embedded in code)</div>", unsafe_allow_html=True)
st.write("---")

tabs = st.tabs(["Dashboard", "URL Scanner", "File Analysis", "Keyword DB", "Utilities"])

# -----------------------------
# TAB: Dashboard
# -----------------------------
with tabs[0]:
    st.subheader("Quick Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><b>Keywords</b><div style='font-size:20px'>{len(keywords)}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><b>Default</b><div style='font-size:20px'>{len(DEFAULT_KEYWORDS)}</div></div>", unsafe_allow_html=True)
    ai_status = "Enabled (key in code)" if GEMINI_API_KEY and not GEMINI_API_KEY.strip().startswith("YOUR_") else "Disabled (set GEMINI_API_KEY)"
    col3.markdown(f"<div class='metric-card'><b>AI</b><div style='font-size:20px'>{ai_status}</div></div>", unsafe_allow_html=True)
    preview = ", ".join([k["term"] for k in keywords[:6]]) + ("…" if len(keywords) > 6 else "")
    col4.markdown(f"<div class='metric-card'><b>Preview</b><div style='font-size:14px'>{preview}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.info("Run URL scans or upload a file to see risk insights and AI verdicts.")

# -----------------------------
# TAB: URL Scanner
# -----------------------------
with tabs[1]:
    st.subheader("URL Scanner")
    url_input = st.text_input("Enter one or more URLs (comma-separated)")
    if st.button("Scan URL(s)"):
        urls = [u.strip() for u in url_input.split(",") if u.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            results = []
            prog = st.progress(0)
            for i, u in enumerate(urls, start=1):
                txt = extract_text_from_url(u)
                if not txt:
                    st.warning(f"Could not fetch text from: {u}")
                    results.append({"source": u, "error": True})
                else:
                    hits, ks = keyword_hits(txt, keywords)
                    sent = sentiment_score(txt)
                    highlights = highlight_sentences(txt, hits)
                    risk = compute_risk(ks, sent)
                    ai_label, ai_expl = call_gemini_classify(txt)  # always call (key embedded)
                    results.append({
                        "source": u, "hits": hits, "k_strength": ks, "sentiment": sent,
                        "highlights": highlights, "risk": risk, "ai_label": ai_label, "ai_expl": ai_expl, "text": txt
                    })
                prog.progress(i/len(urls))
                time.sleep(0.08)
            # display
            for r in results:
                if r.get("error"):
                    st.error(f"Failed to fetch: {r['source']}")
                    continue
                st.markdown("---")
                lcol, rcol = st.columns([1,2])
                with lcol:
                    st.markdown(f"**Source:** {r['source']}")
                    st.markdown(badge_html(r["risk"]), unsafe_allow_html=True)
                    st.write(f"Sentiment: {r['sentiment']:.3f}")
                    st.write("Keyword hits:", r["hits"] or "None")
                    if r["ai_label"]:
                        if r["ai_label"].startswith("Anti-India"):
                            st.error(f"AI: {r['ai_label']}")
                        elif r["ai_label"].startswith("Safe"):
                            st.success(f"AI: {r['ai_label']}")
                        else:
                            st.info(f"AI: {r['ai_label']}")
                        if r.get("ai_expl"):
                            st.caption(r["ai_expl"])
                with rcol:
                    if r["highlights"]:
                        st.markdown("**Suspicious sentences:**")
                        for s in r["highlights"]:
                            st.warning(s)
                    else:
                        st.write("No suspicious sentences highlighted.")
                    # wordcloud
                    try:
                        wc = WordCloud(width=800, height=240, background_color="black").generate(r["text"][:12000])
                        fig, ax = plt.subplots(figsize=(8,3))
                        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                        st.pyplot(fig)
                    except Exception:
                        pass

# -----------------------------
# TAB: File Analysis
# -----------------------------
with tabs[2]:
    st.subheader("File Analysis (CSV/JSON/TXT)")
    uploaded_file = st.file_uploader("Upload CSV (with 'text' column) or TXT", type=["csv","json","txt"])
    if st.button("Analyze file") and uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                if "text" not in df.columns:
                    st.error("CSV must contain a 'text' column.")
                else:
                    texts = df["text"].astype(str).fillna("").tolist()
            elif uploaded_file.name.endswith(".json"):
                df = pd.read_json(uploaded_file)
                texts = df.get("text", df.astype(str).apply(" ".join, axis=1)).astype(str).tolist()
            else:
                texts = [uploaded_file.read().decode("utf-8")]
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            texts = []

        if texts:
            records = []
            hashtags_all = []
            engagements = []
            for t in texts:
                hits, ks = keyword_hits(t, keywords)
                s = sentiment_score(t)
                highlights = highlight_sentences(t, hits)
                # naive engagement proxy = number of words (placeholder)
                eng = min(1.0, len(t.split())/500.0)
                engagements.append(eng)
                risk = compute_risk(ks, s, engagement_norm=eng, account_suspicion=0.0)
                ai_label, ai_expl = call_gemini_classify(t)
                tags = re.findall(r"\B#\w+", t.lower())
                hashtags_all.extend(tags)
                records.append({
                    "text": t[:400], "keyword_hits": hits, "k_strength": ks,
                    "sentiment": s, "risk": risk, "ai_label": ai_label, "ai_expl": ai_expl, "highlights": highlights
                })

            df_out = pd.DataFrame(records).sort_values("risk", ascending=False)
            st.markdown("### Top flagged (by risk)")
            # color-coded risk display
            def risk_style(v):
                color = "#E8FFF1" if v<0.2 else "#FFF8E6" if v<0.6 else "#FFE8E8"
                return f"background:{color}"
            st.dataframe(df_out.head(50), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                counts = [sum(df_out["risk"] < 0.2), sum((df_out["risk"] >= 0.2) & (df_out["risk"] < 0.6)), sum(df_out["risk"] >= 0.6)]
                plt.figure(figsize=(4,3)); plt.pie(counts, labels=["Low","Medium","High"], autopct="%1.1f%%"); plt.title("Risk distribution")
                st.pyplot(plt)
            with c2:
                top_tags = Counter(hashtags_all).most_common(15)
                if top_tags:
                    tag_df = pd.DataFrame(top_tags, columns=["hashtag","count"]).set_index("hashtag")
                    st.bar_chart(tag_df)
                else:
                    st.write("No hashtags detected.")

            # download results
            csv = df_out.to_csv(index=False).encode()
            st.download_button("Download results CSV", csv, file_name="analysis_results.csv", mime="text/csv")

# -----------------------------
# TAB: Keyword DB
# -----------------------------
with tabs[3]:
    st.subheader("Keyword Database")
    st.write("Manage keywords (term must contain 'india' to reduce noise).")
    st.dataframe(pd.DataFrame(keywords), use_container_width=True)
    st.markdown("Add a new keyword (must include 'india'):")
    new_term = st.text_input("Keyword / phrase")
    new_type = st.selectbox("Type", ["phrase","hashtag","keyword","word"], index=0)
    new_weight = st.slider("Weight", 1, 10, 3)
    if st.button("Add keyword"):
        if not new_term or "india" not in new_term.lower():
            st.error("Keyword must include 'india'.")
        else:
            exists = {k["term"].lower() for k in keywords}
            if new_term.lower() in exists:
                st.warning("Term exists.")
            else:
                keywords.append({"term": new_term, "type": new_type, "lang":"en", "weight": int(new_weight)})
                if save_keywords(keywords):
                    st.success("Keyword added. File updated.")
                    st.experimental_rerun()

    del_sel = st.multiselect("Delete terms", [k["term"] for k in keywords])
    if st.button("Delete selected"):
        if del_sel:
            keywords[:] = [k for k in keywords if k["term"] not in del_sel]
            if save_keywords(keywords):
                st.success("Deleted selected terms.")
                st.experimental_rerun()

    # import/export
    st.markdown("---")
    st.write("Export / Import keyword DB")
    yaml_bytes = yaml.safe_dump(keywords, allow_unicode=True).encode()
    st.download_button("Export YAML", yaml_bytes, file_name="keywords.yaml", mime="text/yaml")
    csv_bytes = pd.DataFrame(keywords).to_csv(index=False).encode()
    st.download_button("Export CSV", csv_bytes, file_name="keywords.csv", mime="text/csv")
    upload_k = st.file_uploader("Import keywords YAML/CSV", type=["yaml","yml","csv"])
    if upload_k and st.button("Import (replace)"):
        try:
            if upload_k.name.endswith((".yaml", ".yml")):
                loaded = yaml.safe_load(upload_k.read())
                if isinstance(loaded, list):
                    keywords[:] = loaded
                else:
                    st.error("YAML must contain a list.")
            else:
                dfk = pd.read_csv(upload_k)
                keywords[:] = dfk.to_dict(orient="records")
            if save_keywords(keywords):
                st.success("Imported keywords.")
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")

# -----------------------------
# TAB: Utilities
# -----------------------------
with tabs[4]:
    st.subheader("Utilities & Diagnostics")
    st.write("Quick helpers: sentiment, wordcloud, test AI.")
    txt = st.text_area("Enter text to analyze", height=140)
    c1, c2, c3 = st.columns(3)
    if c1.button("Sentiment"):
        p = sentiment_score(txt)
        label = "Positive" if p > 0 else "Negative" if p < 0 else "Neutral"
        st.info(f"Sentiment: {label} (score={p:.3f})")
    if c2.button("WordCloud"):
        try:
            wc = WordCloud(width=800, height=300, background_color="black").generate(txt if txt else " ".join([k["term"] for k in keywords]))
            fig, ax = plt.subplots(figsize=(8,3)); ax.imshow(wc); ax.axis("off"); st.pyplot(fig)
        except Exception as e:
            st.error(f"Wordcloud error: {e}")
    if c3.button("Test AI (classify)"):
        lbl, expl = call_gemini_classify(txt or "test")
        st.write("AI Label:", lbl)
        st.write("AI Explanation:", expl)

st.markdown("---")
st.caption("Prototype — for demonstration. Always validate results manually.")
