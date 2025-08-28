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
import base64
from datetime import datetime
from collections import Counter
import os

st.set_page_config(page_title="Anti-India Campaign Detector", layout="wide")

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
    """Ensure keyword YAML exists; if not, create with defaults."""
    if not os.path.exists(KEYWORD_FILE):
        save_keywords(DEFAULT_KEYWORDS)

def load_keywords():
    try:
        if os.path.exists(KEYWORD_FILE):
            with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or []
                # normalize to list of dicts
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

# Ensure file exists on start
ensure_keyword_file()
keywords = load_keywords()

# -----------------------------
# Scraping & analysis helpers
# -----------------------------
HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def extract_text_from_url(url, timeout=10):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p","h1","h2","h3"])]
        text = " ".join([t for t in paragraphs if t])
        return text
    except Exception:
        return ""

def keyword_hits(text, keywords):
    text_l = text.lower()
    hits = []
    strength = 0
    for k in keywords:
        term = str(k.get("term","")).lower()
        weight = k.get("weight", 1) if k.get("weight") is not None else 1
        if not term:
            continue
        if term.startswith("#"):
            words = re.findall(r"\B#\w+", text_l)
            if term in words:
                hits.append(term)
                strength += weight
        else:
            if re.search(rf"\b{re.escape(term)}\b", text_l):
                hits.append(term)
                strength += weight
    return list(set(hits)), strength

def sentiment_score(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

def compute_risk(keyword_strength, sentiment, engagement_norm=0.0, account_suspicion=0.0):
    k_norm = min(1.0, keyword_strength / 8.0)
    neg = max(0.0, -sentiment)
    w_k, w_e, w_t, w_a = 0.45, 0.2, 0.2, 0.15
    risk = w_k * k_norm + w_e * engagement_norm + w_t * neg + w_a * account_suspicion
    return min(1.0, risk)

def highlight_sentences(text, hits):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    highlighted = []
    lower_hits = [h.lower() for h in hits]
    for s in sentences:
        s_l = s.lower()
        if any(h in s_l for h in lower_hits):
            highlighted.append(s.strip())
    return highlighted

def account_suspicion_from_row(row):
    score = 0.0
    try:
        followers = float(row.get("followers", 0) or 0)
        if followers < 50:
            score += 0.5
        elif followers < 300:
            score += 0.2
    except Exception:
        pass
    try:
        created = row.get("created_at", None)
        if created:
            try:
                dt = datetime.strptime(str(created).split("T")[0], "%Y-%m-%d")
                if (datetime.now() - dt).days < 365:
                    score += 0.3
            except Exception:
                pass
    except Exception:
        pass
    return min(1.0, score)

# -----------------------------
# SIDEBAR: Keyword Manager UI
# -----------------------------
st.sidebar.title("Config & Keyword DB")
st.sidebar.write("Manage the dynamic keyword DB for suspected anti-India terms.")

# Show current keywords in table
st.sidebar.subheader("Current Keywords")
if keywords:
    # convert to DataFrame for nice display
    try:
        df_kw = pd.DataFrame(keywords)
        st.sidebar.dataframe(df_kw.reset_index(drop=True))
    except Exception:
        for k in keywords:
            st.sidebar.write(f"- {k.get('term')} ({k.get('type')}, w={k.get('weight')})")
else:
    st.sidebar.info("No keywords available.")

# Add new keyword form (must contain 'india')
st.sidebar.subheader("Add New Keyword")
with st.sidebar.form("add_keyword_form", clear_on_submit=True):
    new_term = st.text_input("Keyword / phrase (must contain 'india')").strip()
    new_type = st.selectbox("Type", ["phrase", "hashtag", "keyword", "word"])
    new_lang = st.selectbox("Language", ["en", "hi", "ur"], index=0)
    new_weight = st.number_input("Weight (1-10)", min_value=1, max_value=10, value=3)
    add_submitted = st.form_submit_button("Add")

    if add_submitted:
        if not new_term:
            st.sidebar.error("Please enter a term.")
        elif "india" not in new_term.lower():
            st.sidebar.error("Only terms containing 'india' are allowed.")
        else:
            # avoid duplicates
            existing_terms = [k.get("term","").lower() for k in keywords]
            if new_term.lower() in existing_terms:
                st.sidebar.warning("Term already exists.")
            else:
                keywords.append({
                    "term": new_term,
                    "type": new_type,
                    "lang": new_lang,
                    "weight": int(new_weight)
                })
                if save_keywords(keywords):
                    st.sidebar.success(f"Added: {new_term}")
                else:
                    st.sidebar.error("Failed to save. Check permissions.")
                # reload keywords for consistent view
                keywords = load_keywords()

# Delete keywords (multi-select)
st.sidebar.subheader("Delete Keywords")
terms_for_delete = [k.get("term","") for k in keywords]
del_selection = st.sidebar.multiselect("Select terms to delete", options=terms_for_delete)
if st.sidebar.button("Delete selected"):
    if del_selection:
        keywords = [k for k in keywords if k.get("term","") not in del_selection]
        if save_keywords(keywords):
            st.sidebar.success("Deleted selected terms.")
        else:
            st.sidebar.error("Failed to save deletion.")
        keywords = load_keywords()
    else:
        st.sidebar.info("No terms selected for deletion.")

st.sidebar.markdown("---")
st.sidebar.write("⚠️ Note: This app edits a local `keywords.yaml`. On cloud hosts, filesystem persistence can be ephemeral across deploys. Keep a backup in your repo.")

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🛡️ Anti-India Campaign Detection — Prototype")
st.write("Paste website URLs (comma-separated) or upload a CSV/JSON with posts. The app will extract text, check flagged terms (from sidebar DB), compute risk, run sentiment, and highlight suspicious sentences.")

col1, col2 = st.columns([1,1])
with col1:
    url_input = st.text_input("Enter website URL(s) (comma-separated):")
    scan_button = st.button("Scan URL(s)")
with col2:
    uploaded_file = st.file_uploader("Or upload CSV with: platform, username, text, likes, shares, comments, followers, created_at", type=["csv","json"])
    run_file = st.button("Analyze File")

def process_single_text(source_label, text, keywords, extra_meta=None):
    hits, k_strength = keyword_hits(text, keywords)
    sent = sentiment_score(text)
    highlights = highlight_sentences(text, hits)
    risk = compute_risk(k_strength, sent, engagement_norm=0.0, account_suspicion=0.0)
    return {
        "source": source_label,
        "keyword_hits": hits,
        "keyword_strength": k_strength,
        "sentiment": sent,
        "highlights": highlights,
        "risk": risk,
        "raw_text": text
    }

# Scan URLs
if scan_button and url_input:
    urls = [u.strip() for u in url_input.split(",") if u.strip()]
    results = []
    with st.spinner("Fetching and scanning URLs..."):
        for u in urls:
            text = extract_text_from_url(u)
            if not text:
                st.warning(f"Could not extract text from: {u}")
                continue
            res = process_single_text(u, text, keywords)
            results.append(res)

    if results:
        for r in results:
            st.subheader(f"Source: {r['source']}")
            st.metric("Risk score", f"{r['risk']*100:.1f}%")
            st.write("**Keyword hits:**", r['keyword_hits'] or "None")
            st.write("**Sentiment polarity:**", f"{r['sentiment']:.3f}  (-1 negative .. +1 positive)")
            if r['highlights']:
                st.markdown("**Highlighted suspicious sentences:**")
                for sent in r['highlights']:
                    st.info(sent)
            else:
                st.write("No suspicious sentences highlighted.")
            # wordcloud (if text available)
            try:
                wc = WordCloud(width=600, height=250, background_color="white").generate(r['raw_text'][:10000])
                fig, ax = plt.subplots(figsize=(8,3))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except Exception:
                pass

# Analyze uploaded CSV/JSON
if run_file and uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None

    if df is not None and not df.empty:
        st.success("File loaded. Running analysis...")
        if "text" not in df.columns:
            st.error("Uploaded file must have a 'text' column.")
        else:
            df['text'] = df['text'].astype(str).fillna("")
            keyword_list_per_row = []
            k_strengths = []
            sentiments = []
            suspicion_scores = []
            engagements = []
            hashtags_all = []

            for idx, row in df.iterrows():
                text = row['text']
                hits, ks = keyword_hits(text, keywords)
                s = sentiment_score(text)
                keyword_list_per_row.append(hits)
                k_strengths.append(ks)
                sentiments.append(s)
                suspicion_scores.append(account_suspicion_from_row(row))
                likes = float(row.get("likes", 0) or 0)
                shares = float(row.get("shares", 0) or 0)
                comments = float(row.get("comments", 0) or 0)
                followers = float(row.get("followers", 0) or 0)
                pe = (likes + 2*shares + 3*comments) / (1 + (followers if followers>0 else 1))
                engagements.append(pe)
                tags = re.findall(r"\B#\w+", text.lower())
                hashtags_all.extend(tags)

            max_eng = max(engagements) if engagements else 1
            eng_norm = [e / max_eng if max_eng > 0 else 0 for e in engagements]

            risk_scores = []
            for i in range(len(df)):
                r = compute_risk(k_strengths[i], sentiments[i], engagement_norm=eng_norm[i], account_suspicion=suspicion_scores[i])
                risk_scores.append(r)

            df['keyword_hits'] = keyword_list_per_row
            df['keyword_strength'] = k_strengths
            df['sentiment'] = sentiments
            df['eng_norm'] = eng_norm
            df['suspicion'] = suspicion_scores
            df['risk'] = risk_scores

            st.subheader("Top flagged posts by risk")
            cols = ['platform' if 'platform' in df.columns else None,
                    'username' if 'username' in df.columns else None,
                    'text',
                    'likes' if 'likes' in df.columns else None,
                    'shares' if 'shares' in df.columns else None,
                    'comments' if 'comments' in df.columns else None,
                    'followers' if 'followers' in df.columns else None,
                    'keyword_hits','risk']
            cols = [c for c in cols if c is not None]
            topk = df.sort_values(by="risk", ascending=False).head(15)[cols]
            st.dataframe(topk.fillna(""))

            st.subheader("Dashboard Summary")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                counts = [
                    sum(df['risk']<0.2),
                    sum((df['risk']>=0.2)&(df['risk']<0.6)),
                    sum(df['risk']>=0.6)
                ]
                plt.figure(figsize=(4,3))
                plt.pie(counts, labels=["Low","Medium","High"], autopct="%1.1f%%", startangle=90)
                plt.title("Risk distribution")
                st.pyplot(plt)

            with c2:
                top_tags = Counter(hashtags_all).most_common(15)
                if top_tags:
                    tags_df = pd.DataFrame(top_tags, columns=['hashtag','count'])
                    st.bar_chart(data=tags_df.set_index('hashtag'))
                else:
                    st.write("No hashtags detected in dataset.")

            with c3:
                st.write("Sentiment polarity distribution")
                fig2 = plt.figure(figsize=(4,3))
                plt.hist(df['sentiment'], bins=20)
                st.pyplot(fig2)

            if 'username' in df.columns:
                G = nx.Graph()
                for _, row in df.iterrows():
                    uname = str(row.get("username","")).strip()
                    tags = row.get("keyword_hits",[]) or []
                    tags = tags + re.findall(r"\B#\w+", str(row.get("text","")).lower())
                    for t in tags:
                        G.add_edge(uname, t)
                if G.number_of_nodes() > 0:
                    st.subheader("Author-Hashtag Network (small view)")
                    fig, ax = plt.subplots(figsize=(8,5))
                    pos = nx.spring_layout(G, k=0.6, iterations=50)
                    nx.draw(G, pos=pos, with_labels=True, node_size=200, font_size=8)
                    st.pyplot(fig)
                else:
                    st.write("Not enough data for network graph (need 'username' and hashtags).")

            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown("### Download analysis results")
            href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**Notes & next steps:**")
st.markdown("""
- This is a prototype: keyword matching + simple sentiment is used for early detection.
- For production: replace/augment sentiment with transformer models, add rate-limited API ingestors, store in DB, and implement real-time alerts (webhooks/email).
- Always analyze only public content and respect ToS & robots.txt.
""")
