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

st.set_page_config(page_title="Anti-India Campaign Detector", layout="wide")

# -----------------------------
# Helper: Load keyword DB (YAML)
# -----------------------------
DEFAULT_KEYWORDS_YAML = """
- term: "boycott india"
  type: "phrase"
  lang: "en"
  weight: 4
- term: "#freekashmir"
  type: "hashtag"
  lang: "en"
  weight: 5
- term: "down with india"
  type: "phrase"
  lang: "en"
  weight: 4
- term: "anti-india"
  type: "keyword"
  lang: "en"
  weight: 3
- term: "destroy india"
  type: "phrase"
  lang: "en"
  weight: 5
- term: "traitor india"
  type: "phrase"
  lang: "en"
  weight: 3
"""

@st.cache_data
def load_keywords_from_string(yaml_str):
    try:
        return yaml.safe_load(yaml_str)
    except Exception:
        return []

@st.cache_data
def load_keywords_from_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return []

# -----------------------------
# Helper: Extract text from URL
# -----------------------------
HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def extract_text_from_url(url, timeout=10):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # join paragraphs and headings
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p","h1","h2","h3"])]
        text = " ".join([t for t in paragraphs if t])
        return text
    except Exception as e:
        return ""

# -----------------------------
# Helper: Keyword detection
# -----------------------------
def keyword_hits(text, keywords):
    text_l = text.lower()
    hits = []
    strength = 0
    for k in keywords:
        term = k.get("term","").lower()
        weight = k.get("weight",1)
        if not term: 
            continue
        # hashtag exact word match
        if term.startswith("#"):
            # split text to words to match hashtag tokens
            words = re.findall(r"\B#\w+", text_l)
            if term in words:
                hits.append(term)
                strength += weight
        else:
            # regex whole-word match for phrases/keywords
            # allow phrase matches too
            if re.search(rf"\b{re.escape(term)}\b", text_l):
                hits.append(term)
                strength += weight
    return list(set(hits)), strength

# -----------------------------
# Helper: Sentiment
# -----------------------------
def sentiment_score(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity  # -1 .. 1
    except Exception:
        return 0.0

# -----------------------------
# Simple risk scoring
# -----------------------------
def compute_risk(keyword_strength, sentiment, engagement_norm=0.0, account_suspicion=0.0):
    # Normalize keyword_strength roughly (assuming seed weights sum small)
    k_norm = min(1.0, keyword_strength / 8.0)
    # sentiment negative => higher risk
    neg = max(0.0, -sentiment)  # 0..1
    # Simple weighted sum
    w_k, w_e, w_t, w_a = 0.45, 0.2, 0.2, 0.15
    risk = w_k * k_norm + w_e * engagement_norm + w_t * neg + w_a * account_suspicion
    return min(1.0, risk)

# -----------------------------
# Helper: Highlight sentences containing keywords
# -----------------------------
def highlight_sentences(text, hits):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    highlighted = []
    lower_hits = [h.lower() for h in hits]
    for s in sentences:
        s_l = s.lower()
        if any(h in s_l for h in lower_hits):
            highlighted.append(s.strip())
    return highlighted

# -----------------------------
# Small bot/account suspicion heuristic
# -----------------------------
def account_suspicion_from_row(row):
    # expects 'followers' and 'created_at' columns optionally
    score = 0.0
    # low followers
    try:
        followers = float(row.get("followers", 0))
        if followers < 50:
            score += 0.5
        elif followers < 300:
            score += 0.2
    except Exception:
        pass
    # recently created (if available)
    try:
        created = row.get("created_at", None)
        if created:
            # try parse common formats
            for fmt in ("%Y-%m-%d","%d-%m-%Y","%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(created.split("T")[0], "%Y-%m-%d")
                    # if created within 1 year -> suspicion
                    if (datetime.now() - dt).days < 365:
                        score += 0.3
                    break
                except Exception:
                    pass
    except Exception:
        pass
    return min(1.0, score)

# -----------------------------
# UI: Left sidebar for keywords
# -----------------------------
st.sidebar.title("Config & Keyword DB")
st.sidebar.markdown("You can edit the keyword DB below. Keep terms that indicate anti-India narratives. Save locally if you change.")

yaml_input = st.sidebar.text_area("keywords.yaml (editable)", value=DEFAULT_KEYWORDS_YAML, height=250)
keywords = load_keywords_from_string(yaml_input) or []

# -----------------------------
# Main UI layout
# -----------------------------
st.title("🛡️ Anti-India Campaign Detection — Prototype")
st.write("Paste one or more website URLs (comma-separated) OR upload a CSV/JSON with posts. The app will extract text, check for flagged terms, compute a risk score, run sentiment, and show highlights & simple engagement analysis.")

col1, col2 = st.columns([1,1])

with col1:
    url_input = st.text_input("Enter website URL(s) (comma-separated):")
    scan_button = st.button("Scan URL(s)")

with col2:
    uploaded_file = st.file_uploader("Or upload CSV with columns: platform, username, text, likes, shares, comments, followers, created_at", type=["csv","json"])
    run_file = st.button("Analyze File")

# -----------------------------
# If user scans URL(s)
# -----------------------------
def process_single_text(source_label, text, extra_meta=None):
    hits, k_strength = keyword_hits(text, keywords)
    sent = sentiment_score(text)
    highlights = highlight_sentences(text, hits)
    # no engagement info for raw URL; set 0
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

if scan_button and url_input:
    urls = [u.strip() for u in url_input.split(",") if u.strip()]
    results = []
    with st.spinner("Fetching and scanning URLs..."):
        for u in urls:
            text = extract_text_from_url(u)
            if not text:
                st.warning(f"Could not extract text from: {u}")
                continue
            res = process_single_text(u, text)
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
            # small wordcloud
            wc = WordCloud(width=600, height=250, background_color="white").generate(r['raw_text'][:10000])
            fig, ax = plt.subplots(figsize=(8,3))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# -----------------------------
# If user uploaded CSV/JSON
# -----------------------------
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
        # Ensure text column exists
        if "text" not in df.columns:
            st.error("Uploaded file must have a 'text' column.")
        else:
            # fill NA
            df['text'] = df['text'].astype(str).fillna("")
            # compute fields
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
                # engagement metric: likes + 2*shares + 3*comments normalized later
                likes = float(row.get("likes", 0) or 0)
                shares = float(row.get("shares", 0) or 0)
                comments = float(row.get("comments", 0) or 0)
                followers = float(row.get("followers", 0) or 0)
                pe = (likes + 2*shares + 3*comments) / (1 + (followers if followers>0 else 1))
                engagements.append(pe)
                # extract hashtags from text
                tags = re.findall(r"\B#\w+", text.lower())
                hashtags_all.extend(tags)

            # normalize engagement to 0..1
            if engagements:
                max_eng = max(engagements)
                eng_norm = [e / max_eng if max_eng > 0 else 0 for e in engagements]
            else:
                eng_norm = [0]*len(df)

            # risk scores
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

            # show top suspicious posts
            st.subheader("Top flagged posts by risk")
            topk = df.sort_values(by="risk", ascending=False).head(15)[['platform' if 'platform' in df.columns else None, 'username' if 'username' in df.columns else None, 'text','likes' if 'likes' in df.columns else None,'shares' if 'shares' in df.columns else None,'comments' if 'comments' in df.columns else None,'followers' if 'followers' in df.columns else None,'keyword_hits','risk']].drop(columns=[c for c in [None] if c is None])
            st.dataframe(topk.fillna(""))

            # summary charts
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
                # top hashtags
                top_tags = Counter(hashtags_all).most_common(15)
                if top_tags:
                    tags_df = pd.DataFrame(top_tags, columns=['hashtag','count'])
                    st.bar_chart(data=tags_df.set_index('hashtag'))
                else:
                    st.write("No hashtags detected in dataset.")

            with c3:
                # sentiment histogram
                st.write("Sentiment polarity distribution")
                st.histogram = plt.figure(figsize=(4,3))
                plt.hist(df['sentiment'], bins=20)
                st.pyplot(plt)

            # network graph: username <-> hashtag edges
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

            # allow download of results
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown("### Download analysis results")
            href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

# -----------------------------
# Footer / notes
# -----------------------------
st.markdown("---")
st.markdown("**Notes & next steps:**")
st.markdown("""
- This is a prototype: keyword matching + simple sentiment is used for early detection.
- For production: replace/augment sentiment with transformer models, add rate-limited API ingestors, store in DB, and implement real-time alerts (webhooks/email).
- Always analyze only public content and respect ToS & robots.txt.
""")
