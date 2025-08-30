# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import google.generativeai as genai
import praw
from datetime import datetime

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Anti-India Content Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main { background-color: #f9fafc; }
    .title { font-size:36px; font-weight:700; color:#1a237e; }
    .subtitle { font-size:18px; color:#424242; margin-bottom:20px; }
    .stMetric { background: #ffffff; border-radius: 12px; padding: 10px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------------
# CONFIG: Gemini API + Reddit
# -----------------------------
GEMINI_API_KEY = st.secrets["gemini"]  # keep your API key in secrets
genai.configure(api_key=GEMINI_API_KEY)

# Reddit API credentials from secrets
reddit = praw.Reddit(
    client_id=st.secrets["reddit"]["client_id"],
    client_secret=st.secrets["reddit"]["client_secret"],
    user_agent=st.secrets["reddit"]["user_agent"],
)

# -----------------------------
# Hardcoded Keyword List
# -----------------------------
ANTI_INDIA_KEYWORDS = [
    "boycott india", "down with india", "anti-india", "khalistan",
    "terrorist", "attack", "hate india", "break india", "india is evil",
    "pakistan zindabad", "separate kashmir", "destroy india",
    "burn indian flag", "kill indians", "modi murderer", "rss terrorist",
    "islamophobia india", "down with modi", "ban india", "anti-hindu"
]

# -----------------------------
# Utility Functions
# -----------------------------
def keyword_hits(text, keywords):
    hits = {}
    for kw in keywords:
        c = text.lower().count(kw.lower())
        if c > 0:
            hits[kw] = c
    return hits, list(hits.keys())

def sentiment_score(text):
    return TextBlob(text).sentiment.polarity

def compute_risk(keywords_found, sentiment):
    risk = 0
    if len(keywords_found) > 0:
        risk += len(keywords_found) * 2
    if sentiment < -0.3:
        risk += 3
    elif sentiment < 0:
        risk += 1
    return min(risk, 10)

def badge_html(risk):
    color = "green"
    label = "Safe"
    if risk >= 7:
        color = "red"; label = "High Risk"
    elif risk >= 4:
        color = "orange"; label = "Moderate"
    return f"<span style='background:{color};color:white;padding:3px 6px;border-radius:6px'>{label} ({risk})</span>"

def highlight_sentences(text, keywords_found):
    sentences = text.split(".")
    highlights = []
    for s in sentences:
        for kw in keywords_found:
            if kw.lower() in s.lower():
                highlights.append(s.strip())
    return highlights

def call_gemini_classify(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Classify this text as Anti-India, Neutral, or Safe. Text: {text[:1000]}"
        resp = model.generate_content(prompt)
        return resp.text, "AI classification successful"
    except Exception as e:
        return None, str(e)

def risk_gauge(risk):
    fig, ax = plt.subplots(figsize=(3, 1.5))
    ax.barh([0], [risk], color="red" if risk > 6 else "orange" if risk > 3 else "green")
    ax.set_xlim(0, 10)
    ax.set_yticks([])
    ax.set_xticks(range(0, 11))
    ax.set_title("Risk Gauge", fontsize=10)
    st.pyplot(fig)

# -----------------------------
# UI Tabs
# -----------------------------
tabs = st.tabs([
    "📊 Dashboard", 
    "🔗 URL Scanner", 
    "📂 File Analysis", 
    "📖 Keyword DB", 
    "🛠 Utilities",
    "🌐 Social Media Scanner"
])

# -----------------------------
# TAB 0: Dashboard
# -----------------------------
with tabs[0]:
    st.markdown("<div class='title'>🇮🇳 Anti-India Content Detector</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-powered risk detection for text, files, websites, and social media</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scans", "128")
    with col2:
        st.metric("Flagged Risks", "37")
    with col3:
        st.metric("Safe Content", "91")

    st.info("Use the tabs above to scan URLs, analyze files, or monitor Reddit for potential Anti-India content.")

# -----------------------------
# TAB 1: URL Scanner
# -----------------------------
with tabs[1]:
    st.subheader("🔗 URL Scanner")
    url = st.text_input("Enter website URL")
    if st.button("Scan URL"):
        if url:
            try:
                r = requests.get(url, timeout=10)
                soup = BeautifulSoup(r.text, "html.parser")
                txt = soup.get_text(separator=" ")
                hits, ks = keyword_hits(txt, ANTI_INDIA_KEYWORDS)
                sent = sentiment_score(txt)
                risk = compute_risk(ks, sent)
                st.markdown(badge_html(risk), unsafe_allow_html=True)
                risk_gauge(risk)
                st.write("Keyword Hits:", hits)
                st.write("Sentiment:", sent)
                if hits:
                    wc = WordCloud(width=600, height=400, background_color="white").generate(" ".join(ks))
                    st.image(wc.to_array(), caption="Flagged Keywords WordCloud")
                ai_label, ai_expl = call_gemini_classify(txt)
                st.info(f"AI: {ai_label}")
                st.caption(ai_expl)
            except Exception as e:
                st.error(f"Error scanning URL: {e}")

# -----------------------------
# TAB 2: File Analysis
# -----------------------------
with tabs[2]:
    st.subheader("📂 File Analysis")
    file = st.file_uploader("Upload a text file", type=["txt"])
    if file:
        txt = file.read().decode("utf-8")
        hits, ks = keyword_hits(txt, ANTI_INDIA_KEYWORDS)
        sent = sentiment_score(txt)
        risk = compute_risk(ks, sent)
        st.markdown(badge_html(risk), unsafe_allow_html=True)
        risk_gauge(risk)
        st.write("Keyword Hits:", hits)
        st.write("Sentiment:", sent)
        if hits:
            wc = WordCloud(width=600, height=400, background_color="white").generate(" ".join(ks))
            st.image(wc.to_array(), caption="Flagged Keywords WordCloud")
        ai_label, ai_expl = call_gemini_classify(txt)
        st.info(f"AI: {ai_label}")
        st.caption(ai_expl)
        highlights = highlight_sentences(txt, ks)
        if highlights:
            with st.expander("🔎 Suspicious Sentences"):
                for s in highlights:
                    st.markdown(f"- {s}")

# -----------------------------
# TAB 3: Keyword DB
# -----------------------------
with tabs[3]:
    st.subheader("📖 Keyword Database")
    st.write("Loaded Anti-India keywords (hardcoded in app).")
    st.write(ANTI_INDIA_KEYWORDS)

# -----------------------------
# TAB 4: Utilities
# -----------------------------
with tabs[4]:
    st.subheader("🛠 Utilities")
    st.write("Future tools will be added here.")

# -----------------------------
# TAB 5: Social Media Scanner (Reddit)
# -----------------------------
with tabs[5]:
    st.subheader("🌐 Social Media Scanner (Reddit)")

    subreddit_name = st.text_input("Enter subreddit (e.g. india, worldnews)")
    limit_posts = st.slider("Number of posts", 5, 50, 10)

    if st.button("Scan Subreddit"):
        if not subreddit_name:
            st.warning("Enter a subreddit name.")
        else:
            try:
                posts = reddit.subreddit(subreddit_name).hot(limit=limit_posts)
                results = []
                for post in posts:
                    txt = f"{post.title}\n\n{post.selftext}"
                    hits, ks = keyword_hits(txt, ANTI_INDIA_KEYWORDS)
                    sent = sentiment_score(txt)
                    risk = compute_risk(ks, sent)
                    if risk >= 4:  # only call Gemini for moderate/high
                        ai_label, ai_expl = call_gemini_classify(txt)
                    else:
                        ai_label, ai_expl = "Safe", "Skipped Gemini (low risk)"
                    results.append({
                        "title": post.title,
                        "url": f"https://reddit.com{post.permalink}",
                        "hits": hits,
                        "risk": risk,
                        "sentiment": sent,
                        "ai_label": ai_label,
                        "ai_expl": ai_expl,
                    })

                st.markdown("### Results")
                df = pd.DataFrame(results)
                for r in results:
                    st.markdown("---")
                    st.markdown(f"**[{r['title']}]({r['url']})**")
                    st.markdown(badge_html(r["risk"]), unsafe_allow_html=True)
                    risk_gauge(r["risk"])
                    st.write("Keyword Hits:", r["hits"] or "None")
                    st.write("Sentiment:", r["sentiment"])
                    if r["ai_label"]:
                        if r["ai_label"].startswith("Anti-India"):
                            st.error(f"AI: {r['ai_label']}")
                        elif r["ai_label"].startswith("Safe"):
                            st.success(f"AI: {r['ai_label']}")
                        else:
                            st.info(f"AI: {r['ai_label']}")
                        st.caption(r["ai_expl"])
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Report (CSV)", csv, "reddit_scan_report.csv", "text/csv")
            except Exception as e:
                st.error(f"Reddit API error: {e}")
