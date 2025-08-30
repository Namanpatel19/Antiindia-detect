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
    layout="wide"
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

# -----------------------------
# UI Tabs
# -----------------------------
tabs = st.tabs([
    "Dashboard", 
    "URL Scanner", 
    "File Analysis", 
    "Keyword DB", 
    "Utilities",
    "Social Media Scanner"
])

# -----------------------------
# TAB 0: Dashboard
# -----------------------------
with tabs[0]:
    st.title("🇮🇳 Anti-India Content Detector")
    st.write("Detects & analyzes Anti-India content in text, files, websites, and social media (Reddit).")

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
                st.write("Keyword Hits:", hits)
                st.write("Sentiment:", sent)
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
        st.write("Keyword Hits:", hits)
        st.write("Sentiment:", sent)
        ai_label, ai_expl = call_gemini_classify(txt)
        st.info(f"AI: {ai_label}")
        st.caption(ai_expl)
        highlights = highlight_sentences(txt, ks)
        if highlights:
            st.warning("Suspicious sentences found:")
            for s in highlights:
                st.write(s)

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
                    ai_label, ai_expl = call_gemini_classify(txt)
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
                for r in results:
                    st.markdown("---")
                    st.markdown(f"**[{r['title']}]({r['url']})**")
                    st.markdown(badge_html(r["risk"]), unsafe_allow_html=True)
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
            except Exception as e:
                st.error(f"Reddit API error: {e}")
