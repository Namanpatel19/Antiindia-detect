import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import google.generativeai as genai
import praw
import re
from collections import Counter
import io

# =============================
# Load API Keys from st.secrets
# =============================
GEMINI_API_KEY = st.secrets["api_keys"]["gemini"]
REDDIT_CLIENT_ID = st.secrets["reddit"]["client_id"]
REDDIT_CLIENT_SECRET = st.secrets["reddit"]["client_secret"]
REDDIT_USER_AGENT = st.secrets["reddit"]["user_agent"]

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# =============================
# Keyword Bank
# =============================
ANTI_INDIA_KEYWORDS = [
    "anti-india", "down with india", "boycott india", "hate india",
    "india terrorist", "destroy india", "free kashmir", "india go back"
]

# =============================
# Utility Functions
# =============================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_keywords(text):
    hits = [kw for kw in ANTI_INDIA_KEYWORDS if kw.lower() in text.lower()]
    return hits

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def ai_context_check(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Classify if this text is anti-India: {text}")
        return response.text
    except Exception as e:
        return f"AI check failed: {e}"

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    buf = io.BytesIO()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(buf, format="png")
    st.image(buf)

# =============================
# Scanners
# =============================
def scan_text_input(text):
    text = clean_text(text)
    hits = analyze_keywords(text)
    sentiment = analyze_sentiment(text)
    ai_result = ai_context_check(text)
    return text, hits, sentiment, ai_result

def scan_url(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        text = clean_text(soup.get_text())
        return scan_text_input(text)
    except Exception as e:
        return "", [], 0, f"Error fetching URL: {e}"

def scan_file(uploaded_file):
    try:
        text = uploaded_file.read().decode("utf-8")
        return scan_text_input(text)
    except Exception as e:
        return "", [], 0, f"Error reading file: {e}"

def scan_reddit(subreddit_name, limit=5):
    posts_data = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit):
            text, hits, sentiment, ai_result = scan_text_input(post.title + " " + (post.selftext or ""))
            posts_data.append({
                "title": post.title,
                "hits": hits,
                "sentiment": sentiment,
                "ai_result": ai_result
            })
    except Exception as e:
        st.error(f"Reddit scan failed: {e}")
    return posts_data

# =============================
# Risk Meter
# =============================
def risk_score(hits, sentiment):
    score = len(hits) * 30
    if sentiment < -0.3:
        score += 20
    return min(score, 100)

def show_risk_meter(score):
    st.metric(label="⚠️ Risk Level", value=f"{score}%")
    if score > 70:
        st.error("🚨 High Risk: Anti-India content detected")
    elif score > 40:
        st.warning("⚠️ Medium Risk: Possible harmful content")
    else:
        st.success("✅ Low Risk")

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Anti-India Content Detector", layout="wide")

st.sidebar.title("🛡️ Content Scanner")
menu = st.sidebar.radio("Choose an option", [
    "Home", "Text Scan", "URL Scan", "File Upload", "Reddit Scan", "Image Scan (Future)", "Video Scan (Future)"])

st.title("🇮🇳 AI-Powered Anti-India Content Detector")
st.markdown("---")

# =============================
# HOME
# =============================
if menu == "Home":
    st.subheader("Welcome to the AI Anti-India Content Detection System")
    st.write("This tool detects and flags harmful anti-India content using:")
    st.markdown("""
    - Keyword scanning
    - Sentiment analysis
    - AI Context Detection (Gemini)
    - Wordcloud visualization
    - Reddit & URL integration
    - Future: Image & Video Meme scanning
    """)

# =============================
# TEXT SCAN
# =============================
elif menu == "Text Scan":
    st.subheader("📝 Enter Text")
    text = st.text_area("Paste any text to analyze:")
    if st.button("Analyze Text") and text:
        full_text, hits, sentiment, ai_result = scan_text_input(text)
        st.write("**Keyword Hits:**", hits)
        st.write("**Sentiment:**", sentiment)
        st.write("**AI Result:**", ai_result)
        score = risk_score(hits, sentiment)
        show_risk_meter(score)
        generate_wordcloud(full_text)

# =============================
# URL SCAN
# =============================
elif menu == "URL Scan":
    st.subheader("🔗 Enter Website URL")
    url = st.text_input("Paste a website link:")
    if st.button("Scan URL") and url:
        full_text, hits, sentiment, ai_result = scan_url(url)
        st.write("**Keyword Hits:**", hits)
        st.write("**Sentiment:**", sentiment)
        st.write("**AI Result:**", ai_result)
        score = risk_score(hits, sentiment)
        show_risk_meter(score)
        if full_text:
            generate_wordcloud(full_text)

# =============================
# FILE UPLOAD
# =============================
elif menu == "File Upload":
    st.subheader("📂 Upload a Text File")
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])
    if uploaded_file:
        full_text, hits, sentiment, ai_result = scan_file(uploaded_file)
        st.write("**Keyword Hits:**", hits)
        st.write("**Sentiment:**", sentiment)
        st.write("**AI Result:**", ai_result)
        score = risk_score(hits, sentiment)
        show_risk_meter(score)
        if full_text:
            generate_wordcloud(full_text)

# =============================
# REDDIT SCAN
# =============================
elif menu == "Reddit Scan":
    st.subheader("📡 Scan Reddit Subreddit")
    subreddit_name = st.text_input("Enter subreddit (e.g., worldnews)")
    limit = st.slider("Number of posts", 1, 20, 5)
    if st.button("Scan Subreddit") and subreddit_name:
        results = scan_reddit(subreddit_name, limit)
        for r in results:
            st.markdown(f"**Post:** {r['title']}")
            st.write("Keyword Hits:", r["hits"])
            st.write("Sentiment:", r["sentiment"])
            st.write("AI Result:", r["ai_result"])
            st.markdown("---")

# =============================
# FUTURE PLACEHOLDERS
# =============================
elif menu == "Image Scan (Future)":
    st.subheader("🖼️ Image/Meme Scanning (Coming Soon)")
    st.info("This will allow AI to detect Anti-India content hidden in memes & images.")

elif menu == "Video Scan (Future)":
    st.subheader("🎥 Video/Meme Scanning (Coming Soon)")
    st.info("This will extend detection to videos and viral content analysis.")
