# app.py
import streamlit as st
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

# ================== UI THEME ==================
st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")

st.markdown(
    """
    <style>
    /* Global */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Roboto', sans-serif;
    }

    /* Title */
    .title {
        font-size: 40px;
        font-weight: 700;
        color: #00ff99;
        text-shadow: 0px 0px 12px #00ff99;
    }

    .subtitle {
        font-size: 18px;
        color: #8b949e;
        margin-bottom: 20px;
    }

    /* Metrics */
    .stMetric {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0px 0px 12px rgba(0, 255, 153, 0.2);
        color: #c9d1d9;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        color: #c9d1d9;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff99 !important;
        color: black !important;
        font-weight: bold;
        text-shadow: none;
    }

    /* Risk labels */
    .high-risk {
        background: red;
        color: white;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: bold;
        box-shadow: 0px 0px 10px red;
    }
    .moderate-risk {
        background: orange;
        color: black;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: bold;
        box-shadow: 0px 0px 10px orange;
    }
    .safe-risk {
        background: #00ff99;
        color: black;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: bold;
        box-shadow: 0px 0px 10px #00ff99;
    }
    </style>
    """, unsafe_allow_html=True
)

# ================== TITLE ==================
st.markdown("<h1 class='title'>🛡️ Anti-India Campaign Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Cybersecurity Dashboard – Monitor, Detect, and Visualize Threats</p>", unsafe_allow_html=True)

# ================== INPUT SECTION ==================
st.sidebar.header("🔍 Input Options")
option = st.sidebar.radio("Choose Input Type:", ["Text", "Upload File", "Enter URL", "Image OCR"])

input_text = ""
if option == "Text":
    input_text = st.sidebar.text_area("Paste text here:")
elif option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")
elif option == "Enter URL":
    input_text = st.sidebar.text_input("Enter a URL to scan:")
elif option == "Image OCR":
    st.sidebar.info("📷 OCR scanning can be integrated with Tesseract or EasyOCR here.")

# ================== KEYWORDS ==================
keywords = [
    "anti-india", "boycott india", "down with india", "kashmir liberation", "free kashmir",
    "india terrorist", "modi fascist", "anti hindu", "down with modi", "break india"
]

# ================== ANALYSIS ==================
if input_text:
    words = re.findall(r'\b\w+\b', input_text.lower())
    word_counts = Counter(words)
    matched_keywords = [word for word in words if word in keywords]
    sentiment = TextBlob(input_text).sentiment.polarity

    # Risk Level
    if len(matched_keywords) > 5 or sentiment < -0.5:
        risk_level = "<span class='high-risk'>🚨 HIGH RISK</span>"
    elif len(matched_keywords) > 0 or sentiment < 0:
        risk_level = "<span class='moderate-risk'>⚠️ MODERATE RISK</span>"
    else:
        risk_level = "<span class='safe-risk'>✅ SAFE</span>"

    # ================== DASHBOARD TABS ==================
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "☁️ Wordcloud", "🌐 Network Graph"])

    # --- Overview ---
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Words", len(words))
        col2.metric("Matched Keywords", len(matched_keywords))
        col3.metric("Sentiment Score", round(sentiment, 3))

        st.markdown(f"### Threat Level: {risk_level}", unsafe_allow_html=True)

        st.subheader("Detected Keywords")
        if matched_keywords:
            st.write(", ".join(set(matched_keywords)))
        else:
            st.write("✅ No harmful keywords detected.")

    # --- Wordcloud ---
    with tab2:
        st.subheader("Keyword Cloud")
        wc = WordCloud(width=800, height=400, background_color="black", colormap="RdYlGn").generate(" ".join(words))
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # --- Network Graph ---
    with tab3:
        st.subheader("Keyword Co-occurrence Graph")
        G = nx.Graph()
        for word in matched_keywords:
            G.add_node(word)
        for i in range(len(matched_keywords)-1):
            G.add_edge(matched_keywords[i], matched_keywords[i+1])

        if G.number_of_nodes() > 0:
            fig, ax = plt.subplots(figsize=(6,6))
            nx.draw(G, with_labels=True, node_color="red", edge_color="cyan", font_color="white")
            st.pyplot(fig)
        else:
            st.info("No suspicious connections detected.")
