import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
from dashboard_backend import build_dashboard_html

# --- Sample DataFrame (replace with your real detection results) ---
df = pd.DataFrame([
    {"platform": "X", "text": "Down with India", "risk": "High"},
    {"platform": "Instagram", "text": "Boycott India now!", "risk": "Medium"},
    {"platform": "Reddit", "text": "India is bad", "risk": "High"},
    {"platform": "Threads", "text": "Peaceful unity is better", "risk": "Low"},
])

# --- Build HTML with inline CSS + injected JSON ---
html = build_dashboard_html(df, "dashboard.html", "styles.css", inline_css=True)

# Debug: save file so you can open it in browser and check console if blank
Path("debug_dashboard.html").write_text(html, encoding="utf-8")

# --- Streamlit UI ---
st.set_page_config(page_title="Detection Dashboard", layout="wide")
st.title("🚨 Anti-Campaign Detection Dashboard")

components.html(html, height=900, scrolling=True)
