# app.py
"""
Anti-India Campaign Detection — Cybersecurity Dashboard Style
"""

import os, re, yaml, requests, base64, json
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

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
    </style>
    """, unsafe_allow_html=True
)

# -----------------------------
# Keyword DB (used in File Analysis)
# -----------------------------
KEYWORD_FILE = "keywords.yaml"
DEFAULT_KEYWORDS = [
    {"term": "boycott india", "type": "phrase", "language": "English", "weight": 4},
    {"term": "#freekashmir", "type": "hashtag", "language": "English", "weight": 5},
    {"term": "down with india", "type": "phrase", "language": "English", "weight": 4},
    {"term": "anti-india", "type": "keyword", "language": "English", "weight": 3},
    {"term": "destroy india", "type": "phrase", "language": "English", "weight": 5},
    {"term": "traitor india", "type": "phrase", "language": "English", "weight": 3},
]

def ensure_keyword_file():
    """Create keywords.yaml if it doesn't exist."""
    if not os.path.exists(KEYWORD_FILE):
        with open(KEYWORD_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_KEYWORDS, f, allow_unicode=True)

def load_keywords():
    """Load and migrate old keywords to new schema (lang → language)."""
    if os.path.exists(KEYWORD_FILE):
        with open(KEYWORD_FILE, "r", encoding="utf-8") as f:
            kws = yaml.safe_load(f) or DEFAULT_KEYWORDS
    else:
        kws = DEFAULT_KEYWORDS

    # --- MIGRATION step ---
    migrated = []
    for kw in kws:
        if "lang" in kw:   # old schema
            kw["language"] = kw.pop("lang")
        if "language" not in kw:
            kw["language"] = "English"
        if "type" not in kw:
            kw["type"] = "phrase"
        migrated.append(kw)

    save_keywords(migrated)  # overwrite file with new schema
    return migrated

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
def extract_content_from_url(url, timeout=10):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = " ".join([p.get_text(" ", strip=True) for p in soup.find_all(["p","h1","h2","h3","li"])])
        images = [img["src"] for img in soup.find_all("img") if img.get("src")]
        from urllib.parse import urljoin
        images = [urljoin(url, i) for i in images]
        return text, images
    except:
        return "", []

# -----------------------------
# Gemini Vision call (text + image support)
# -----------------------------
def call_gemini_content_check(text, image_urls=[], include_images=False):
    if not GEMINI_API_KEY:
        return ("NoKey", "Gemini API key not set.")

    prompt = """
    Detect if this content (text and/or images) contains Anti-India propaganda, 
    hate speech, or coordinated campaigns.
    Language may be English, Hindi, Urdu, or Arabic.
    Reply strictly in JSON with fields: "label" and "explanation".
    """

    parts = [{"text": prompt}]
    if text.strip():
        parts.append({"text": text[:2000]})

    # include images only if requested
    if include_images:
        for img in image_urls[:3]:  # scan up to 3 images
            try:
                img_data = requests.get(img, timeout=10).content
                b64_data = base64.b64encode(img_data).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_data
                    }
                })
            except:
                continue

    payload = {"contents":[{"parts": parts}]}
    headers = {"Content-Type":"application/json","X-goog-api-key":GEMINI_API_KEY}

    try:
        r = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=40)
        if r.status_code != 200:
            return ("Error", r.text[:200])
        j = r.json()
        ai_text = j.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
        try:
            data = json.loads(ai_text)
            return (data.get("label","Unknown"), data.get("explanation",""))
        except:
            return ("Result", ai_text)
    except Exception as e:
        return ("Error", str(e))

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center'>🛡️ Anti-India Campaign Detection Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

# -----------------------------
# TABS
# -----------------------------
tabs = st.tabs(["📊 Dashboard","🔍 URL Scanner","📂 File Analysis","📝 Keyword DB"])

# -----------------------------
# TAB: Dashboard
# -----------------------------
with tabs[0]:
    col1,col2,col3,col4=st.columns(4)
    col1.markdown(f"<div class='metric-card'><h3>{len(keywords)}</h3><p>Keywords</p></div>",unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>29</h3><p>High-Risk Events</p></div>",unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3>08</h3><p>Risky Activities</p></div>",unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><h3>06</h3><p>High-Risk Users</p></div>",unsafe_allow_html=True)

# -----------------------------
# TAB: URL Scanner
# -----------------------------
with tabs[1]:
    st.subheader("Scan URLs with AI (Text + Optional Images)")
    url_input = st.text_input("Enter URL(s), comma-separated")
    include_images = st.checkbox("🔎 Include images in scan", value=False)

    if st.button("Scan"):
        for u in [x.strip() for x in url_input.split(",") if x.strip()]:
            text, images = extract_content_from_url(u)
            ai_label, ai_expl = call_gemini_content_check(text, images, include_images)

            st.markdown("---")
            st.markdown(f"**{u}**")
            if "Anti" in ai_label or "Detected" in ai_label:
                st.error(f"🚨 {ai_label}\n\n{ai_expl}")
            else:
                st.success(f"✅ {ai_label}\n\n{ai_expl}")

            if include_images and images:
                st.info(f"📷 {min(len(images),3)} image(s) were also analyzed for propaganda content.")

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
            lbl,expl=call_gemini_content_check(t, [], False)
            recs.append({"text":t[:80],"ai":lbl,"explanation":expl})
        st.dataframe(pd.DataFrame(recs))

# -----------------------------
# TAB: Keyword DB
# -----------------------------
with tabs[3]:
    st.subheader("Manage Keywords")

    df_kw = pd.DataFrame(keywords)
    st.dataframe(df_kw)

    # Add keyword
    new = st.text_input("New keyword") 
    if st.button("Add Keyword"):
        if "india" in new.lower():
            keywords.append({"term": new, "type":"phrase", "language":"English", "weight":3})
            save_keywords(keywords)
            st.success("✅ Keyword added! Refresh to see changes")

    # Delete multiple keywords
    st.write("### Delete Keywords")
    to_delete = st.multiselect("Select keywords to delete", [kw["term"] for kw in keywords])
    if st.button("Delete Selected"):
        keywords = [kw for kw in keywords if kw["term"] not in to_delete]
        save_keywords(keywords)
        st.success("🗑️ Selected keywords deleted! Refresh to see changes")
