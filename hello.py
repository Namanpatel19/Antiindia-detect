# app.py  —  Anti-India Campaign Detection (Upgraded UI/UX, single file)
# ---------------------------------------------------------------
# Features:
# - Modern tabbed UI: Dashboard | URL Scanner | File Analysis | Keyword DB | Settings
# - Color-coded risk tables + filters + search
# - Wordcloud, risk pie, sentiment histogram, hashtag chart, mini network graph
# - Import/Export keywords (YAML/CSV)
# - Optional Gemini classification (REST) with sidebar key override
# - Direct API key fallback (edit DIRECT_GEMINI_API_KEY below)
# ---------------------------------------------------------------

import os, io, re, time, json, base64, yaml
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from bs4 import BeautifulSoup
from datetime import datetime
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob

# -----------------------------
# 🔧 Config + Styling
# -----------------------------
st.set_page_config(page_title="🛡️ Anti-India Campaign Detector", page_icon="🛡️", layout="wide")

CUSTOM_CSS = """
<style>
/* overall polish */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: .3px; }
.stMetric { border-radius: 14px; padding: 8px 10px; background: rgba(127,127,127,.05); }
hr { margin: .25rem 0 1rem 0; }
/* risk badges */
.badge { display:inline-block; padding:.25rem .55rem; border-radius:999px; font-size:.8rem; font-weight:600; }
.badge-low { background:#E8FFF1; color:#0B7B37; border:1px solid #AEE8C4; }
.badge-med { background:#FFF8E6; color:#8A6100; border:1px solid #F1D28A; }
.badge-high{ background:#FFE8E8; color:#8E0000; border:1px solid #F5A3A3; }
.kpill { border-radius: 12px; padding:10px 12px; background: rgba(0,0,0,0.04); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# 🔑 API key handling
# -----------------------------
# If you want to hard-code a key, put it here:
DIRECT_GEMINI_API_KEY = ""  # <- optional: paste your key for quick testing

def get_gemini_api_key():
    # priority: sidebar > st.secrets > env > DIRECT constant
    k = st.session_state.get("sidebar_api_key")
    if k: return k
    try:
        # supports st.secrets["GEMINI_API_KEY"] or st.secrets["api_keys"]["gemini"]
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
        if "api_keys" in st.secrets and "gemini" in st.secrets["api_keys"]:
            return st.secrets["api_keys"]["gemini"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or DIRECT_GEMINI_API_KEY

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# -----------------------------
# 🧩 Keyword DB
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
    except Exception as e:
        st.error(f"Saving keywords failed: {e}")
        return False

ensure_keyword_file()
keywords = load_keywords()

# -----------------------------
# ⚙️ Helpers (cached where useful)
# -----------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

@st.cache_data(show_spinner=False)
def extract_text_from_url(url, timeout=12):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p","h1","h2","h3","li"])]
        return " ".join([t for t in paragraphs if t])
    except Exception:
        return ""

def keyword_hits(text, kw):
    text_l = text.lower()
    hits, strength = [], 0
    for k in kw:
        term = str(k.get("term","")).lower().strip()
        if not term: continue
        weight = int(k.get("weight", 1) or 1)
        if term.startswith("#"):
            tags = re.findall(r"\B#\w+", text_l)
            if term in tags: hits.append(term); strength += weight
        else:
            if re.search(rf"\b{re.escape(term)}\b", text_l):
                hits.append(term); strength += weight
    return sorted(set(hits)), strength

def sentiment_score(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

def compute_risk(keyword_strength, sentiment, engagement_norm=0.0, account_suspicion=0.0):
    k_norm = min(1.0, keyword_strength / 8.0)
    neg = max(0.0, -sentiment)
    w_k, w_e, w_t, w_a = 0.45, 0.2, 0.2, 0.15
    return float(min(1.0, w_k*k_norm + w_e*engagement_norm + w_t*neg + w_a*account_suspicion))

def highlight_sentences(text, hits):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    lh = [h.lower() for h in hits]
    return [s.strip() for s in sentences if any(h in s.lower() for h in lh)]

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

def color_badge(r):
    if r < 0.2:  return f'<span class="badge badge-low">Low · {r*100:.1f}%</span>'
    if r < 0.6:  return f'<span class="badge badge-med">Medium · {r*100:.1f}%</span>'
    return f'<span class="badge badge-high">High · {r*100:.1f}%</span>'

def style_risk(df, col="risk"):
    # returns pandas Styler with risk color scale
    def _color(v):
        if v < 0.2:  return "background-color:#E8FFF1; color:#0B7B37;"
        if v < 0.6:  return "background-color:#FFF8E6; color:#8A6100;"
        return "background-color:#FFE8E8; color:#8E0000;"
    sty = df.style.format({col: "{:.2%}"}).applymap(lambda v: _color(v) if isinstance(v,(int,float,np.floating)) else "")
    return sty

def call_gemini_classify(text, timeout=16):
    """
    Returns (label, explanation).
    Uses simple REST call; expects secrets/env/sidebar/direct key.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        return ("NoKey", "Gemini API key not configured (Settings → API Key).")

    prompt = (
        "Classify the following text for whether it contains anti-India propaganda, calls for boycott/violence, "
        "or coordinated disinformation targeted at India. "
        "Return strict JSON with keys: label ('Safe' or 'Anti-India Detected'), confidence (0-1), explanation.\n\n"
        f"Text:\n{text}"
    )
    payload = {"contents":[{"parts":[{"text": prompt}]}]}
    headers = {"Content-Type":"application/json","X-goog-api-key": api_key}

    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return ("Error", f"API {resp.status_code}: {resp.text[:280]}")
        j = resp.json()
        ai_text = j.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","").strip()
        try:
            parsed = json.loads(ai_text)
            label = parsed.get("label","Unknown")
            conf  = parsed.get("confidence", None)
            expl  = parsed.get("explanation","")
            return (f"{label} ({conf})" if conf is not None else label, expl or ai_text)
        except Exception:
            t = ai_text.lower()
            if "anti-india" in t or "boycott" in t: return ("Anti-India Detected", ai_text)
            if "safe" in t and "anti" not in t:   return ("Safe", ai_text)
            return ("Unknown", ai_text)
    except Exception as e:
        return ("Error", f"Request failed: {e}")

def process_single_text(source_label, text, kw, ai_enabled=False):
    hits, k_strength = keyword_hits(text, kw)
    sent       = sentiment_score(text)
    highlights = highlight_sentences(text, hits)
    risk       = compute_risk(k_strength, sent)

    ai_label, ai_expl = None, None
    if ai_enabled and (k_strength > 0 or risk >= 0.25):
        ai_label, ai_expl = call_gemini_classify(text)
        time.sleep(0.2)  # light throttle

    return {
        "source": source_label,
        "keyword_hits": hits,
        "keyword_strength": k_strength,
        "sentiment": sent,
        "highlights": highlights,
        "risk": risk,
        "raw_text": text,
        "ai_label": ai_label,
        "ai_explanation": ai_expl
    }

# -----------------------------
# 🧭 Sidebar (Settings)
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Tip: Use **API Key** below to enable Gemini AI.")
    st.session_state["sidebar_api_key"] = st.text_input("Gemini API Key", type="password", placeholder="Paste key or leave blank")
    use_ai = st.toggle("Enable Gemini AI classification", value=False, help="Consumes API quota.")
    st.divider()
    st.markdown("**Notes**")
    st.write("• This prototype analyzes public content only.")
    st.write("• Respect robots.txt and platform ToS.")
    st.write("• Risk scores are heuristic and need human review.")

# -----------------------------
# 🧱 Main Tabs
# -----------------------------
st.title("🛡️ Anti-India Campaign Detection — Prototype")
tabs = st.tabs(["📊 Dashboard", "🔗 URL Scanner", "📁 File Analysis", "🗂️ Keyword DB", "🛠️ Utilities"])

# --------------- TAB: Dashboard
with tabs[0]:
    st.subheader("📊 Quick Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Keywords", len(keywords))
    c2.metric("Default Keywords", len(DEFAULT_KEYWORDS))
    c3.metric("AI", "Enabled" if use_ai and get_gemini_api_key() else "Disabled")
    kpreview = ", ".join([k["term"] for k in keywords[:6]]) + ("…" if len(keywords) > 6 else "")
    c4.markdown(f"<div class='kpill'><b>Preview:</b> {kpreview}</div>", unsafe_allow_html=True)

    st.write("Use **URL Scanner** or **File Analysis** to populate insights here.")
    st.markdown("---")
    st.info("Pro tip: Add more phrases/hashtags in **Keyword DB** to improve recall.")

# --------------- TAB: URL Scanner
with tabs[1]:
    st.subheader("🔗 Scan Website(s)")
    url_input = st.text_input("Enter website URL(s), comma-separated")
    run = st.button("🚀 Scan URLs")

    if run and url_input:
        urls = [u.strip() for u in url_input.split(",") if u.strip()]
        results = []
        progress = st.progress(0)
        for i, u in enumerate(urls, start=1):
            txt = extract_text_from_url(u)
            if not txt:
                st.warning(f"Could not extract text from: {u}")
            else:
                results.append(process_single_text(u, txt, keywords, ai_enabled=use_ai))
            progress.progress(i/len(urls))

        for r in results:
            st.markdown("----")
            left, right = st.columns([1,2])
            with left:
                st.markdown(f"**Source**: {r['source']}")
                st.markdown(color_badge(r["risk"]), unsafe_allow_html=True)
                st.write(f"**Sentiment**: {r['sentiment']:.3f}")
                st.write("**Keyword hits:**", r["keyword_hits"] or "None")
                if r['ai_label']:
                    st.info(f"AI: {r['ai_label']}")
                    st.caption(r.get("ai_explanation",""))
            with right:
                if r["highlights"]:
                    st.markdown("**⚠️ Suspicious Sentences:**")
                    for s in r["highlights"]:
                        st.error(s)
                else:
                    st.success("No suspicious sentences highlighted.")

            # wordcloud (best-effort)
            try:
                wc = WordCloud(width=800, height=250, background_color="white").generate(r['raw_text'][:12000])
                fig, ax = plt.subplots(figsize=(9,3))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                st.pyplot(fig)
            except Exception:
                pass

# --------------- TAB: File Analysis
with tabs[2]:
    st.subheader("📁 Upload CSV/JSON")
    st.caption("Expected columns (optional but helpful): platform, username, text, likes, shares, comments, followers, created_at")
    uf = st.file_uploader("Upload file", type=["csv","json"])
    analyze = st.button("📊 Analyze File")

    if analyze and uf:
        try:
            df = pd.read_csv(uf) if uf.name.endswith(".csv") else pd.read_json(uf)
        except Exception as e:
            st.error(f"Read error: {e}")
            df = None

        if df is not None and not df.empty:
            if "text" not in df.columns:
                st.error("File must include a 'text' column.")
            else:
                df['text'] = df['text'].astype(str).fillna("")
                kw_hits, k_strengths, sents, susp, engs, tags_all, ai_labels, ai_expls = [], [], [], [], [], [], [], []

                for _, row in df.iterrows():
                    hits, ks = keyword_hits(row['text'], keywords)
                    s = sentiment_score(row['text'])
                    kw_hits.append(hits); k_strengths.append(ks); sents.append(s)

                    susp.append(account_suspicion_from_row(row))
                    likes = float(row.get("likes", 0) or 0)
                    shares = float(row.get("shares", 0) or 0)
                    comments = float(row.get("comments", 0) or 0)
                    followers = float(row.get("followers", 0) or 0)
                    pe = (likes + 2*shares + 3*comments) / (1 + (followers if followers>0 else 1))
                    engs.append(pe)
                    tags_all.extend(re.findall(r"\B#\w+", row['text'].lower()))

                    if use_ai:
                        approx_risk = compute_risk(ks, s)
                        if ks > 0 or approx_risk >= 0.25:
                            lab, expl = call_gemini_classify(row['text'])
                            ai_labels.append(lab); ai_expls.append(expl)
                            time.sleep(0.15)
                        else:
                            ai_labels.append(None); ai_expls.append(None)
                    else:
                        ai_labels.append(None); ai_expls.append(None)

                max_eng = max(engs) if engs else 1.0
                eng_norm = [e/max_eng if max_eng>0 else 0 for e in engs]
                risks = [compute_risk(k_strengths[i], sents[i], eng_norm[i], susp[i]) for i in range(len(df))]

                df["keyword_hits"] = kw_hits
                df["keyword_strength"] = k_strengths
                df["sentiment"] = sents
                df["eng_norm"] = eng_norm
                df["suspicion"] = susp
                df["risk"] = risks
                df["ai_label"] = ai_labels
                df["ai_explanation"] = ai_expls

                # ---- dashboard quick stats
                st.markdown("### 🔥 Top Flagged Posts")
                view_cols = [c for c in ["platform","username","text","likes","shares","comments","followers","keyword_hits","risk","ai_label"] if c in df.columns or c in ["text","keyword_hits","risk","ai_label"]]
                topk = df.sort_values("risk", ascending=False).head(15)[view_cols].copy()
                st.dataframe(style_risk(topk, "risk"), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    counts = [sum(df['risk']<0.2), sum((df['risk']>=0.2)&(df['risk']<0.6)), sum(df['risk']>=0.6)]
                    plt.figure(figsize=(4,3))
                    plt.pie(counts, labels=["Low","Medium","High"], autopct="%1.1f%%", startangle=90)
                    plt.title("Risk distribution")
                    st.pyplot(plt)
                with c2:
                    st.write("Hashtag frequency")
                    tags_df = pd.DataFrame(Counter(tags_all).most_common(15), columns=["hashtag","count"])
                    if not tags_df.empty:
                        st.bar_chart(data=tags_df.set_index("hashtag"))
                    else:
                        st.caption("No hashtags detected.")
                with c3:
                    plt.figure(figsize=(4,3))
                    plt.hist(df["sentiment"], bins=20)
                    plt.title("Sentiment polarity")
                    st.pyplot(plt)

                # Mini author-hashtag graph
                if "username" in df.columns:
                    G = nx.Graph()
                    for _, row in df.iterrows():
                        uname = str(row.get("username","")).strip()
                        tags = (row.get("keyword_hits") or []) + re.findall(r"\B#\w+", row["text"].lower())
                        for t in tags: G.add_edge(uname, t)
                    if G.number_of_nodes() > 0:
                        st.subheader("Author–Hashtag Network (mini view)")
                        fig, ax = plt.subplots(figsize=(7.5,5))
                        pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)
                        nx.draw(G, pos=pos, with_labels=True, node_size=220, font_size=8)
                        st.pyplot(fig)

                # Filter/search + download
                st.markdown("### 🔎 Explore & Export")
                colf1, colf2, colf3 = st.columns([1,1,1])
                with colf1:
                    rmin, rmax = st.slider("Risk filter", 0.0, 1.0, (0.0, 1.0), step=0.05)
                with colf2:
                    query = st.text_input("Text contains (optional)").strip().lower()
                with colf3:
                    only_ai = st.checkbox("Only rows with AI label")
                fdf = df[(df["risk"]>=rmin)&(df["risk"]<=rmax)].copy()
                if query:
                    fdf = fdf[fdf["text"].str.lower().str.contains(re.escape(query))]
                if only_ai:
                    fdf = fdf[~fdf["ai_label"].isna()]
                st.dataframe(style_risk(fdf.head(200), "risk"), use_container_width=True)
                csv_bytes = fdf.to_csv(index=False).encode()
                st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="analysis_results.csv", mime="text/csv")

# --------------- TAB: Keyword DB
with tabs[3]:
    st.subheader("🗂️ Keyword Manager")
    st.caption("Only allow terms that **contain 'india'** to reduce noise.")
    kcol1, kcol2 = st.columns([2,1])
    with kcol1:
        st.dataframe(pd.DataFrame(keywords), use_container_width=True)

    with kcol2:
        st.markdown("**Add keyword**")
        new_term = st.text_input("Term (must include 'india')", key="k_add_term")
        new_type = st.selectbox("Type", ["phrase","hashtag","keyword","word"], index=0)
        new_lang = st.selectbox("Language", ["en","hi","ur"], index=0)
        new_weight = st.slider("Weight", 1, 10, 3)
        if st.button("➕ Add"):
            if "india" not in (new_term or "").lower():
                st.error("Only terms containing 'india' are allowed.")
            else:
                exists = {k["term"].lower() for k in keywords}
                if new_term.lower() in exists:
                    st.warning("Term already exists.")
                else:
                    keywords.append({"term": new_term, "type": new_type, "lang": new_lang, "weight": int(new_weight)})
                    if save_keywords(keywords):
                        st.success(f"Added: {new_term}")
                        st.rerun()

        st.markdown("---")
        st.markdown("**Delete keywords**")
        del_sel = st.multiselect("Select", [k["term"] for k in keywords])
        if st.button("🗑️ Delete selected"):
            if del_sel:
                newk = [k for k in keywords if k["term"] not in del_sel]
                if save_keywords(newk):
                    st.success("Deleted.")
                    st.rerun()

        st.markdown("---")
        st.markdown("**Import/Export**")
        exp_yaml = yaml.safe_dump(keywords, allow_unicode=True).encode()
        st.download_button("⬇️ Export YAML", data=exp_yaml, file_name="keywords.yaml", mime="text/yaml")
        exp_csv = pd.DataFrame(keywords).to_csv(index=False).encode()
        st.download_button("⬇️ Export CSV", data=exp_csv, file_name="keywords.csv", mime="text/csv")
        imp = st.file_uploader("Import YAML or CSV", type=["yaml","yml","csv"], key="kw_imp")
        if imp and st.button("📥 Import (replace)"):
            try:
                if imp.name.endswith((".yaml",".yml")):
                    data = yaml.safe_load(imp.read()) or []
                    if isinstance(data, list):
                        keywords[:] = data
                    else:
                        st.error("YAML must be a list of objects.")
                else:
                    dfk = pd.read_csv(imp)
                    keywords[:] = dfk.to_dict(orient="records")
                if save_keywords(keywords):
                    st.success("Imported.")
                    st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

# --------------- TAB: Utilities
with tabs[4]:
    st.subheader("🛠️ Utilities & Help")
    st.markdown("""
- **Gemini API**: set key in the **sidebar**. If blank, app skips AI or shows `NoKey`.
- **Risk score** = weighted blend of keyword strength, negative sentiment, engagement proxy, and account suspicion.
- **Export**: use CSV export in File Analysis; keywords export here.
- **Disclaimer**: This tool assists human moderators. Always verify context to avoid false positives.
""")
    st.markdown("---")
    st.write("**Quick risk badge preview:**")
    st.markdown(" ".join([color_badge(x/10) for x in [1,3,7,9]]), unsafe_allow_html=True)
