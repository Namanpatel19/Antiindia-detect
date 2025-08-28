# dashboard_backend.py
"""
Backend helpers for rendering the client-side dashboard.html from app.py.

Usage:
    from dashboard_backend import build_dashboard_html, render_dashboard_in_streamlit

    # df is the DataFrame you prepared in your app (with columns like platform, username, text, keyword_hits, risk)
    html = build_dashboard_html(df)
    # then render using components.html(html, height=900, scrolling=True)
    # OR simply call render_dashboard_in_streamlit(df) inside a Streamlit app.

Functions:
- build_dashboard_html(df, template_path, css_path, inline_css, include_wordcloud)
- render_dashboard_in_streamlit(df, template_path, css_path, inline_css, height)
- generate_wordcloud_image(text, out_path)
"""

import json
import html as html_lib
from pathlib import Path
from wordcloud import WordCloud
import base64
import io

def _safe_json_for_html(obj):
    """
    Convert Python object to JSON string and make it safe to embed in a <script> tag.
    Specifically escape closing tags so browser doesn't prematurely end scripts.
    """
    s = json.dumps(obj, ensure_ascii=False)
    # Escape the sequence "</" which can break out of <script> contexts
    s = s.replace("</", "<\\/")
    return s

def build_dashboard_html(df,
                         template_path: str = "dashboard.html",
                         css_path: str = "styles.css",
                         inline_css: bool = False,
                         include_wordcloud: bool = False,
                         wordcloud_col: str = "text",
                         wordcloud_output: str = "wordcloud.png",
                         top_n: int = 1000):
    """
    Build HTML string with injected DATA placeholder replaced by your DF records.

    Parameters:
    - df: pandas.DataFrame (should be analysis results; lists like keyword_hits will become JSON arrays)
    - template_path: path to dashboard.html (contains {{DATA}})
    - css_path: path to styles.css (if inline_css=True it will inline it)
    - inline_css: if True, read styles.css and insert into <style>...</style> inside HTML (useful to keep single file)
    - include_wordcloud: if True, generate a wordcloud (png) from `wordcloud_col` and embed it as data URI
    - wordcloud_col: column from df used to build wordcloud (joined)
    - wordcloud_output: filename used if you want to save to disk (optional)
    - top_n: maximum number of rows to embed to prevent huge payloads

    Returns:
    - html_with_data: string ready to pass to components.html(...)
    """
    # Lazy import pandas inside function to avoid hard dependency when not used
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas required to use build_dashboard_html") from e

    tpl_path = Path(template_path)
    if not tpl_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    html_text = tpl_path.read_text(encoding="utf-8")

    # optionally inline CSS
    if inline_css:
        css_file = Path(css_path)
        if css_file.exists():
            css_text = css_file.read_text(encoding="utf-8")
            # naive insertion: replace <link rel="stylesheet" href="styles.css"> with <style>...</style>
            html_text = html_text.replace(f'<link rel="stylesheet" href="{css_path}">', f"<style>\n{css_text}\n</style>")
        else:
            # remove link tag if css not found
            html_text = html_text.replace(f'<link rel="stylesheet" href="{css_path}">', "")

    # build records (safely choose top rows)
    if df is None:
        records = []
    else:
        # Ensure keyword_hits is JSON serializable
        tmp = df.fillna("").copy()
        # truncate large text to avoid massive payloads
        if 'text' in tmp.columns:
            tmp['text'] = tmp['text'].astype(str).apply(lambda s: s if len(s) <= 2000 else s[:2000] + "…")
        records = tmp.head(top_n).to_dict(orient="records")

    # Optionally include wordcloud as data URI and place into records as 'wordcloud_image_data'
    if include_wordcloud and not df is None and wordcloud_col in df.columns:
        all_text = " ".join(df[wordcloud_col].astype(str).fillna("").tolist())
        try:
            img_bytes = generate_wordcloud_image_bytes(all_text)
            data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
            # put in a top-level variable accessible from the HTML by searching for KEYWORD 'WORDCLOUD_DATA'
            # We'll simply replace a placeholder if present: {{WORDCLOUD_DATA}}
            html_text = html_text.replace("{{WORDCLOUD_DATA}}", f"'{data_uri}'")
        except Exception:
            html_text = html_text.replace("{{WORDCLOUD_DATA}}", "null")
    else:
        html_text = html_text.replace("{{WORDCLOUD_DATA}}", "null")

    # inject data JSON safely
    safe_json = _safe_json_for_html(records)
    # Replace the placeholder exactly "{{DATA}}"
    if "{{DATA}}" not in html_text:
        # if user used different placeholder, try to be forgiving
        html_text += f"\n<!-- DATA: -->\n<script>const DATA = {safe_json};</script>\n"
    else:
        html_with_data = html_text.replace("{{DATA}}", safe_json)
        return html_with_data

    return html_text

def generate_wordcloud_image_bytes(text, width=900, height=300, max_words=200):
    """
    Generate a wordcloud image and return PNG bytes.
    """
    wc = WordCloud(width=width, height=height, background_color="white", max_words=max_words)
    img = wc.generate(text or " ")
    buf = io.BytesIO()
    img.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

# Convenience function to render directly in Streamlit
def render_dashboard_in_streamlit(df,
                                  template_path: str = "dashboard.html",
                                  css_path: str = "styles.css",
                                  inline_css: bool = False,
                                  include_wordcloud: bool = False,
                                  height: int = 880,
                                  top_n: int = 1000):
    """
    Build HTML and call components.html(...) to render in Streamlit.
    Must be called from a running Streamlit context.
    """
    import streamlit as st
    import streamlit.components.v1 as components

    html = build_dashboard_html(df,
                                template_path=template_path,
                                css_path=css_path,
                                inline_css=inline_css,
                                include_wordcloud=include_wordcloud,
                                top_n=top_n)
    components.html(html, height=height, scrolling=True)
