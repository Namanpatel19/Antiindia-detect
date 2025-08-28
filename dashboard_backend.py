from pathlib import Path
import json

def build_dashboard_html(df, template_path="dashboard.html", css_path="styles.css", inline_css=True):
    """
    Builds the dashboard HTML by injecting dataframe data.
    - df: pandas DataFrame
    - template_path: path to your dashboard.html
    - css_path: path to your styles.css
    - inline_css: if True, CSS is inlined into <style> tag
    """
    # Convert dataframe to JSON
    data_json = df.to_dict(orient="records")
    data_str = json.dumps(data_json)

    # Load HTML template
    template = Path(template_path).read_text(encoding="utf-8")

    # Inject CSS if required
    if inline_css and Path(css_path).exists():
        css_code = Path(css_path).read_text(encoding="utf-8")
        template = template.replace(
            "</head>", f"<style>{css_code}</style>\n</head>"
        )

    # Replace placeholder {{DATA}} with actual JSON string
    html = template.replace("{{DATA}}", data_str)

    return html
