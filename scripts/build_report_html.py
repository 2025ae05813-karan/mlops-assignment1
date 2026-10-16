"""Convert report/REPORT.md into a single self-contained HTML file (Task 9).

Images are embedded as base64 so the output is one portable file. Open it in a
browser and use "Print → Save as PDF" to produce the submission PDF/DOCX.

Usage:
    python scripts/build_report_html.py
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report" / "REPORT.md"
REPORT_HTML = ROOT / "report" / "REPORT.html"
REPORT_DIR = ROOT / "report"

CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       max-width: 860px; margin: 2rem auto; padding: 0 1.5rem; line-height: 1.55;
       color: #222; }
h1, h2, h3 { color: #1a1a2e; margin-top: 1.6rem; }
h1 { border-bottom: 3px solid #D1495B; padding-bottom: .3rem; }
h2 { border-bottom: 1px solid #ddd; padding-bottom: .2rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .92rem; }
th, td { border: 1px solid #ccc; padding: .4rem .6rem; text-align: left; }
th { background: #f3f3f7; }
img { max-width: 100%; height: auto; display: block; margin: 1rem auto;
      border: 1px solid #eee; border-radius: 6px; }
code { background: #f4f4f4; padding: .1rem .3rem; border-radius: 3px; }
pre { background: #f7f7f9; padding: .8rem; border-radius: 6px; overflow-x: auto; }
blockquote { color: #666; border-left: 4px solid #ddd; margin: 1rem 0; padding: .2rem 1rem; }
@media print { body { margin: 0; } h1, h2 { page-break-after: avoid; } img { page-break-inside: avoid; } }
"""


def embed_images(md_text: str) -> str:
    """Replace ![alt](screenshots/x.png) with base64 data URIs."""
    def repl(match: re.Match) -> str:
        alt, path = match.group(1), match.group(2)
        img_path = (REPORT_DIR / path).resolve()
        if not img_path.exists():
            return match.group(0)
        data = base64.b64encode(img_path.read_bytes()).decode()
        return f"![{alt}](data:image/png;base64,{data})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", repl, md_text)


def main() -> None:
    md_text = REPORT_MD.read_text()
    md_text = embed_images(md_text)
    html_body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "toc", "sane_lists"]
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Heart Disease MLOps Report</title>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )
    REPORT_HTML.write_text(html)
    size_kb = REPORT_HTML.stat().st_size / 1024
    print(f"Wrote {REPORT_HTML} ({size_kb:.0f} KB, images embedded)")
    print("Open it in a browser and use Print -> Save as PDF for submission.")


if __name__ == "__main__":
    main()
