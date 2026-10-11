#!/usr/bin/env python3
"""
Build design/Design_Document.pdf from design/Design_Document.md.

    python design/build_pdf.py [--keep-html]

Markdown (with a small YAML-style front matter) is rendered to styled HTML with markdown-it-py,
then printed to PDF by a headless Chromium browser (Microsoft Edge or Google Chrome), so tables,
images and page numbers look the same as in the browser.
"""
import argparse
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Design_Document.md"
TARGET = HERE / "Design_Document.pdf"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm;
        @bottom-center { content: counter(page); font: 9pt Arial, sans-serif; color: #64748B; } }
@page :first { @bottom-center { content: none; } }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #0F172A; }
.cover { height: 240mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
.cover .kicker { color: #2563EB; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: 10pt; }
.cover h1.title { font-size: 28pt; line-height: 1.15; margin: 8mm 0 4mm 0; border: none; padding: 0;
                  page-break-before: avoid; break-before: avoid; color: #0F172A; }
.cover .subtitle { font-size: 17pt; color: #334155; margin-bottom: 14mm; }
.cover table.meta { border: none; width: auto; }
.cover table.meta td { border: none; padding: 1.2mm 8mm 1.2mm 0; font-size: 11pt; background: none; }
.cover table.meta td:first-child { color: #64748B; }
.cover .bar { height: 2mm; width: 40mm; background: #2563EB; margin-bottom: 8mm; }
nav.toc { page-break-after: always; }
nav.toc h2 { border: none; }
nav.toc ol { list-style: none; padding-left: 0; margin: 0; }
nav.toc li { margin: 1.4mm 0; }
nav.toc li.h2 { padding-left: 7mm; color: #334155; font-size: 9.6pt; }
nav.toc a { color: inherit; text-decoration: none; }
h1 { font-size: 17pt; color: #1E3A8A; border-bottom: 1.5px solid #BFDBFE; padding-bottom: 2mm; margin: 0 0 4mm 0;
     page-break-before: always; }
h2 { font-size: 12.5pt; color: #1E40AF; margin: 6mm 0 2mm 0; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 4mm 0 1.5mm 0; page-break-after: avoid; }
p { margin: 0 0 2.6mm 0; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 6mm; }
li { margin: .6mm 0; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm 0; font-size: 8.9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #EFF6FF; color: #1E3A8A; text-align: left; font-weight: 650; }
th, td { border: 1px solid #CBD5E1; padding: 1.5mm 2mm; vertical-align: top; }
tbody tr:nth-child(even) td { background: #F8FAFC; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt; background: #F1F5F9; padding: 0 1mm;
       border-radius: 1mm; }
pre { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 2mm; padding: 3mm; font-size: 8.3pt;
      line-height: 1.35; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { margin: 2mm 0 3mm 0; padding: 2mm 4mm; border-left: 3px solid #93C5FD; background: #F8FAFC;
             color: #334155; }
img { max-width: 100%; display: block; margin: 2mm auto 4mm auto; page-break-inside: avoid; }
strong { font-weight: 650; }
"""


def parse_front_matter(text: str):
    meta = {}
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
        text = text[match.end():]
    return meta, text


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_html(source: Path) -> str:
    meta, body = parse_front_matter(source.read_text(encoding="utf-8"))
    md = MarkdownIt("commonmark", {"html": False, "typographer": False}).enable("table")
    tokens = md.parse(body)

    toc = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag in ("h1", "h2"):
            title = tokens[i + 1].content
            anchor = slug(title)
            token.attrSet("id", anchor)
            toc.append((token.tag, title, anchor))
    content = md.renderer.render(tokens, md.options, {})

    rows = "".join(f"<tr><td>{label}</td><td>{html.escape(meta[key])}</td></tr>"
                   for key, label in (("project", "Project"), ("author", "Author"), ("date", "Date")) if meta.get(key))
    cover = (f'<section class="cover"><div class="bar"></div><div class="kicker">{html.escape(meta.get("subtitle", ""))}</div>'
             f'<h1 class="title">{html.escape(meta.get("title", "Design Document"))}</h1>'
             '<div class="subtitle">Investigate, classify, prioritise and respond to network security incidents '
             'with retrieval, machine learning and a multi-agent LLM workflow</div>'
             f'<table class="meta">{rows}</table></section>')
    items = "".join(f'<li class="{tag}"><a href="#{anchor}">{html.escape(title)}</a></li>' for tag, title, anchor in toc)
    nav = f'<nav class="toc"><h2>Contents</h2><ol>{items}</ol></nav>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(meta.get("title", ""))} · '
            f'Design Document</title><style>{CSS}</style></head><body>{cover}{nav}{content}</body></html>')


def find_browser() -> str:
    for candidate in BROWSERS:
        path = shutil.which(candidate) or (candidate if os.path.exists(candidate) else None)
        if path:
            return path
    sys.exit("No Edge or Chrome found to print the PDF.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--keep-html", action="store_true", help="Keep the intermediate HTML next to the PDF.")
    args = parser.parse_args()

    page = HERE / "_Design_Document.html"  # next to the Markdown, so relative image paths resolve
    page.write_text(render_html(SOURCE), encoding="utf-8")
    try:
        subprocess.run([find_browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=5000",
                        f"--print-to-pdf={TARGET}", page.as_uri()],
                       check=True, capture_output=True, timeout=120)
    finally:
        if not args.keep_html:
            page.unlink(missing_ok=True)
    print(f"wrote {TARGET} ({TARGET.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
