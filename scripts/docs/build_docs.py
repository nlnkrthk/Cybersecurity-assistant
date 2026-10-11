#!/usr/bin/env python3
"""
Build the submission PDFs from their Markdown sources.

    python scripts/docs/build_docs.py                 # all documents
    python scripts/docs/build_docs.py design api      # selected documents

| Target        | Source                                  | Output                                   |
|---------------|-----------------------------------------|------------------------------------------|
| design        | design/Design_Document.md               | design/Design_Document.pdf               |
| api           | docs/API_Documentation.md (generated)   | docs/API_Documentation.pdf               |
| dataset       | docs/Dataset_Details.md                 | docs/Dataset_Details.pdf                 |
| evaluation    | docs/Evaluation_Report.md               | docs/Evaluation_Report.pdf               |
| architecture  | docs/diagrams/*.png                     | architecture/Architecture_Diagram.pdf    |

docs/API_Documentation.md is produced by scripts/docs/generate_api_docs.py (it calls the real API).
Markdown is rendered with markdown-it-py and printed to PDF by headless Microsoft Edge or Google
Chrome, so tables, images and page numbers match the browser rendering.
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

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = {
    "design": (ROOT / "design" / "Design_Document.md", ROOT / "design" / "Design_Document.pdf"),
    "api": (ROOT / "docs" / "API_Documentation.md", ROOT / "docs" / "API_Documentation.pdf"),
    "dataset": (ROOT / "docs" / "Dataset_Details.md", ROOT / "docs" / "Dataset_Details.pdf"),
    "evaluation": (ROOT / "docs" / "Evaluation_Report.md", ROOT / "docs" / "Evaluation_Report.pdf"),
}
ARCHITECTURE_PDF = ROOT / "architecture" / "Architecture_Diagram.pdf"
DIAGRAMS = [
    ("data_flow.png", "Data flow",
     "Dataset → preprocessing → chunking → embeddings → vector database → retriever → LLM / RAG → agents → "
     "API → user interface, with the machine-learning track and the feedback loop."),
    ("architecture.png", "System architecture",
     "The six layers: Streamlit front end, FastAPI backend, LangGraph multi-agent assistant with the LLM "
     "provider, tools and services, storage, and offline pipelines."),
    ("agent_workflow.png", "Agent workflow",
     "One assistant turn: input guard, supervisor routing, specialist agents and tools, follow-up question or "
     "final answer, output guard, optional LLM-as-Judge."),
]
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chromium-browser",
]

BASE_CSS = """
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #0F172A; margin: 0; }
"""

DOC_CSS = BASE_CSS + """
@page { size: A4; margin: 18mm 17mm 18mm 17mm;
        @bottom-center { content: counter(page); font: 9pt Arial, sans-serif; color: #64748B; } }
@page :first { @bottom-center { content: none; } }
.cover { height: 240mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
.cover .kicker { color: #2563EB; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: 10pt; }
.cover h1.title { font-size: 28pt; line-height: 1.15; margin: 8mm 0 4mm 0; border: none; padding: 0;
                  page-break-before: avoid; break-before: avoid; color: #0F172A; }
.cover .subtitle { font-size: 16pt; color: #334155; margin-bottom: 14mm; }
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
h3 { font-size: 11pt; margin: 4.5mm 0 1.5mm 0; page-break-after: avoid; }
h4 { font-size: 10.2pt; margin: 3mm 0 1mm 0; color: #334155; page-break-after: avoid; }
p { margin: 0 0 2.6mm 0; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 6mm; }
li { margin: .6mm 0; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm 0; font-size: 8.9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #EFF6FF; color: #1E3A8A; text-align: left; font-weight: 650; }
th, td { border: 1px solid #CBD5E1; padding: 1.5mm 2mm; vertical-align: top; }
tbody tr:nth-child(even) td { background: #F8FAFC; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt; background: #F1F5F9; padding: 0 1mm;
       border-radius: 1mm; overflow-wrap: anywhere; }
pre { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 2mm; padding: 3mm; font-size: 8pt;
      line-height: 1.35; white-space: pre-wrap; overflow-wrap: anywhere; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { margin: 2mm 0 3mm 0; padding: 2mm 4mm; border-left: 3px solid #93C5FD; background: #F8FAFC;
             color: #334155; }
img { max-width: 100%; display: block; margin: 2mm auto 4mm auto; page-break-inside: avoid; }
strong { font-weight: 650; }
"""

ARCH_CSS = BASE_CSS + """
@page { size: A4 landscape; margin: 10mm 12mm; }
section { height: 186mm; display: flex; flex-direction: column; page-break-after: always; }
section:last-child { page-break-after: auto; }
.head { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1.5px solid #BFDBFE;
        padding-bottom: 2mm; margin-bottom: 3mm; }
.head h1 { font-size: 16pt; color: #1E3A8A; margin: 0; }
.head .tag { color: #64748B; font-size: 9pt; }
.frame { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; }
.frame img { max-width: 100%; max-height: 100%; object-fit: contain; }
.caption { color: #334155; font-size: 9.5pt; margin-top: 2mm; }
"""


def parse_front_matter(text: str):
    meta = {}
    match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
        text = text[match.end():]
    return meta, text


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def document_html(source: Path) -> str:
    meta, body = parse_front_matter(source.read_text(encoding="utf-8"))
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(body)
    toc, seen = [], set()
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag in ("h1", "h2", "h3"):
            title = tokens[i + 1].content
            anchor = slug(title)
            while anchor in seen:
                anchor += "-x"
            seen.add(anchor)
            token.attrSet("id", anchor)
            if token.tag in ("h1", "h2"):
                toc.append((token.tag, title, anchor))
    content = md.renderer.render(tokens, md.options, {})

    rows = "".join(f"<tr><td>{label}</td><td>{html.escape(meta[key])}</td></tr>"
                   for key, label in (("project", "Project"), ("author", "Author"), ("version", "Version"),
                                      ("date", "Date")) if meta.get(key))
    cover = (f'<section class="cover"><div class="bar"></div>'
             f'<div class="kicker">{html.escape(meta.get("subtitle", ""))}</div>'
             f'<h1 class="title">{html.escape(meta.get("title", ""))}</h1>'
             f'<div class="subtitle">{html.escape(meta.get("tagline", ""))}</div>'
             f'<table class="meta">{rows}</table></section>')
    nav = ""
    if meta.get("toc", "yes") != "no":
        items = "".join(f'<li class="{tag}"><a href="#{anchor}">{html.escape(title)}</a></li>'
                        for tag, title, anchor in toc)
        nav = f'<nav class="toc"><h2>Contents</h2><ol>{items}</ol></nav>'
    title = html.escape(f'{meta.get("title", "")} · {meta.get("subtitle", "")}')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
            f'<style>{DOC_CSS}</style></head><body>{cover}{nav}{content}</body></html>')


def architecture_html() -> str:
    sections = []
    for i, (image, title, caption) in enumerate(DIAGRAMS, start=1):
        src = (ROOT / "docs" / "diagrams" / image).as_uri()
        sections.append(
            f'<section><div class="head"><h1>{i}. {html.escape(title)}</h1>'
            f'<span class="tag">AI-Powered Cybersecurity Incident Response Assistant · Architecture</span></div>'
            f'<div class="frame"><img src="{src}" alt="{html.escape(title)}"></div>'
            f'<div class="caption">{html.escape(caption)}</div></section>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Architecture Diagram</title>'
            f'<style>{ARCH_CSS}</style></head><body>{"".join(sections)}</body></html>')


def find_browser() -> str:
    for candidate in BROWSERS:
        path = shutil.which(candidate) or (candidate if os.path.exists(candidate) else None)
        if path:
            return path
    sys.exit("No Edge or Chrome found to print the PDFs.")


def print_pdf(page_html: str, workdir: Path, target: Path, keep_html: bool = False) -> None:
    """Write the HTML next to its source (so relative image paths resolve) and print it to PDF."""
    page = workdir / f"_{target.stem}.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(page_html, encoding="utf-8")
    try:
        subprocess.run([find_browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=5000",
                        f"--print-to-pdf={target}", page.as_uri()],
                       check=True, capture_output=True, timeout=180)
    finally:
        if not keep_html:
            page.unlink(missing_ok=True)
    print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    targets = [*DOCUMENTS, "architecture"]
    parser = argparse.ArgumentParser(description="Build the submission PDFs.")
    parser.add_argument("targets", nargs="*", choices=targets, help="Documents to build (default: all).")
    parser.add_argument("--keep-html", action="store_true", help="Keep the intermediate HTML files.")
    args = parser.parse_args()
    for target in args.targets or targets:
        if target == "architecture":
            print_pdf(architecture_html(), ARCHITECTURE_PDF.parent, ARCHITECTURE_PDF, args.keep_html)
            continue
        source, output = DOCUMENTS[target]
        if not source.exists():
            print(f"skipped {target}: {source.relative_to(ROOT)} not found")
            continue
        print_pdf(document_html(source), source.parent, output, args.keep_html)


if __name__ == "__main__":
    main()
