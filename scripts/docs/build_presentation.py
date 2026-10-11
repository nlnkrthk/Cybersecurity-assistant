#!/usr/bin/env python3
"""
Build presentation/Project_Presentation.pptx (and a PDF copy when PowerPoint is installed).

    python scripts/docs/build_presentation.py [--no-pdf]

A 15-slide, 10-minute deck: title, problem, solution overview, architecture and data flow, dataset,
technology stack, retrieval, ML models, multi-agent workflow, safety and quality, user accounts,
evaluation and performance, challenges and learnings, future work, Q&A. Diagrams are drawn with
native PowerPoint shapes (editable), and every slide has short speaker notes with a time budget.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "presentation" / "Project_Presentation.pptx"
DIAGRAMS = ROOT / "docs" / "diagrams"
REPO_URL = "github.com/nlnkrthk/Cybersecurity-assistant"
FONT = "Segoe UI"

NAVY = RGBColor(0x0B, 0x1B, 0x33)
BLUE = RGBColor(0x25, 0x63, 0xEB)
INK = RGBColor(0x0F, 0x17, 0x2A)
MUTED = RGBColor(0x47, 0x55, 0x69)
LIGHT = RGBColor(0xF8, 0xFA, 0xFC)
LINE = RGBColor(0xCB, 0xD5, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ARROW = RGBColor(0x64, 0x74, 0x8B)
ACCENTS = {
    "blue": (RGBColor(0xEF, 0xF6, 0xFF), RGBColor(0x25, 0x63, 0xEB)),
    "violet": (RGBColor(0xF5, 0xF3, 0xFF), RGBColor(0x7C, 0x3A, 0xED)),
    "green": (RGBColor(0xEC, 0xFD, 0xF5), RGBColor(0x05, 0x96, 0x69)),
    "amber": (RGBColor(0xFF, 0xFB, 0xEB), RGBColor(0xD9, 0x77, 0x06)),
    "pink": (RGBColor(0xFD, 0xF2, 0xF8), RGBColor(0xDB, 0x27, 0x77)),
    "slate": (RGBColor(0xF1, 0xF5, 0xF9), RGBColor(0x64, 0x74, 0x8B)),
    "red": (RGBColor(0xFE, 0xF2, 0xF2), RGBColor(0xDC, 0x26, 0x26)),
}
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


def emu(*values):
    """Whole EMUs: computed lengths (e.g. width / 3) are floats, which PowerPoint rejects in the XML."""
    return [Emu(int(round(v))) for v in values]


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.count = 0

    # ------------------------------------------------------------------ text
    @staticmethod
    def _run(paragraph, text, size, bold=False, color=INK, italic=False):
        run = paragraph.add_run()
        run.text = text
        run.font.name, run.font.size, run.font.bold, run.font.italic = FONT, Pt(size), bold, italic
        run.font.color.rgb = color
        return run

    def text(self, slide, x, y, w, h, lines, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, spacing=6):
        """lines: str or list of str / (text, {size, bold, color, bullet}) tuples."""
        box = slide.shapes.add_textbox(*emu(x, y, w, h))
        frame = box.text_frame
        frame.word_wrap, frame.vertical_anchor = True, anchor
        frame.margin_left = frame.margin_right = Inches(0.05)
        frame.margin_top = frame.margin_bottom = Inches(0.03)
        for i, line in enumerate([lines] if isinstance(lines, str) else lines):
            style = {}
            if isinstance(line, tuple):
                line, style = line
            paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            paragraph.alignment = align
            paragraph.space_after = Pt(style.get("space", spacing))
            if style.get("bullet"):
                self._run(paragraph, "•  ", style.get("size", size), color=style.get("bullet_color", BLUE), bold=True)
            parts = line.split("**")
            for k, part in enumerate(parts):
                if part:
                    self._run(paragraph, part, style.get("size", size), bold=style.get("bold", bold) or k % 2 == 1,
                              color=style.get("color", color), italic=style.get("italic", False))
        return box

    def bullets(self, slide, x, y, w, h, items, size=18, spacing=10):
        return self.text(slide, x, y, w, h, [(item, {"bullet": True}) for item in items], size=size, spacing=spacing)

    # ---------------------------------------------------------------- shapes
    def card(self, slide, x, y, w, h, title, body=(), accent="blue", title_size=17, body_size=13.5, align=PP_ALIGN.LEFT,
             bullets=False):
        fill, stroke = ACCENTS[accent]
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, *emu(x, y, w, h))
        shape.adjustments[0] = 0.07
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        shape.line.color.rgb = stroke
        shape.line.width = Pt(1.25)
        shape.shadow.inherit = False
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, *emu(x + Inches(0.18), y + Inches(0.2), Inches(0.06),
                                                             Inches(0.36)))
        bar.fill.solid()
        bar.fill.fore_color.rgb = stroke
        bar.line.fill.background()
        lines = [(title, {"size": title_size, "bold": True, "space": 6})]
        lines += [(b, {"size": body_size, "color": MUTED, "space": 6 if bullets else 3, "bullet": bullets,
                       "bullet_color": stroke}) for b in body]
        self.text(slide, x + Inches(0.32), y + Inches(0.14), w - Inches(0.45), h - Inches(0.2), lines, align=align)
        return shape

    def stat(self, slide, x, y, w, h, value, label, accent="blue"):
        fill, stroke = ACCENTS[accent]
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, *emu(x, y, w, h))
        shape.adjustments[0] = 0.08
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        shape.line.color.rgb = stroke
        shape.line.width = Pt(1.25)
        shape.shadow.inherit = False
        self.text(slide, x, y + Inches(0.12), w, Inches(0.75), value, size=30, bold=True, color=stroke,
                  align=PP_ALIGN.CENTER)
        self.text(slide, x + Inches(0.1), y + Inches(0.85), w - Inches(0.2), h - Inches(0.9), label, size=12.5,
                  color=MUTED, align=PP_ALIGN.CENTER)

    def table(self, slide, x, y, w, rows, widths, size=13, header_fill=RGBColor(0xEF, 0xF6, 0xFF), row_h=0.42):
        shape = slide.shapes.add_table(len(rows), len(rows[0]), *emu(x, y, w, Inches(row_h * len(rows))))
        table = shape.table
        total = sum(widths)
        for i, width in enumerate(widths):
            table.columns[i].width = Emu(int(w * width / total))
        for r, row in enumerate(rows):
            table.rows[r].height = Inches(row_h)
            for c, value in enumerate(row):
                cell = table.cell(r, c)
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.margin_top = cell.margin_bottom = Inches(0.04)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill if r == 0 else (WHITE if r % 2 else LIGHT)
                frame = cell.text_frame
                frame.word_wrap = True
                paragraph = frame.paragraphs[0]
                parts = str(value).split("**")
                for k, part in enumerate(parts):
                    if part:
                        self._run(paragraph, part, size, bold=(r == 0) or k % 2 == 1,
                                  color=RGBColor(0x1E, 0x3A, 0x8A) if r == 0 else INK)
        return table

    def image(self, slide, path, x, y, max_w, max_h):
        with Image.open(path) as img:
            ratio = img.width / img.height
        w, h = max_w, int(max_w / ratio)
        if h > max_h:
            h, w = max_h, int(max_h * ratio)
        return slide.shapes.add_picture(str(path), *emu(x + (max_w - w) / 2, y + (max_h - h) / 2, w, h))

    # -------------------------------------------------------------- diagrams
    def node(self, slide, x, y, w, h, title, sub=(), accent="blue", title_size=14, sub_size=11.5):
        """A diagram box: bold title and short grey lines, centred."""
        fill, stroke = ACCENTS[accent]
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, *emu(x, y, w, h))
        shape.adjustments[0] = 0.12
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        shape.line.color.rgb = stroke
        shape.line.width = Pt(1.5)
        shape.shadow.inherit = False
        lines = [(title, {"size": title_size, "bold": True, "space": 2})]
        lines += [(line, {"size": sub_size, "color": MUTED, "space": 0}) for line in sub]
        self.text(slide, x + Inches(0.05), y + Inches(0.03), w - Inches(0.1), h - Inches(0.06), lines,
                  align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        return shape

    def arrow(self, slide, points, color=ARROW, both=False, label=None, label_at=None):
        """A line through the points with an arrowhead at the end (and at the start when both)."""
        segments = list(zip(points, points[1:]))
        for i, ((x1, y1), (x2, y2)) in enumerate(segments):
            line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, *emu(x1, y1, x2, y2))
            line.line.color.rgb = color
            line.line.width = Pt(1.75)
            ln = line.line._get_or_add_ln()
            if both and i == 0:
                ln.append(ln.makeelement(qn("a:headEnd"), {"type": "triangle", "w": "med", "len": "med"}))
            if i == len(segments) - 1:
                ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
        if label:
            self.text(slide, label_at[0], label_at[1], Inches(2.6), Inches(0.3), label, size=10.5, color=MUTED)

    def chain(self, slide, x, y, w, h, items, gap=Inches(0.33), title_size=14, sub_size=11.5):
        """Boxes in a row joined by arrows; items are (title, sub lines, accent). Returns (lefts, width)."""
        nw = (w - (len(items) - 1) * gap) / len(items)
        xs = [x + i * (nw + gap) for i in range(len(items))]
        for left, (title, sub, accent) in zip(xs, items):
            self.node(slide, left, y, nw, h, title, sub, accent, title_size=title_size, sub_size=sub_size)
        for left in xs[:-1]:
            self.arrow(slide, [(left + nw, y + h / 2), (left + nw + gap, y + h / 2)])
        return xs, nw

    def group(self, slide, x, y, w, h, label, accent="slate", label_size=11.5):
        """A dashed frame with a small label, grouping diagram boxes."""
        _, stroke = ACCENTS[accent]
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, *emu(x, y, w, h))
        shape.adjustments[0] = 0.04
        shape.fill.background()
        shape.line.color.rgb = stroke
        shape.line.width = Pt(1.25)
        shape.line.dash_style = MSO_LINE.DASH
        shape.shadow.inherit = False
        self.text(slide, x + Inches(0.12), y + Inches(0.05), w - Inches(0.24), Inches(0.3), label, size=label_size,
                  bold=True, color=stroke)
        return shape

    # ---------------------------------------------------------------- slides
    def slide(self, title, kicker="", notes=""):
        self.count += 1
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(0.42), Inches(0.75), Inches(0.07))
        accent.fill.solid()
        accent.fill.fore_color.rgb = BLUE
        accent.line.fill.background()
        if kicker:
            self.text(slide, MARGIN, Inches(0.52), Inches(10), Inches(0.35), kicker.upper(), size=11.5, bold=True,
                      color=BLUE)
        self.text(slide, MARGIN, Inches(0.8), W - 2 * MARGIN, Inches(0.8), title, size=30, bold=True, color=INK)
        footer = "AI-Powered Cybersecurity Incident Response Assistant  ·  Capstone Project 8"
        self.text(slide, MARGIN, H - Inches(0.45), Inches(9), Inches(0.3), footer, size=10, color=MUTED)
        self.text(slide, W - MARGIN - Inches(1), H - Inches(0.45), Inches(1), Inches(0.3), str(self.count), size=10,
                  color=MUTED, align=PP_ALIGN.RIGHT)
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
        return slide

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(path)


def build() -> Deck:
    d = Deck()
    top = Inches(1.75)
    content_w = W - 2 * MARGIN
    I = Inches

    # 1 Title ---------------------------------------------------------------------------------------
    d.count += 1
    s = d.prs.slides.add_slide(d.prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, I(0.9), I(2.0), I(1.1), I(0.09))
    band.fill.solid()
    band.fill.fore_color.rgb = RGBColor(0x60, 0xA5, 0xFA)
    band.line.fill.background()
    d.text(s, I(0.9), I(2.25), I(11.5), I(1.9), "AI-Powered Cybersecurity\nIncident Response Assistant",
           size=44, bold=True, color=WHITE, spacing=0)
    d.text(s, I(0.9), I(4.15), I(11), I(0.9),
           "Investigate, classify, prioritise and respond to network security incidents with hybrid RAG, "
           "machine learning and a multi-agent LLM workflow", size=19, color=RGBColor(0xCB, 0xD5, 0xE1))
    d.text(s, I(0.9), I(5.6), I(11), I(0.9),
           [("Capstone Project 8  ·  CIC-IDS2017", {"size": 15, "color": RGBColor(0x93, 0xC5, 0xFD), "bold": True}),
            ("Nalin  ·  October 2026", {"size": 15, "color": RGBColor(0xE5, 0xEC, 0xF6)})])
    s.notes_slide.notes_text_frame.text = (
        "[0:00–0:20] One sentence: an AI assistant that helps security analysts investigate, classify, "
        "prioritise and respond to network incidents, built on CIC-IDS2017 and running locally.")

    # 2 Problem statement ----------------------------------------------------------------------------
    s = d.slide("Analysts drown in network alerts", "Problem statement", notes=(
        "[0:20–0:55] SOC teams see many alert types. Each alert needs the same four answers, and finding them "
        "by hand across millions of flow records is slow and inconsistent, so urgent incidents can wait."))
    d.bullets(s, MARGIN, top, I(6.6), I(4.8), [
        "Security teams monitor traffic for **brute force, DoS / DDoS, web attacks, botnets, infiltration "
        "and port scans**",
        "Every alert needs the same four answers (right)",
        "Finding them by hand across **millions of flow records** is slow and inconsistent",
        "Urgent incidents can **wait behind routine ones**",
        "**Goal:** cut the time from alert to decision, with answers analysts can trust",
    ], size=17, spacing=12)
    questions = [("What attack is it?", "attack type and confidence", "blue"),
                 ("How dangerous is it?", "risk, anomaly and P1–P4 priority", "amber"),
                 ("Has it happened before?", "similar past incidents", "violet"),
                 ("What should we do?", "investigation and response steps", "green")]
    qw, qh = I(2.5), I(1.55)
    for i, (title, sub, accent) in enumerate(questions):
        d.node(s, I(7.6) + (i % 2) * (qw + I(0.13)), top + (i // 2) * (qh + I(0.2)), qw, qh, title, [sub],
               accent, title_size=16, sub_size=12.5)
    d.text(s, I(7.6), top + I(3.65), I(5.13), I(0.8),
           "Data: CIC-IDS2017, 2.8 million labelled network flows, 15 traffic types", size=13, color=MUTED,
           align=PP_ALIGN.CENTER)

    # 3 Solution overview ---------------------------------------------------------------------------
    s = d.slide("One question in, an evidence-based answer out", "Solution overview", notes=(
        "[0:55–1:35] Walk the flow: the question is safety-checked, the supervisor picks the agents, the "
        "agents call real models and search, and the answer comes back checked and cited. Then the six "
        "capabilities."))
    flow = [("Analyst question", ["chat or flow metrics"], "slate"), ("Safety check", ["input guard"], "amber"),
            ("Supervisor", ["picks a route"], "violet"), ("Specialist agents", ["4 agents call tools"], "violet"),
            ("Tools", ["ML models, hybrid search"], "green"), ("Answer", ["checked, cited, graded"], "blue")]
    d.chain(s, MARGIN, I(1.85), content_w, I(1.0), flow)
    caps = [("Hybrid RAG search", "97,549 past incidents: keyword + semantic search, fused, with exact filters",
             "blue"),
            ("ML classification & risk", "Attack type (15 classes), intrusion risk, anomaly score", "green"),
            ("Explainable decisions", "SHAP shows which metrics drove each prediction, vs normal traffic", "green"),
            ("Prioritisation", "Risk + severity + anomaly → P1–P4, for one flow, a batch or a CSV upload", "amber"),
            ("Multi-agent investigation", "LangGraph supervisor + 4 specialist agents, some running in parallel",
             "violet"),
            ("Safety & quality", "Guardrails, LLM-as-Judge, DeepEval, analyst feedback and safe retraining",
             "pink")]
    cw, ch = (content_w - I(0.6)) / 3, I(1.55)
    for i, (title, body, accent) in enumerate(caps):
        d.card(s, MARGIN + (i % 3) * (cw + I(0.3)), I(3.3) + (i // 3) * (ch + I(0.2)), cw, ch, title, [body], accent,
               title_size=16, body_size=13)

    # 4 Architecture and data flow ------------------------------------------------------------------
    s = d.slide("Architecture and data flow", "How the parts fit together", notes=(
        "[1:35–2:25] Top row runs once: the dataset is cleaned with PySpark, each flow becomes an incident "
        "text, and it is embedded into ChromaDB alongside a BM25 index. Bottom row runs per question: the "
        "Streamlit UI calls only the FastAPI backend, which runs the LangGraph agents; the agents call the "
        "ML service, prioritiser and hybrid retrieval. The LLM is one .env setting. SQLite keeps accounts, "
        "saved chats and feedback."))
    d.group(s, MARGIN, I(1.7), content_w, I(1.65), "OFFLINE · build the incident library once", "slate")
    offline = [("Dataset", ["CIC-IDS2017, 8 CSV files"], "slate"),
               ("Preprocessing", ["PySpark: clean, de-duplicate"], "slate"),
               ("Chunking", ["97,549 incidents, 1 flow = 1 text"], "blue"),
               ("Embeddings", ["all-MiniLM-L6-v2, 384-d"], "blue"),
               ("Vector DB + BM25", ["ChromaDB + keyword index"], "blue")]
    oxs, onw = d.chain(s, MARGIN + I(0.2), I(2.1), content_w - I(0.4), I(1.0), offline, title_size=13.5,
                       sub_size=11.5)
    d.group(s, MARGIN, I(3.55), content_w, I(3.25), "ONLINE · answer a question", "violet")
    online = [("Streamlit UI", ["chat, flow analysis, search, triage"], "blue"),
              ("FastAPI backend", ["REST + live events, sign-in"], "blue"),
              ("LangGraph agents", ["supervisor + 4 agents, guardrails"], "violet"),
              ("Tools", ["ML service, prioritiser, hybrid retrieval"], "green")]
    ow, gap = I(2.6), (content_w - I(0.4) - 4 * I(2.6)) / 3
    xs = [MARGIN + I(0.2) + i * (ow + gap) for i in range(4)]
    for x, (title, sub, accent) in zip(xs, online):
        d.node(s, x, I(3.95), ow, I(1.0), title, sub, accent, title_size=13.5, sub_size=11.5)
    for a, b in zip(xs, xs[1:]):
        d.arrow(s, [(a + ow, I(4.45)), (b, I(4.45))], both=True)
    below = [("Analyst", ["in the browser"], "slate"), ("SQLite", ["accounts, saved chats, feedback"], "slate"),
             ("LLM provider", ["gpt-oss-20b on NVIDIA NIM"], "pink"), ("Models + indexes", ["3 ML models, search index"],
                                                                        "slate")]
    for x, (title, sub, accent) in zip(xs, below):
        d.node(s, x, I(5.5), ow, I(0.95), title, sub, accent, title_size=13, sub_size=11.5)
        d.arrow(s, [(x + ow / 2, I(4.95)), (x + ow / 2, I(5.5))], both=True)
    last = oxs[-1] + onw / 2
    d.arrow(s, [(last, I(3.1)), (last, I(3.95))], label="search index", label_at=(last + I(0.1), I(3.6)))

    # 5 Dataset --------------------------------------------------------------------------------------
    s = d.slide("Dataset: CIC-IDS2017", "Canadian Institute for Cybersecurity", notes=(
        "[2:25–3:05] Five days of real lab traffic, 2.8 million flows, 15 labels. After cleaning and "
        "de-duplication 2.24 million remain; a stratified sample of 97,549 becomes the searchable incident "
        "library, and the models use 24 flow metrics. This release has no IP, protocol or timestamp."))
    stats = [("2,830,743", "network flows in 8 CSV files", "blue"), ("15", "labels: BENIGN + 14 attack types", "violet"),
             ("5 days", "of lab traffic, 3–7 July 2017", "slate"),
             ("2,236,725", "flows after cleaning and de-duplication", "green"),
             ("97,549", "incidents in the search library", "blue"), ("24", "flow metrics used by the models", "amber")]
    sw, sh = I(2.2), I(1.5)
    for i, (value, label, accent) in enumerate(stats):
        d.stat(s, MARGIN + (i % 3) * (sw + I(0.2)), top + (i // 3) * (sh + I(0.25)), sw, sh, value, label, accent)
    d.card(s, MARGIN, top + I(3.5), 3 * sw + I(0.4), I(1.45), "Labels include", [
        "DoS (Hulk, GoldenEye, Slowloris, Slowhttptest), DDoS, PortScan, SSH- and FTP-Patator, Web attacks "
        "(brute force, XSS, SQL injection), Bot, Infiltration, Heartbleed"], "violet", title_size=15, body_size=12.5)
    steps = [("Raw CSVs", ["79 columns, 8 files"], "slate"),
             ("Clean", ["fix headers, types, inf → empty, labels"], "slate"),
             ("De-duplicate", ["SHA-256 per row"], "slate"),
             ("Sample + describe", ["stratified, 1 flow → 1 incident text"], "blue")]
    py = top
    for i, (title, sub, accent) in enumerate(steps):
        d.node(s, I(8.15), py, I(4.58), I(0.78), title, sub, accent, title_size=13.5, sub_size=11.5)
        if i:
            d.arrow(s, [(I(10.44), py - I(0.22)), (I(10.44), py)])
        py += I(1.0)
    d.card(s, I(8.15), top + I(4.05), I(4.58), I(0.9), "Limitation", [
        "No IP, protocol or timestamp in this release"], "amber", title_size=14, body_size=12.5)

    # 6 Technology stack -----------------------------------------------------------------------------
    s = d.slide("Technology stack", "Built with", notes=(
        "[3:05–3:30] Python throughout. The LLM is gpt-oss-20b on NVIDIA NIM and can be switched to Gemini "
        "or Claude in .env. Everything else runs locally."))
    d.table(s, MARGIN, top, content_w, [
        ["Component", "Technology"],
        ["Language", "Python 3.10"],
        ["LLM", "openai/gpt-oss-20b on NVIDIA NIM (switchable to Gemini or Claude in .env)"],
        ["Embedding model", "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)"],
        ["Vector database", "ChromaDB (persistent, cosine distance)"],
        ["Keyword search", "rank-bm25 (BM25Okapi), fused with Reciprocal Rank Fusion"],
        ["Agent framework", "LangGraph, LangChain"],
        ["Machine learning", "scikit-learn (Random Forest, gradient boosting, Isolation Forest), SHAP"],
        ["Data processing", "PySpark, Pandas, PyArrow"],
        ["Backend", "FastAPI, Uvicorn (REST + Server-Sent Events, 26 endpoints)"],
        ["Front end", "Streamlit, Plotly"],
        ["Storage", "SQLite: accounts, saved conversations (LangGraph SqliteSaver), feedback"],
        ["Evaluation", "DeepEval, LLM-as-Judge, pytest"],
    ], [2.2, 8], size=13, row_h=0.38)

    # 7 Retrieval ------------------------------------------------------------------------------------
    s = d.slide("Retrieval strategy: hybrid RAG", "Finding similar past incidents", notes=(
        "[3:30–4:15] Keyword search catches exact names and ports, semantic search catches similar "
        "behaviour; Reciprocal Rank Fusion combines both rankings. Hard filters then keep only the right "
        "attack type, port or numeric range. Hybrid had the best hit rate on 12 labelled queries."))
    d.node(s, MARGIN, I(2.45), I(1.9), I(1.0), "Query", ["e.g. SSH brute force"], "slate")
    d.node(s, I(3.0), I(1.8), I(2.35), I(0.95), "Keyword search", ["BM25: exact words, labels"], "blue")
    d.node(s, I(3.0), I(3.15), I(2.35), I(0.95), "Semantic search", ["MiniLM vectors in ChromaDB"], "blue")
    d.node(s, I(5.85), I(2.45), I(2.1), I(1.0), "Fusion", ["Reciprocal Rank", "Fusion (k = 60)"], "violet")
    d.node(s, I(8.45), I(2.45), I(2.1), I(1.0), "Hard filters", ["attack type, port, numeric"], "amber")
    d.node(s, I(11.05), I(2.45), I(1.68), I(1.0), "Top 5", ["cited incidents"], "green")
    d.arrow(s, [(I(2.5), I(2.95)), (I(2.75), I(2.95)), (I(2.75), I(2.275)), (I(3.0), I(2.275))])
    d.arrow(s, [(I(2.5), I(2.95)), (I(2.75), I(2.95)), (I(2.75), I(3.625)), (I(3.0), I(3.625))])
    d.arrow(s, [(I(5.35), I(2.275)), (I(5.6), I(2.275)), (I(5.6), I(2.95)), (I(5.85), I(2.95))])
    d.arrow(s, [(I(5.35), I(3.625)), (I(5.6), I(3.625)), (I(5.6), I(2.95)), (I(5.85), I(2.95))])
    d.arrow(s, [(I(7.95), I(2.95)), (I(8.45), I(2.95))])
    d.arrow(s, [(I(10.55), I(2.95)), (I(11.05), I(2.95))])
    d.table(s, MARGIN, I(4.6), I(6.0), [["Method (12 labelled queries)", "Hit@5", "Precision@5", "MRR"],
                                        ["BM25 keyword", "0.583", "0.567", "0.583"],
                                        ["Vector (semantic)", "0.833", "**0.833**", "**0.833**"],
                                        ["**Hybrid (RRF)**", "**0.917**", "0.733", "0.792"]],
            [2.6, 1, 1.3, 1], size=13, row_h=0.42)
    d.bullets(s, I(7.0), I(4.55), I(5.73), I(2.2), [
        "Each incident is one flow written as a short text (one chunk)",
        "Agents set the filters: 'SSH brute force' → **SSH-Patator**",
        "Numbers in the question become filters: 'more than 10000 packets per second'",
    ], size=14, spacing=8)

    # 8 Machine-learning models ----------------------------------------------------------------------
    s = d.slide("Machine-learning models", "Classify, score, explain, prioritise", notes=(
        "[4:15–5:00] Three models read the same 24 metrics; partial input is filled with normal values and "
        "flagged. The prioritiser combines risk, attack severity and anomaly into P1–P4 with safety rules. "
        "SHAP explains every prediction. Scores at the bottom are on held-out data."))
    d.node(s, MARGIN, I(2.25), I(2.2), I(1.6), "Flow metrics", ["any of the 24", "missing → filled + flagged"],
           "slate")
    models = [("Attack type", ["Random Forest, 15 classes"]), ("Intrusion risk", ["gradient boosting, calibrated"]),
              ("Anomaly", ["Isolation Forest, normal traffic"])]
    for i, (title, sub) in enumerate(models):
        y = I(1.8) + i * I(0.95)
        d.node(s, I(3.3), y, I(3.0), I(0.8), title, sub, "green", title_size=13.5, sub_size=11.5)
        d.arrow(s, [(I(2.8), I(3.05)), (I(3.05), I(3.05)), (I(3.05), y + I(0.4)), (I(3.3), y + I(0.4))])
        d.arrow(s, [(I(6.3), y + I(0.4)), (I(6.6), y + I(0.4)), (I(6.6), I(3.05)), (I(6.9), I(3.05))])
    d.node(s, I(6.9), I(2.25), I(2.7), I(1.6), "Prioritiser", ["0.45 risk + 0.35 severity", "+ 0.20 anomaly",
                                                               "+ safety rules"], "amber")
    d.node(s, I(10.1), I(2.25), I(2.63), I(1.6), "P1 → P4", ["Critical: now", "High: today",
                                                            "Medium: when possible", "Low: monitor"], "red",
           sub_size=11.5)
    d.arrow(s, [(I(9.6), I(3.05)), (I(10.1), I(3.05))])
    d.text(s, I(6.9), I(4.0), I(5.83), I(0.7),
           "Rules: BENIGN but highly unusual → at least P2 · critical attack with risk ≥ 90% → P1 · "
           "PortScan capped at P3", size=11.5, color=MUTED)
    d.card(s, MARGIN, I(4.9), I(3.4), I(1.6), "SHAP explanations", [
        "Which metrics drove each prediction, compared with normal traffic"], "green", title_size=15, body_size=12.5)
    mstats = [("0.986", "classifier accuracy (macro F1 0.839)", "violet"),
              ("0.9997", "risk model ROC-AUC, 0.7% false alarms", "violet"),
              ("0.855", "anomaly detector ROC-AUC", "violet"),
              ("71%", "unseen Infiltration flagged as anomalous", "violet")]
    for i, (value, label, accent) in enumerate(mstats):
        d.stat(s, I(4.25) + i * I(2.15), I(4.9), I(1.95), I(1.6), value, label, accent)

    # 9 Multi-agent workflow --------------------------------------------------------------------------
    s = d.slide("Multi-agent workflow: LangGraph", "One turn through the supervisor graph", notes=(
        "[5:00–6:00] The input guard checks the question; the supervisor picks one of seven routes. An "
        "investigation runs in two parallel stages: Classification with Risk & Priority, then Incident "
        "Search with Response. The final answer is written from all findings and checked again. A question "
        "for a single agent gets that agent's reply directly. Missing key metrics trigger a follow-up "
        "question instead of a guess."))
    cy, nh = I(2.85), I(1.1)
    d.node(s, MARGIN, cy - nh / 2, I(1.5), nh, "Input guard", ["safety check"], "amber", title_size=13)
    d.node(s, I(2.4), cy - nh / 2, I(1.7), nh, "Supervisor", ["picks 1 of 7 routes"], "violet", title_size=13)
    for gx, label, agents in [(I(4.4), "STAGE 1 · in parallel", ["Classification", "Risk & Priority"]),
                              (I(7.1), "STAGE 2 · in parallel", ["Incident Search", "Response"])]:
        d.group(s, gx, I(1.7), I(2.4), I(2.35), label, "violet", label_size=10.5)
        for k, agent in enumerate(agents):
            d.node(s, gx + I(0.15), I(2.1) + k * I(0.95), I(2.1), I(0.8), agent, ["agent + tools"], "violet",
                   title_size=13, sub_size=11)
    d.node(s, I(9.8), cy - nh / 2, I(1.35), nh, "Final answer", ["one reply"], "blue", title_size=13)
    d.node(s, I(11.4), cy - nh / 2, I(1.33), nh, "Output guard", ["checks facts"], "amber", title_size=13)
    for a, b in [(I(2.1), I(2.4)), (I(4.1), I(4.4)), (I(6.8), I(7.1)), (I(9.5), I(9.8)), (I(11.15), I(11.4))]:
        d.arrow(s, [(a, cy), (b, cy)])
    d.table(s, MARGIN, I(4.3), content_w, [
        ["classify", "risk", "search", "recommend", "investigate", "general", "out of scope"],
        ["Classification", "Risk & Priority", "Incident Search", "Response", "all 4 agents, 2 stages",
         "answer directly", "fixed refusal"]], [1, 1, 1, 1, 1.3, 1, 1], size=12, row_h=0.42)
    tips = [("Direct reply", "A single agent writes the answer itself: one LLM call saved", "blue"),
            ("Follow-up", "Missing duration, packets or bytes → asks the user, never guesses", "amber"),
            ("Bounded and visible", "At most 6 tool rounds per agent; every step streamed live to the UI", "green")]
    tw = (content_w - I(0.6)) / 3
    for i, (title, body, accent) in enumerate(tips):
        d.card(s, MARGIN + i * (tw + I(0.3)), I(5.35), tw, I(1.3), title, [body], accent, title_size=15,
               body_size=12.5)

    # 10 Safety and quality --------------------------------------------------------------------------
    s = d.slide("Safety and quality", "Guardrails, LLM-as-Judge, feedback", notes=(
        "[6:00–6:50] Rule-based guards sit before and after the LLM. The judge grades every answer on five "
        "criteria, with rules that cap scores when facts are wrong. Analysts rate answers and correct attack "
        "types; corrections appear in later answers and can retrain the classifier, but the new model is "
        "kept only if macro-F1 does not drop."))
    loop = [("Input guard", ["blocks, redacts"], "amber"), ("Agents + tools", ["grounded findings"], "violet"),
            ("Output guard", ["fixes the answer"], "amber"), ("LLM-as-Judge", ["scores 1–5"], "pink"),
            ("Analyst feedback", ["ratings, corrections"], "green"), ("Safe retraining", ["only if F1 holds"], "green")]
    xs, nw = d.chain(s, MARGIN, I(1.85), content_w, I(0.95), loop)
    d.arrow(s, [(xs[5] + nw / 2, I(2.8)), (xs[5] + nw / 2, I(3.1)), (xs[1] + nw / 2, I(3.1)), (xs[1] + nw / 2, I(2.8))],
            label="new classifier version", label_at=(xs[2] + I(0.4), I(2.86)))
    items = [("Input guard", ["Blocks prompt injection and attack requests; defensive questions allowed",
                              "Redacts keys, passwords, emails, card numbers"], "amber"),
             ("Output guard", ["Marks incident IDs no tool returned", "Flags wrong figures and priorities",
                               "Removes dangerous commands, adds caveats"], "amber"),
             ("LLM-as-Judge + DeepEval", ["Faithfulness, correctness, relevance, actionability, safety",
                                          "Rule caps; calibration 6/6",
                                          "DeepEval: 24 cases, 5 metrics + exact checks"], "pink"),
             ("Feedback loop", ["Thumbs up / down, comments, attack-type corrections",
                                "Corrections shown in later answers", "Exportable for review"], "green")]
    cw = (content_w - I(0.9)) / 4
    for i, (title, body, accent) in enumerate(items):
        d.card(s, MARGIN + i * (cw + I(0.3)), I(3.5), cw, I(2.75), title, body, accent, title_size=16, body_size=13,
               bullets=True)

    # 11 User accounts -------------------------------------------------------------------------------
    s = d.slide("User accounts and saved conversations", "Each analyst keeps their own history", notes=(
        "[6:50–7:30] Sign up and log in from the Streamlit page. Passwords are hashed with scrypt; tokens are "
        "random, only their hash is stored, they expire after 7 days and logout revokes them. Each "
        "conversation belongs to one user and survives restarts through LangGraph's SqliteSaver."))
    d.node(s, MARGIN, I(2.0), I(2.4), I(1.0), "Sign-in page", ["Streamlit: sign up, log in"], "blue")
    d.node(s, I(3.6), I(2.0), I(2.6), I(1.0), "/auth endpoints", ["scrypt hashing, tokens"], "blue")
    d.node(s, I(7.2), I(2.0), I(2.6), I(1.0), "users.db", ["users, sessions, conversations"], "slate")
    d.node(s, MARGIN, I(3.6), I(2.4), I(1.0), "AI Assistant page", ["chat + My conversations"], "blue")
    d.node(s, I(3.6), I(3.6), I(2.6), I(1.0), "/chat endpoints", ["bearer token, owner check"], "violet")
    d.node(s, I(7.2), I(3.6), I(2.6), I(1.0), "checkpoints.db", ["LangGraph SqliteSaver"], "slate")
    for y in (I(2.5), I(4.1)):
        d.arrow(s, [(I(3.0), y), (I(3.6), y)])
        d.arrow(s, [(I(6.2), y), (I(7.2), y)], both=True)
    d.arrow(s, [(I(4.9), I(3.0)), (I(4.9), I(3.6))], label="token", label_at=(I(4.98), I(3.18)))
    d.bullets(s, I(10.15), I(1.95), I(2.58), I(2.8), ["Sign up / log in / log out", "Chats saved per user",
                                                       "Resume after restart"], size=13.5, spacing=8)
    d.card(s, MARGIN, I(5.0), (content_w - I(0.3)) / 2, I(1.7), "Security", [
        "Passwords hashed with scrypt; tokens random, only a SHA-256 digest stored",
        "Tokens expire after 7 days; logout revokes them; 5 failed logins in 5 min → blocked"],
           "amber", title_size=15, body_size=12.5, bullets=True)
    d.card(s, MARGIN + (content_w + I(0.3)) / 2, I(5.0), (content_w - I(0.3)) / 2, I(1.7), "Conversations", [
        "List, resume, rename and delete your own chats",
        "Another user's chat returns 404; feedback records who gave it"],
           "violet", title_size=15, body_size=12.5, bullets=True)

    # 12 Evaluation and performance ------------------------------------------------------------------
    s = d.slide("Evaluation results and performance", "Measured, then improved", notes=(
        "[7:30–8:40] DeepEval on 24 labelled cases plus exact checks computed from the data. After making "
        "agents run in parallel and letting single agents answer directly, quality held or improved and "
        "answers got much faster. Retrieval, model and test numbers on the top row."))
    stats = [("0.97", "faithfulness (DeepEval)", "green"), ("0.97", "answer relevancy", "green"),
             ("99%", "exact checks passed", "blue"), ("0.93", "safety and caveats", "green"),
             ("0.917", "hybrid search Hit@5", "violet"), ("413", "automated tests, all passing", "slate")]
    sw = (content_w - I(1.5)) / 6
    for i, (value, label, accent) in enumerate(stats):
        d.stat(s, MARGIN + i * (sw + I(0.3)), top, sw, I(1.5), value, label, accent)
    d.table(s, MARGIN, I(3.55), I(6.1), [["Quality (24 cases)", "Before", "After"],
                                         ["Exact checks", "98%", "**99%**"], ["Faithfulness", "0.95", "**0.97**"],
                                         ["Answer relevancy", "0.95", "**0.97**"], ["Recommendation", "0.88", "**0.92**"],
                                         ["Safety", "0.92", "**0.93**"], ["Contextual relevancy", "0.85", "0.75*"],
                                         ["Judge vs DeepEval agreement", "67%", "**81%**"]],
            [2.8, 1, 1], size=13, row_h=0.38)
    d.table(s, I(7.0), I(3.55), I(5.73), [["Response time", "Before", "After"],
                                          ["Full investigation", "72–133 s", "**54–68 s**"],
                                          ["Simple question", "16–23 s", "**7–13 s**"]],
            [2.4, 1.2, 1.2], size=13, row_h=0.4)
    d.bullets(s, I(7.0), I(4.95), I(5.73), I(1.8), [
        "Investigations run agents in **2 parallel stages**",
        "Single-agent questions **skip the rewrite**",
        "Evaluation found and fixed 4 real defects",
    ], size=13.5, spacing=6)
    d.text(s, I(7.0), I(6.15), I(5.73), I(0.6),
           "* The evaluator misread 2 correctly filtered searches (right attack type, right incidents)", size=11,
           color=MUTED)

    # 13 Challenges and learnings --------------------------------------------------------------------
    s = d.slide("Challenges and learnings", "What was hard and what we learned", notes=(
        "[8:40–9:20] Challenges: missing fields in the dataset, free LLM limits, small-model quirks, partial "
        "input and latency. Learnings: wrap the LLM in deterministic checks, measure early, route instead of "
        "running everything, and keep the LLM a setting."))
    half = (content_w - I(0.4)) / 2
    d.card(s, MARGIN, top, half, I(4.1), "Challenges", [
        "Dataset has no IP, protocol or timestamp: filters built, but only attack type, port and numbers have data",
        "Free LLM tiers: Claude credit ran out, Gemini 20 requests/day → provider-agnostic, gpt-oss-20b on NIM",
        "Small-model quirks: broken JSON, invented IDs, extra general knowledge",
        "Partial input: analysts know 3–5 of the 24 metrics",
        "Latency: 1–3 minutes per investigation → now about 1 minute with parallel agents",
    ], "red", title_size=21, body_size=14.5, bullets=True)
    d.card(s, MARGIN + half + I(0.4), top, half, I(4.1), "Learnings", [
        "Wrap the LLM in deterministic checks: guardrails, rule caps and exact checks catch what it misses",
        "Measure early: evaluation exposed four real defects",
        "Route, don't pipeline; run independent agents in parallel",
        "Filters beat clever ranking for precise retrieval",
        "Show the work: a live agent trace builds trust",
        "Keep the LLM a configuration choice, not a dependency",
    ], "green", title_size=21, body_size=14.5, bullets=True)

    # 14 Future work ---------------------------------------------------------------------------------
    s = d.slide("Future work", "Next steps", notes=(
        "[9:20–9:45] The main next steps: a richer dataset release, a stronger evaluator, easier deployment "
        "and production-grade security."))
    d.bullets(s, MARGIN, top, I(7.4), I(4.9), [
        "Use the **GeneratedLabelledFlows** release: IP, protocol and timestamp filters; multi-flow incidents",
        "**Stronger evaluator model** and a larger labelled case set",
        "**Docker / docker-compose** for one-command deployment",
        "**CI pipeline** running the 413 tests on every push",
        "Stay signed in across a browser refresh; **HTTPS** for non-local use",
        "Cache repeated tool lookups to cut LLM calls further",
    ], size=17, spacing=14)
    d.card(s, I(8.4), top, I(4.33), I(2.4), "Repository", [
        REPO_URL, "", "README · Architecture diagram", "Design document · API documentation",
        "Dataset details · Evaluation report", "Sample data · 413 tests"], "blue", title_size=19, body_size=13.5)

    # 15 Thank you / Q&A -----------------------------------------------------------------------------
    d.count += 1
    s = d.prs.slides.add_slide(d.prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    d.text(s, 0, I(2.4), W, I(1.2), "Thank you", size=54, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    d.text(s, 0, I(3.7), W, I(0.7), "Questions?", size=26, color=RGBColor(0x93, 0xC5, 0xFD), align=PP_ALIGN.CENTER)
    d.text(s, 0, I(5.0), W, I(0.5), REPO_URL, size=16, color=RGBColor(0xCB, 0xD5, 0xE1), align=PP_ALIGN.CENTER)
    s.notes_slide.notes_text_frame.text = "[9:45–10:00] Thank the panel and invite questions."
    return d


def export_pdf(pptx: Path) -> None:
    """Save a PDF copy through PowerPoint (Windows only); skipped quietly when unavailable."""
    pdf = pptx.with_suffix(".pdf")
    script = (f"$ppt = New-Object -ComObject PowerPoint.Application; "
              f"$p = $ppt.Presentations.Open('{pptx}', $true, $false, $false); "
              f"$p.SaveAs('{pdf}', 32); $p.Close(); $ppt.Quit()")
    try:
        pdf.unlink(missing_ok=True)
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True, timeout=180)
        if not pdf.exists():
            raise OSError("PowerPoint did not write the PDF")
        print(f"wrote {pdf.relative_to(ROOT)}")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"PDF export skipped ({exc.__class__.__name__}); open the .pptx in PowerPoint to export it")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the project presentation.")
    parser.add_argument("--no-pdf", action="store_true", help="Do not export a PDF copy through PowerPoint.")
    args = parser.parse_args()
    deck = build()
    deck.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({deck.count} slides)")
    if not args.no_pdf and sys.platform == "win32":
        export_pdf(OUT)


if __name__ == "__main__":
    main()
