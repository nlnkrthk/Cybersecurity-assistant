#!/usr/bin/env python3
"""
Draw the project diagrams as SVG (for GitHub and documents) and PNG (for slides).

    python docs/diagrams/render_diagrams.py

Writes docs/diagrams/architecture.{svg,png} and docs/diagrams/agent_workflow.{svg,png}.
Every box is placed explicitly (no auto-layout), so the pictures stay tidy and the SVG and PNG
match. The PNG uses Pillow and the Arial fonts that ship with Windows (DejaVu Sans elsewhere).
"""
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent
PNG_SCALE = 2  # PNG pixels per SVG unit

FONT_FILES = {
    False: ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"],
    True: ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
           "DejaVuSans-Bold.ttf"],
}

# Colour per layer: (fill, stroke)
BLUE = ("#EFF6FF", "#2563EB")
INDIGO = ("#EEF2FF", "#4F46E5")
VIOLET = ("#F5F3FF", "#7C3AED")
AMBER = ("#FFFBEB", "#D97706")
PINK = ("#FDF2F8", "#DB2777")
GREEN = ("#ECFDF5", "#059669")
BROWN = ("#FFF7ED", "#B45309")
SLATE = ("#F1F5F9", "#64748B")
RED = ("#FEF2F2", "#DC2626")
TEXT = "#0F172A"
MUTED = "#475569"
LINE = "#334155"


def _font(size: float, bold: bool) -> ImageFont.FreeTypeFont:
    for path in FONT_FILES[bold]:
        try:
            return ImageFont.truetype(path, round(size * PNG_SCALE))
        except OSError:
            continue
    return ImageFont.load_default()


def _hex(color: str):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


class Canvas:
    """Draws the same shapes into an SVG document and a Pillow image."""

    def __init__(self, width: int, height: int, title: str):
        self.w, self.h, self.title = width, height, title
        self.svg = []
        self.img = Image.new("RGB", (width * PNG_SCALE, height * PNG_SCALE), "white")
        self.draw = ImageDraw.Draw(self.img)
        self._fonts = {}

    def font(self, size, bold):
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = _font(size, bold)
        return self._fonts[key]

    @staticmethod
    def s(v):
        return v * PNG_SCALE

    # ------------------------------------------------------------------ shapes
    def rect(self, x, y, w, h, fill, stroke, radius=8, width=1.5, dash=False):
        dash_attr = ' stroke-dasharray="6 4"' if dash else ""
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
                        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>')
        s = self.s
        self.draw.rounded_rectangle([s(x), s(y), s(x + w), s(y + h)], radius=s(radius), fill=_hex(fill),
                                    outline=None if dash else _hex(stroke), width=round(s(width)))
        if dash:
            r = radius
            self._dashed([(x + r, y), (x + w - r, y)], stroke, width)
            self._dashed([(x + w, y + r), (x + w, y + h - r)], stroke, width)
            self._dashed([(x + w - r, y + h), (x + r, y + h)], stroke, width)
            self._dashed([(x, y + h - r), (x, y + r)], stroke, width)

    def cylinder(self, x, y, w, h, fill, stroke, width=1.5):
        e = 10  # ellipse half-height
        self.svg.append(
            f'<path d="M{x},{y + e} A{w / 2},{e} 0 0 1 {x + w},{y + e} L{x + w},{y + h - e} '
            f'A{w / 2},{e} 0 0 1 {x},{y + h - e} Z" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
            f'<path d="M{x},{y + e} A{w / 2},{e} 0 0 0 {x + w},{y + e}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}"/>')
        s, lw = self.s, round(self.s(width))
        self.draw.rectangle([s(x), s(y + e), s(x + w), s(y + h - e)], fill=_hex(fill))
        self.draw.ellipse([s(x), s(y + h - 2 * e), s(x + w), s(y + h)], fill=_hex(fill), outline=_hex(stroke), width=lw)
        self.draw.rectangle([s(x) + lw, s(y + e), s(x + w) - lw, s(y + h - e)], fill=_hex(fill))
        self.draw.line([s(x), s(y + e), s(x), s(y + h - e)], fill=_hex(stroke), width=lw)
        self.draw.line([s(x + w), s(y + e), s(x + w), s(y + h - e)], fill=_hex(stroke), width=lw)
        self.draw.ellipse([s(x), s(y), s(x + w), s(y + 2 * e)], fill=_hex(fill), outline=_hex(stroke), width=lw)

    def text(self, x, y, value, size=13, bold=False, color=TEXT, anchor="middle"):
        """y is the vertical centre of the line."""
        weight = ' font-weight="bold"' if bold else ""
        svg_anchor = {"middle": "middle", "start": "start", "end": "end"}[anchor]
        self.svg.append(f'<text x="{x}" y="{y}" font-size="{size}"{weight} fill="{color}" text-anchor="{svg_anchor}" '
                        f'dominant-baseline="central">{escape(value)}</text>')
        font = self.font(size, bold)
        pil_anchor = {"middle": "mm", "start": "lm", "end": "rm"}[anchor]
        self.draw.text((self.s(x), self.s(y)), value, font=font, fill=_hex(color), anchor=pil_anchor)

    def lines(self, cx, cy, items, gap=17):
        """Centred block of (text, size, bold, color) lines around (cx, cy)."""
        top = cy - gap * (len(items) - 1) / 2
        for i, (value, size, bold, color) in enumerate(items):
            self.text(cx, top + i * gap, value, size, bold, color)

    def box(self, x, y, w, h, title, body=(), colors=VIOLET, dash=False, radius=8, title_size=14, body_size=12):
        self.rect(x, y, w, h, colors[0], colors[1], radius=radius, dash=dash)
        items = [(title, title_size, True, TEXT)] + [(b, body_size, False, MUTED) for b in body]
        self.lines(x + w / 2, y + h / 2, items)

    def store(self, x, y, w, h, title, body=(), colors=BROWN):
        self.cylinder(x, y, w, h, colors[0], colors[1])
        items = [(title, 14, True, TEXT)] + [(b, 12, False, MUTED) for b in body]
        self.lines(x + w / 2, y + h / 2 + 5, items)

    def group(self, x, y, w, h, label, colors):
        self.rect(x, y, w, h, "#FFFFFF", colors[1], radius=12, width=1.2)
        self.rect(x, y, w, h, colors[0], colors[1], radius=12, width=1.2)
        self.text(x + 14, y + 16, label, 13, True, colors[1], anchor="start")

    # ------------------------------------------------------------------ arrows
    def _dashed(self, points, color, width, on=7, off=5):
        s = self.s
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            length = math.hypot(x2 - x1, y2 - y1)
            if not length:
                continue
            ux, uy, pos = (x2 - x1) / length, (y2 - y1) / length, 0.0
            while pos < length:
                end = min(pos + on, length)
                self.draw.line([s(x1 + ux * pos), s(y1 + uy * pos), s(x1 + ux * end), s(y1 + uy * end)],
                               fill=_hex(color), width=round(s(width)))
                pos += on + off

    def arrow(self, points, label=None, label_at=None, color=LINE, dash=False, width=1.6, both=False,
              label_anchor="start"):
        dash_attr = ' stroke-dasharray="7 5"' if dash else ""
        path = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>')
        if dash:
            self._dashed(points, color, width)
        else:
            self.draw.line([(self.s(x), self.s(y)) for x, y in points], fill=_hex(color), width=round(self.s(width)),
                           joint="curve")
        self._head(points[-2], points[-1], color)
        if both:
            self._head(points[1], points[0], color)
        if label:
            lx, ly = label_at or ((points[0][0] + points[1][0]) / 2 + 8, (points[0][1] + points[1][1]) / 2)
            self.text(lx, ly, label, 11.5, False, MUTED, anchor=label_anchor)

    def _head(self, start, end, color, size=9):
        (x1, y1), (x2, y2) = start, end
        angle = math.atan2(y2 - y1, x2 - x1)
        left = (x2 - size * math.cos(angle - 0.42), y2 - size * math.sin(angle - 0.42))
        right = (x2 - size * math.cos(angle + 0.42), y2 - size * math.sin(angle + 0.42))
        pts = [end, left, right]
        self.svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="{color}"/>')
        self.draw.polygon([(self.s(x), self.s(y)) for x, y in pts], fill=_hex(color))

    # ------------------------------------------------------------------ output
    def save(self, name: str) -> None:
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" font-family="Arial, Helvetica, sans-serif">'
               f'<title>{escape(self.title)}</title><rect width="100%" height="100%" fill="white"/>'
               + "".join(self.svg) + "</svg>\n")
        (OUT_DIR / f"{name}.svg").write_text(svg, encoding="utf-8")
        self.img.save(OUT_DIR / f"{name}.png", optimize=True)
        print(f"wrote {name}.svg and {name}.png ({self.img.width}x{self.img.height})")


# ============================================================================
# System architecture
# ============================================================================
def architecture() -> None:
    c = Canvas(1620, 1150, "System architecture: AI-Powered Cybersecurity Incident Response Assistant")
    c.text(810, 22, "AI-Powered Cybersecurity Incident Response Assistant: system architecture", 18, True)

    # Analyst
    c.box(680, 48, 260, 38, "Security analyst (web browser)", colors=SLATE, radius=19)
    c.arrow([(810, 86), (810, 106)])

    # 1 Front end
    c.group(30, 106, 1540, 112, "1  Streamlit front end  ·  localhost:8501  (thin client, no keys in the UI)", BLUE)
    pages = [("Sign in", ["create account", "or log in"]), ("AI Assistant", ["chat, live agent tracker,", "saved conversations"]),
             ("Flow Analysis", ["attack type, risk,", "priority, SHAP charts"]), ("Incident Search", ["similar incidents,", "attack-type profiles"]),
             ("Triage Queue", ["rank a CSV or a list", "of incident IDs"]), ("Quality & Feedback", ["grade answers,", "analyst feedback"])]
    for i, (title, body) in enumerate(pages):
        c.box(55 + i * 250, 136, 230, 68, title, body, colors=BLUE)
    c.arrow([(810, 218), (810, 256)], "HTTP JSON  ·  live agent trace over Server-Sent Events  ·  bearer token",
            label_at=(822, 237))

    # 2 API
    c.group(30, 256, 1540, 112, "2  FastAPI backend  ·  localhost:8000", INDIGO)
    api = [("/auth", ["sign-up, login,", "logout (scrypt, tokens)"]), ("/chat  ·  /chat/stream", ["ask, list, rename,", "delete own conversations"]),
           ("/judge  ·  /feedback", ["grade answers, ratings,", "label corrections"]), ("/ml  ·  /search  ·  /incidents", ["direct model and search", "calls (batch, CSV upload)"])]
    for i, (title, body) in enumerate(api):
        c.box(55 + i * 377, 286, 357, 68, title, body, colors=INDIGO)

    # 3 Assistant + LLM
    c.group(30, 410, 1250, 175, "3  LangGraph multi-agent assistant", VIOLET)
    c.arrow([(611, 354), (611, 410)], "chat", label_at=(620, 382))
    c.arrow([(986, 354), (986, 380), (1222, 380), (1222, 452)], "grade on request", label_at=(1212, 432),
            label_anchor="end")
    nodes = [
        (50, 150, "Input guard", ["blocks injection and", "harmful requests,", "redacts secrets"], AMBER),
        (225, 160, "Supervisor", ["routes: classify, risk,", "search, recommend,", "investigate, general"], VIOLET),
        (410, 250, "Specialist agents", ["Classification", "Risk & Priority", "Incident Search  ·  Response"], VIOLET),
        (685, 140, "Final answer", ["one reply from all", "agents' findings"], VIOLET),
        (850, 175, "Output guard", ["checks IDs, figures,", "priority; strips", "dangerous commands"], AMBER),
        (1050, 210, "LLM-as-Judge", ["optional: 5 criteria", "scored 1-5, rule caps"], PINK),
    ]
    for x, w, title, body, colors in nodes:
        c.box(x, 452, w, 108, title, body, colors=colors)
    for (x1, w1, *_), (x2, *_) in zip(nodes, nodes[1:]):
        c.arrow([(x1 + w1, 506), (x2, 506)])

    c.box(1320, 440, 250, 120, "LLM provider", ["NVIDIA NIM: gpt-oss-20b", "(Gemini or Claude:", "one setting in .env)"],
          colors=PINK, dash=True)
    c.arrow([(1280, 500), (1320, 500)], color="#DB2777", dash=True, both=True)
    c.text(1445, 575, "prompts / replies", 11.5, False, "#DB2777")

    # 4 Tools
    c.group(30, 630, 1540, 135, "4  Agent tools and services  (the API's /ml and /search call the same functions)", GREEN)
    c.arrow([(535, 585), (535, 630)], "tool calls", label_at=(545, 607))
    c.arrow([(1360, 354), (1360, 396), (1595, 396), (1595, 700), (1570, 700)], "direct calls", label_at=(1590, 610),
            label_anchor="end")
    c.box(55, 662, 530, 88, "ML service", ["attack type: Random Forest (15 classes)",
                                           "intrusion risk: calibrated gradient boosting",
                                           "anomaly: Isolation Forest  ·  SHAP explanations"], colors=GREEN)
    c.box(610, 662, 380, 88, "Prioritizer", ["risk + attack severity + anomaly", "→ score → P1 Critical … P4 Low",
                                             "with safety overrides"], colors=GREEN)
    c.box(1015, 662, 530, 88, "Hybrid retrieval", ["BM25 keywords + Chroma embeddings (MiniLM)",
                                                   "fused with Reciprocal Rank Fusion",
                                                   "hard filters: attack type, port, numeric"], colors=GREEN)

    # 5 Storage
    c.group(30, 805, 1540, 120, "5  Storage", BROWN)
    c.arrow([(320, 750), (320, 835)])
    c.arrow([(1280, 750), (1280, 835)])
    c.store(55, 835, 360, 78, "ML models", ["ml/models: .pkl + metadata,", "versioned classifiers"])
    c.store(440, 835, 340, 78, "Accounts + saved chats", ["data/app (SQLite):", "users, sessions, checkpoints"])
    c.store(805, 835, 340, 78, "Analyst feedback", ["data/feedback (SQLite):", "ratings, corrections"])
    c.store(1170, 835, 375, 78, "Incident library", ["97,549 incidents: Parquet,", "BM25 index, Chroma vectors"])

    # 6 Offline
    c.group(30, 965, 1540, 125, "6  Offline pipelines", SLATE)
    c.box(55, 995, 360, 82, "ML notebooks", ["train and evaluate the three models", "on cleaned CIC-IDS2017 flows"], colors=SLATE)
    c.box(440, 995, 340, 82, "Evaluation", ["DeepEval metrics + exact checks,", "LLM-judge calibration"], colors=SLATE)
    c.box(805, 995, 340, 82, "Guarded retraining", ["corrections → new classifier,", "kept only if macro-F1 holds"], colors=SLATE)
    c.box(1170, 995, 375, 82, "PySpark data pipeline", ["2.8M flows → clean, dedupe, sample", "→ incident texts → indexes"], colors=SLATE)
    c.arrow([(235, 995), (235, 913)], "models", label_at=(245, 950))
    c.arrow([(975, 913), (975, 995)], "corrections", label_at=(985, 950))
    c.arrow([(1357, 995), (1357, 913)], "incidents + indexes", label_at=(1367, 950))
    c.arrow([(880, 995), (880, 946), (330, 946), (330, 913)], color=LINE)
    c.text(605, 936, "new classifier version", 11.5, False, MUTED)

    # Legend
    c.arrow([(40, 1122), (80, 1122)])
    c.text(88, 1122, "request / data flow", 12, False, MUTED, anchor="start")
    c.arrow([(240, 1122), (280, 1122)], color="#DB2777", dash=True)
    c.text(288, 1122, "LLM calls", 12, False, MUTED, anchor="start")
    c.rect(390, 1114, 16, 16, AMBER[0], AMBER[1], radius=3)
    c.text(412, 1122, "guardrails", 12, False, MUTED, anchor="start")
    c.text(1570, 1122, "Data: CIC-IDS2017 (MachineLearningCVE)  ·  every component runs locally", 12, False, MUTED,
           anchor="end")
    c.save("architecture")


# ============================================================================
# One assistant turn
# ============================================================================
def agent_workflow() -> None:
    c = Canvas(1140, 1010, "Assistant workflow: one turn through the LangGraph supervisor graph")
    c.text(570, 22, "One assistant turn (LangGraph supervisor graph)", 18, True)

    c.box(445, 46, 250, 40, "User message", colors=SLATE, radius=20)
    c.arrow([(570, 86), (570, 112)])
    c.box(445, 112, 250, 70, "Input guard", ["checks for injection and harm,", "redacts secrets"], colors=AMBER)
    c.box(830, 112, 270, 70, "Blocked: fixed refusal", ["no LLM call; the message is", "replaced in the history"], colors=RED)
    c.arrow([(695, 147), (830, 147)], "injection or harmful", label_at=(762, 136), label_anchor="middle")
    c.arrow([(570, 182), (570, 222)], "allowed", label_at=(580, 202))
    c.box(445, 222, 250, 66, "Supervisor (router)", ["LLM picks one route", "for the question"], colors=VIOLET)

    routes = [("classify", "Classification", "agent"), ("risk", "Risk & Priority", "agent"),
              ("search", "Incident Search", "agent"), ("recommend", "Response", "agent"),
              ("investigate", "All four agents", "2 parallel stages"), ("general / out of scope", "No tools", "answer directly")]
    col_w, gap, x0 = 168, 14, 33
    centers = []
    for i, (route, name, sub) in enumerate(routes):
        x = x0 + i * (col_w + gap)
        centers.append(x + col_w / 2)
        c.rect(x, 340, col_w, 74, VIOLET[0] if i < 5 else SLATE[0], VIOLET[1] if i < 5 else SLATE[1])
        c.lines(x + col_w / 2, 377, [(route, 11.5, False, "#7C3AED" if i < 5 else MUTED), (name, 14, True, TEXT),
                                      (sub, 12, False, MUTED)])
        c.arrow([(570, 288), (570, 312), (x + col_w / 2, 312), (x + col_w / 2, 340)])

    c.box(x0, 462, 5 * col_w + 4 * gap, 72, "Agent tools",
          ["look up incident · classify attack type · intrusion risk · anomaly · SHAP explanation",
           "prioritize (single or batch) · hybrid incident search · attack-type profiles"], colors=GREEN)
    for cx in centers[:5]:
        c.arrow([(cx, 414), (cx, 462)])

    c.box(x0, 590, 440, 72, "Follow-up question", ["key metrics missing: asks for duration,", "packets or bytes instead of guessing"],
          colors=AMBER)
    c.box(520, 590, 587, 72, "Final answer", ["Assessment · Why · Similar incidents · Actions ·",
                                              "Confidence  (one agent: its reply, no rewrite)"], colors=VIOLET)
    c.arrow([(253, 534), (253, 590)], "metrics missing", label_at=(263, 562))
    c.arrow([(700, 534), (700, 590)], "findings", label_at=(710, 562))
    c.arrow([(centers[5], 414), (centers[5], 590)])

    c.box(420, 712, 300, 72, "Output guard", ["verifies IDs, figures and priority,", "removes dangerous commands"], colors=AMBER)
    c.arrow([(253, 662), (253, 690), (520, 690), (520, 712)])
    c.arrow([(813, 662), (813, 690), (620, 690), (620, 712)])

    c.box(800, 812, 300, 72, "LLM-as-Judge", ["faithfulness, correctness, relevance,", "actionability, safety (1-5)"], colors=PINK)
    c.arrow([(720, 760), (950, 760), (950, 812)], "quality check on", label_at=(760, 749))
    c.box(345, 912, 450, 60, "Answer + agent trace", ["streamed live; saved to the user's conversation"],
          colors=SLATE, radius=24)
    c.arrow([(570, 784), (570, 912)], "quality check off", label_at=(580, 850))
    c.arrow([(950, 884), (950, 942), (795, 942)])
    c.arrow([(1100, 147), (1120, 147), (1120, 990), (570, 990), (570, 972)])
    c.save("agent_workflow")


# ============================================================================
# Data flow: dataset → … → user interface
# ============================================================================
def data_flow() -> None:
    c = Canvas(1600, 860, "Data flow: from the CIC-IDS2017 dataset to the user interface")
    c.text(800, 24, "Data flow: dataset → preprocessing → chunking → embeddings → vector database → retriever "
                    "→ LLM / RAG → agents → API → UI", 17, True)

    c.text(40, 74, "OFFLINE  ·  build the incident library once", 13, True, SLATE[1], anchor="start")
    row1 = [
        ("1  Dataset", ["CIC-IDS2017 MachineLearningCVE", "8 CSV files, 2,830,743 flows", "15 traffic labels"], SLATE),
        ("2  Preprocessing", ["PySpark: normalise headers, cast,", "inf → null, fix labels,", "SHA-256 dedupe → 2,236,725"], SLATE),
        ("3  Chunking", ["stratified sample: 97,549 incidents", "1 flow = 1 template text (one chunk)", "≤ 241 tokens, no splitting"], BLUE),
        ("4  Embeddings", ["all-MiniLM-L6-v2", "384-dimension vectors", "batches of 128 (CPU)"], BLUE),
        ("5  Vector database", ["ChromaDB, cosine distance", "+ incident metadata", "(label, port, …)"], BROWN),
    ]
    for i, (title, body, colors) in enumerate(row1):
        x = 40 + i * 310
        c.box(x, 96, 270, 120, title, body, colors=colors)
        if i:
            c.arrow([(x - 40, 156), (x, 156)])
    c.box(970, 262, 270, 84, "Keyword index", ["BM25 (rank-bm25)", "same texts and metadata"], colors=BROWN)
    c.arrow([(795, 216), (795, 304), (970, 304)])

    c.text(40, 410, "AT QUESTION TIME  ·  arrows show how evidence reaches the analyst", 13, True, VIOLET[1], anchor="start")
    row2 = [
        ("10  User interface", ["Streamlit: chat with live", "agent tracker, flow analysis,", "search, triage, feedback"], BLUE),
        ("9  API layer", ["FastAPI: REST + Server-Sent", "Events, bearer-token auth,", "saved conversations"], INDIGO),
        ("8  Agents", ["LangGraph supervisor +", "4 specialist agents,", "guardrails, LLM judge"], VIOLET),
        ("7  LLM / RAG", ["grounded answer from the", "evidence, cited incident IDs", "(gpt-oss-20b on NVIDIA NIM)"], PINK),
        ("6  Retriever", ["filters (attack type, port,", "numeric) → BM25 + vector", "→ Reciprocal Rank Fusion"], GREEN),
    ]
    for i, (title, body, colors) in enumerate(row2):
        x = 40 + i * 310
        c.box(x, 432, 270, 120, title, body, colors=colors)
        if i:
            c.arrow([(x, 492), (x - 40, 492)])
    c.arrow([(1480, 216), (1480, 432)], "vector search", label_at=(1490, 330))
    c.arrow([(1240, 304), (1330, 304), (1330, 432)], "keyword search", label_at=(1340, 380))

    c.text(40, 610, "MACHINE-LEARNING TRACK  ·  models the agents call as tools", 13, True, GREEN[1], anchor="start")
    c.box(350, 632, 270, 110, "Feature table", ["97,548 sampled flows,", "24 numeric features", "(dst_port excluded)"], colors=SLATE)
    c.box(660, 632, 270, 110, "Models + SHAP", ["attack type · intrusion risk ·", "anomaly · prioritiser P1–P4", "SHAP explanations"], colors=GREEN)
    c.arrow([(620, 687), (660, 687)], "train", label_at=(640, 674), label_anchor="middle")
    c.arrow([(795, 632), (795, 552)], "tool calls", label_at=(805, 595))
    c.box(970, 632, 580, 110, "Feedback loop", ["analyst ratings and attack-type corrections (SQLite) →",
                                                "guarded retraining: a new classifier is kept only if macro-F1 holds"], colors=AMBER)
    c.arrow([(970, 687), (930, 687)])

    c.text(800, 800, "Steps 1–5 run once (about 40 minutes on a CPU). Steps 6–10 run for every question: the question "
                     "goes from the interface (left) to the retriever (right), and the evidence and answer come back "
                     "along the arrows.", 12, False, MUTED)
    c.text(800, 824, "Each incident is one network flow, so each search hit is a complete, citable incident with its "
                     "own ID and metadata.", 12, False, MUTED)
    c.save("data_flow")


if __name__ == "__main__":
    architecture()
    agent_workflow()
    data_flow()
