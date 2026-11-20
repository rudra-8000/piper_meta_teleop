"""Tiny layout helpers shared by the two guide builders (reportlab platypus)."""
from __future__ import annotations

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("Body", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Mono", FD + "DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("Mono-Bold", FD + "DejaVuSansMono-Bold.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")

INK = colors.HexColor("#18202c")
MUTED = colors.HexColor("#5b6578")
ACCENT = colors.HexColor("#1b5fc7")
LINE = colors.HexColor("#d5dbe6")
WARN_BG, WARN_BAR = colors.HexColor("#fff4e5"), colors.HexColor("#d97706")
DANGER_BG, DANGER_BAR = colors.HexColor("#fdecec"), colors.HexColor("#c62828")
NOTE_BG, NOTE_BAR = colors.HexColor("#eaf2ff"), ACCENT
OK_BG, OK_BAR = colors.HexColor("#e9f7ef"), colors.HexColor("#1b8a4b")
CODE_BG = colors.HexColor("#f3f5f9")

S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=14, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8.2, leading=11.5, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=17, leading=21, textColor=INK, spaceBefore=4, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=12, leading=16, textColor=ACCENT, spaceBefore=11, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=10, leading=14, textColor=INK, spaceBefore=7, spaceAfter=2),
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=27, leading=32, textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="Body", fontSize=12, leading=17, textColor=MUTED, spaceAfter=14),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.6, leading=14, textColor=INK, leftIndent=14,
                             bulletIndent=3, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.4, leading=11.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.4, leading=11.2, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=10.6, textColor=INK),
    "callout": ParagraphStyle("callout", fontName="Body", fontSize=9.2, leading=13, textColor=INK),
}


def P(t, st="body"): return Paragraph(t, S[st])
def H1(t): return Paragraph(t, S["h1"])
def H2(t): return Paragraph(t, S["h2"])
def H3(t): return Paragraph(t, S["h3"])
def SP(h=4): return Spacer(1, h)


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def steps(items):
    return [Paragraph(t, S["bullet"], bulletText=f"{i}.") for i, t in enumerate(items, 1)]


def code(text):
    t = Table([[Preformatted(text.strip("\n"), S["code"])]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CODE_BG), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return [t, SP(5)]


def callout(kind, title, text):
    bg, bar = {"warn": (WARN_BG, WARN_BAR), "danger": (DANGER_BG, DANGER_BAR),
               "note": (NOTE_BG, NOTE_BAR), "ok": (OK_BG, OK_BAR)}[kind]
    body = [Paragraph(f"<b>{title}</b>", S["callout"]), Paragraph(text, S["callout"])]
    t = Table([[body]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return [KeepTogether([t]), SP(6)]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), S["cellb" if (header and r == 0) else "cell"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f9")))
    t.setStyle(TableStyle(st))
    return [t, SP(7)]


def _box(d, x, y, w, h, label, sub="", fill="#eaf2ff", stroke="#1b5fc7"):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor(stroke), strokeWidth=1))
    d.add(String(x + w / 2, y + h / 2 + (3 if sub else -3), label, fontName="Body-Bold", fontSize=8.5,
                 textAnchor="middle", fillColor=INK))
    if sub:
        d.add(String(x + w / 2, y + h / 2 - 8, sub, fontName="Body", fontSize=7, textAnchor="middle", fillColor=MUTED))


def _arrow(d, x1, y1, x2, y2, label="", color="#5b6578", dy=4):
    d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor(color), strokeWidth=1.3))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    d.add(Polygon([x2, y2, x2 - 6 * math.cos(a - .4), y2 - 6 * math.sin(a - .4),
                   x2 - 6 * math.cos(a + .4), y2 - 6 * math.sin(a + .4)],
                  fillColor=colors.HexColor(color), strokeColor=colors.HexColor(color)))
    if label:
        d.add(String((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, fontName="Body", fontSize=6.8, textAnchor="middle", fillColor=MUTED))


def wiring_diagram():
    d = Drawing(480, 215)
    _box(d, 8, 150, 82, 40, "24 V PSU", "24-26 V, >= 10 A", "#fff4e5", "#d97706")
    _box(d, 8, 55, 82, 40, "24 V PSU", "(second arm)", "#fff4e5", "#d97706")
    _box(d, 150, 150, 92, 40, "Piper arm #1", "aviation plug at base", "#e9f7ef", "#1b8a4b")
    _box(d, 150, 55, 92, 40, "Piper arm #2", "aviation plug at base", "#e9f7ef", "#1b8a4b")
    _box(d, 305, 150, 80, 40, "USB-CAN #1", "CAN_H / CAN_L", "#eaf2ff", "#1b5fc7")
    _box(d, 305, 55, 80, 40, "USB-CAN #2", "CAN_H / CAN_L", "#eaf2ff", "#1b5fc7")
    _box(d, 415, 85, 62, 75, "PC", "Ubuntu", "#f3f5f9", "#5b6578")
    _arrow(d, 90, 170, 150, 170, "power +/-", "#d97706", 4)
    _arrow(d, 90, 75, 150, 75, "power +/-", "#d97706", 4)
    _arrow(d, 242, 170, 305, 170, "CAN_H, CAN_L", "#1b8a4b", 4)
    _arrow(d, 242, 75, 305, 75, "CAN_H, CAN_L", "#1b8a4b", 4)
    _arrow(d, 385, 170, 428, 150, "USB", "#1b5fc7", 6)
    _arrow(d, 385, 75, 428, 95, "USB", "#1b5fc7", -9)
    d.add(String(240, 15, "Each arm needs its OWN USB-CAN adapter: both arms use identical CAN IDs.",
                 fontName="Body", fontSize=7.6, textAnchor="middle", fillColor=INK))
    d.add(String(240, 4, "(Power and CAN share the arm's single aviation plug; red dot to red dot.)",
                 fontName="Body", fontSize=7.2, textAnchor="middle", fillColor=MUTED))
    return d


def latency_diagram():
    d = Drawing(480, 95)
    xs = [(6, "Quest\ntracking"), (100, "WebXR\nframe"), (194, "USB or\nWi-Fi"), (288, "PC loop\nIK+limits"), (382, "CAN bus\n+ arm")]
    for i, (x, lab) in enumerate(xs):
        a, b = lab.split("\n")
        _box(d, x, 40, 86, 38, a, b)
        if i:
            _arrow(d, x - 8, 59, x, 59)
    d.add(String(240, 22, "Every box adds delay - the guide's latency chapter says what you can change in each.",
                 fontName="Body", fontSize=7.6, textAnchor="middle", fillColor=INK))
    return d


class Doc(BaseDocTemplate):
    def __init__(self, path, title, short):
        super().__init__(path, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm,
                         bottomMargin=18 * mm, title=title, author="piper_meta_teleop", subject=short)
        self.short = short
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f", leftPadding=0, rightPadding=0,
                   topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="p", frames=[fr], onPage=self._deco)])

    def _deco(self, c, doc):
        c.saveState()
        c.setStrokeColor(LINE)
        c.line(20 * mm, 13 * mm, A4[0] - 20 * mm, 13 * mm)
        c.setFont("Body", 7.5)
        c.setFillColor(MUTED)
        c.drawString(20 * mm, 8.5 * mm, self.short)
        c.drawRightString(A4[0] - 20 * mm, 8.5 * mm, f"page {doc.page}")
        c.restoreState()


class Story(list):
    """list that also accepts `story += single_flowable`."""

    def __iadd__(self, other):
        if isinstance(other, (list, tuple)):
            return super().__iadd__(other)
        self.append(other)
        return self
