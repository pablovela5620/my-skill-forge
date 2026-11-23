"""Diagram primitives in the style of the Rerun docs illustrations (see references/DESIGN.md, "Diagrams").

Copy this file next to the script that builds the artifact and compose an inline SVG from the pieces:

    from rerun_figure import BG, chunk, file_icon, arrow, para, leader, speed_svg
    parts = [f'<svg viewBox="0 0 1120 300" xmlns="http://www.w3.org/2000/svg">', f'<rect width="1120" height="300" rx="14" fill="{BG}"/>',
             file_icon(48, 70, "head_pose.csv", "8,915 rows"), arrow(104, 96, 150), para(150, 76, 110, 40, "READER"),
             arrow(260, 96, 330, ".stream()"), chunk(334, 64, 64, 64, rows=5), leader(366, 128, 166, "whole chunks"), "</svg>"]

Pieces: chunk (the docs chunk glyph: header, row-id/time/component columns), file_icon, arrow (optional green API
label), para (D2 parallelogram: reader / terminal call), store (D2 cylinder), box (process), plan (dashed green: a lazy
plan), leader (dotted line to a plain label), lane_title, speed_svg (measured bars). Render and check that no label
touches a shape or another label. `example_svg()` is the step-1 figure from the SelfCap guide (logging SDK vs chunk
pipeline), kept as a worked example.

Second theme, the kit board (bottom of the file): the Rerun FigJam diagram starter kit, translated to dark. Use it for
slide-like overview figures: kit_board, kit_box, kit_table, kit_file, kit_arrow, kit_note, kit_label, kit_caption;
`kit_example_svg()` is a worked example. Both themes are dark; artifacts are dark by default.
"""

BG, PANEL, HEAD, LINE, TXT, DIM = "#0e0f10", "#1a1c1e", "#25282b", "#5b6066", "#e6e8ea", "#8a9097"
ROW, TIME, COMP = "#22b5e6", "#de7df3", "#6cefb4"
ROW_BG, TIME_BG, COMP_BG = "#11313c", "#3a2742", "#1d4536"
GREEN = "#3fe0a0"
MONO = "'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "Inter,ui-sans-serif,system-ui,sans-serif"


def chunk(x: float, y: float, w: float, h: float, rows: int) -> str:
    """The docs' chunk glyph: header bar, then row-id, time and two component columns."""
    pad, gap = 0.07 * w, 0.05 * w
    head_h = 0.16 * h
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{0.06 * w:.1f}" fill="{PANEL}"/>',
           f'<rect x="{x + pad}" y="{y + pad}" width="{w - 2 * pad}" height="{head_h}" rx="3" fill="{HEAD}"/>']
    cy, ch = y + 2 * pad + head_h, h - 3 * pad - head_h
    widths = [0.17, 0.17, 0.22, 0.22]
    cx = x + pad
    inner = w - 2 * pad - 3 * gap
    for frac, (fg, bg) in zip(widths, [(ROW, ROW_BG), (TIME, TIME_BG), (COMP, COMP_BG), (COMP, COMP_BG)], strict=True):
        cw = inner * frac / sum(widths)
        out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw:.1f}" height="{ch:.1f}" rx="3" fill="{bg}"/>')
        bar_h = min(ch / rows * 0.62, 7)
        for r in range(rows):
            by = cy + 3 + r * (ch - 6) / rows
            out.append(f'<rect x="{cx + 2.5:.1f}" y="{by:.1f}" width="{cw - 5:.1f}" height="{bar_h:.1f}" rx="1.5" fill="{fg}"/>')
        cx += cw + gap
    return "".join(out)


def file_icon(x: float, y: float, label: str, sub: str) -> str:
    fold = 12
    return (f'<path d="M{x} {y} h{40 - fold} l{fold} {fold} v{52 - fold} h-40 z" fill="{PANEL}" stroke="{LINE}"/>'
            + "".join(f'<rect x="{x + 7}" y="{y + 20 + i * 7}" width="26" height="3" rx="1" fill="{COMP}" opacity=".75"/>' for i in range(4))
            + f'<text x="{x + 20}" y="{y + 72}" fill="{TXT}" font-size="13" text-anchor="middle" font-family="{MONO}">{label}</text>'
            + f'<text x="{x + 20}" y="{y + 89}" fill="{DIM}" font-size="12" text-anchor="middle" font-family="{SANS}">{sub}</text>')


def arrow(x1: float, y: float, x2: float, label: str = "", dashed: bool = False) -> str:
    dash = ' stroke-dasharray="3 4"' if dashed else ""
    out = f'<path d="M{x1} {y} H{x2 - 6}" stroke="{LINE}" stroke-width="1.4"{dash}/><path d="M{x2 - 7} {y - 4} L{x2} {y} L{x2 - 7} {y + 4} z" fill="{LINE}"/>'
    if label:
        out += f'<text x="{(x1 + x2) / 2}" y="{y - 8}" fill="{GREEN}" font-size="12.5" text-anchor="middle" font-family="{MONO}">{label}</text>'
    return out


def para(x: float, y: float, w: float, h: float, label: str) -> str:
    s = 12
    return (f'<path d="M{x + s} {y} H{x + w} L{x + w - s} {y + h} H{x} z" fill="{PANEL}" stroke="{LINE}" stroke-width="1.2"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + 4.5}" fill="{TXT}" font-size="12.5" text-anchor="middle" letter-spacing="1" font-family="{MONO}">{label}</text>')


def leader(x: float, y1: float, y2: float, text: str, anchor: str = "middle") -> str:
    ty = y2 + (14 if y2 > y1 else -6)
    return (f'<path d="M{x} {y1} V{y2}" stroke="{DIM}" stroke-width="1" stroke-dasharray="1.5 3"/>'
            f'<text x="{x}" y="{ty}" fill="{TXT}" font-size="13.5" text-anchor="{anchor}" font-family="{SANS}">{text}</text>')


def example_svg() -> str:
    w, h = 1120, 500
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Two ways into an rrd file. Logging SDK: a Python loop calls rr.log once per row of head_pose.csv, making 8,915 one-row chunks that a batcher merges. Chunk pipeline: a reader yields whole chunks into a LazyChunkStream plan of filters and lenses, which runs only at the terminal call write_rrd." xmlns="http://www.w3.org/2000/svg">',
         f'<rect width="{w}" height="{h}" rx="14" fill="{BG}"/>']
    # Lane 1: logging SDK
    y1 = 70
    p.append(f'<text x="36" y="44" fill="{DIM}" font-size="12.5" letter-spacing="1.5" font-family="{MONO}">LOGGING SDK · rr.log</text>')
    p.append(file_icon(48, y1, "head_pose.csv", "8,915 rows"))
    p.append(arrow(104, y1 + 26, 178))
    p.append(f'<rect x="178" y="{y1}" width="190" height="56" rx="8" fill="{PANEL}" stroke="{LINE}"/>'
             f'<text x="194" y="{y1 + 23}" fill="{TXT}" font-size="12.5" font-family="{MONO}">for row in csv:</text>'
             f'<text x="210" y="{y1 + 42}" fill="{GREEN}" font-size="12.5" font-family="{MONO}">rr.log(…)</text>')
    p.append(leader(273, y1 + 56, y1 + 96, "Python, once per row"))
    p.append(arrow(368, y1 + 26, 404))
    for i in range(7):
        p.append(chunk(408 + i * 34, y1 + 12, 28, 30, 1))
    p.append(leader(527, y1 + 42, y1 + 96, "8,915 one-row chunks"))
    p.append(arrow(646, y1 + 26, 688))
    p.append(f'<rect x="688" y="{y1 + 8}" width="104" height="36" rx="8" fill="{PANEL}" stroke="{LINE}"/>'
             f'<text x="740" y="{y1 + 31}" fill="{TXT}" font-size="12.5" text-anchor="middle" letter-spacing="1" font-family="{MONO}">BATCHER</text>')
    p.append(leader(740, y1 + 44, y1 + 96, "merges them back"))
    p.append(arrow(792, y1 + 26, 834))
    p.append(chunk(838, y1 - 4, 92, 62, 5))
    p.append(arrow(930, y1 + 26, 1040))
    p.append(file_icon(1044, y1, ".rrd", ""))
    # divider
    p.append(f'<path d="M36 268 H{w - 36}" stroke="{HEAD}" stroke-width="1"/>')
    # Lane 2: chunk pipeline
    y2 = 340
    p.append(f'<text x="36" y="306" fill="{GREEN}" font-size="12.5" letter-spacing="1.5" font-family="{MONO}">CHUNK PIPELINE · rerun.chunk</text>')
    p.append(file_icon(48, y2, "files", "mp4 · rrd · mcap"))
    p.append(arrow(104, y2 + 26, 134))
    p.append(para(134, y2 + 6, 110, 40, "READER"))
    p.append(arrow(244, y2 + 26, 330, ".stream()"))
    for i in range(3):
        p.append(chunk(334 + i * 74, y2 - 6, 64, 64, 5))
    p.append(leader(441, y2 + 58, y2 + 96, "whole chunks, many rows each"))
    p.append(arrow(552, y2 + 26, 596))
    p.append(f'<rect x="596" y="{y2 - 10}" width="200" height="72" rx="6" fill="none" stroke="{GREEN}" stroke-width="1.2" stroke-dasharray="5 4"/>'
             f'<text x="696" y="{y2 + 18}" fill="{GREEN}" font-size="13.5" text-anchor="middle" font-family="{MONO}">LazyChunkStream</text>'
             f'<text x="696" y="{y2 + 42}" fill="{TXT}" font-size="12.5" text-anchor="middle" font-family="{MONO}">filter · lenses · map</text>')
    p.append(leader(696, y2 + 62, y2 + 118, "a plan: building it reads nothing"))
    p.append(arrow(796, y2 + 26, 890, "write_rrd()"))
    p.append(para(890, y2 + 6, 132, 40, "TERMINAL CALL"))
    p.append(leader(956, y2 + 46, y2 + 96, "runs here, on all cores"))
    p.append(arrow(1022, y2 + 26, 1040))
    p.append(file_icon(1044, y2, ".rrd", ""))
    p.append("</svg>")
    return "".join(p)


def speed_svg(ms: dict[str, float]) -> str:
    """Measured time per method, as bars on the same near-black panel."""
    w, h, left, right = 1120, 200, 250, 150
    slowest = max(ms.values())
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Measured: rr.log per row {ms["rr.log per row"]:.0f} ms, rr.send_columns {ms["rr.send_columns"]:.1f} ms, chunk pipeline with a lens {ms["chunk pipeline + lens"]:.1f} ms" xmlns="http://www.w3.org/2000/svg">',
         f'<rect width="{w}" height="{h}" rx="14" fill="{BG}"/>',
         f'<text x="36" y="36" fill="{DIM}" font-size="12.5" letter-spacing="1.5" font-family="{MONO}">MEASURED · d31 head pose + left hand · 8,915 rows each</text>']
    colors = {"rr.log per row": TIME, "rr.send_columns": ROW, "chunk pipeline + lens": COMP}
    for i, (name, value) in enumerate(ms.items()):
        y = 62 + i * 42
        bw = max((w - left - right) * value / slowest, 4)
        speed = "" if value == slowest else f"  ·  {slowest / value:.0f}× faster"
        p.append(f'<text x="{left - 16}" y="{y + 15}" fill="{TXT}" font-size="13.5" text-anchor="end" font-family="{MONO}">{name}</text>'
                 f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="22" rx="4" fill="{colors[name]}"/>'
                 f'<text x="{left + bw + 10:.1f}" y="{y + 16}" fill="{TXT}" font-size="13.5" font-family="{SANS}">{value:.0f} ms{speed}</text>')
    p.append("</svg>")
    return "".join(p)


def store(x: float, y: float, w: float, h: float, label: str) -> str:
    """D2 cylinder: a store."""
    ry = 0.14 * h
    return (f'<path d="M{x} {y + ry} A{w / 2} {ry} 0 0 1 {x + w} {y + ry} V{y + h - ry} A{w / 2} {ry} 0 0 1 {x} {y + h - ry} Z" fill="{PANEL}" stroke="{LINE}" stroke-width="1.2"/>'
            f'<path d="M{x} {y + ry} A{w / 2} {ry} 0 0 0 {x + w} {y + ry}" fill="none" stroke="{LINE}" stroke-width="1.2"/>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + 8}" fill="{TXT}" font-size="12.5" text-anchor="middle" letter-spacing="1" font-family="{MONO}">{label}</text>')


def box(x: float, y: float, w: float, h: float, label: str, sub: str = "") -> str:
    """Rectangle: a process step. `sub` is an optional green API name under the label."""
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{PANEL}" stroke="{LINE}"/>'
    ly = y + h / 2 + (-4 if sub else 4.5)
    out += f'<text x="{x + w / 2}" y="{ly}" fill="{TXT}" font-size="12.5" text-anchor="middle" letter-spacing="1" font-family="{MONO}">{label}</text>'
    if sub:
        out += f'<text x="{x + w / 2}" y="{ly + 18}" fill="{GREEN}" font-size="12.5" text-anchor="middle" font-family="{MONO}">{sub}</text>'
    return out


def plan(x: float, y: float, w: float, h: float, title: str, ops: str) -> str:
    """Dashed green rectangle: a lazy plan (nothing runs until the terminal call)."""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="{GREEN}" stroke-width="1.2" stroke-dasharray="5 4"/>'
            f'<text x="{x + w / 2}" y="{y + 28}" fill="{GREEN}" font-size="13.5" text-anchor="middle" font-family="{MONO}">{title}</text>'
            f'<text x="{x + w / 2}" y="{y + 52}" fill="{TXT}" font-size="12.5" text-anchor="middle" font-family="{MONO}">{ops}</text>')


def lane_title(x: float, y: float, text: str, recommended: bool = False) -> str:
    """Mono uppercase lane title; green for the recommended lane."""
    return f'<text x="{x}" y="{y}" fill="{GREEN if recommended else DIM}" font-size="12.5" letter-spacing="1.5" font-family="{MONO}">{text}</text>'


# ---- Kit board: the Rerun FigJam diagram starter kit, translated to dark ----------------------------------------
# The kit (Figma "Diagram starter kit") draws slide-style boards: mono labels in normal case, 2 px strokes, ~8 px
# corners, open line arrowheads, table glyphs (blue header dashes, pink index dot, green data dashes), file glyphs
# with a violet-blue pill, green annotation text. Its own dark rule: step from darkest to lightest with minimal steps,
# alternating two shades for nesting. Colours were read from screenshots, not exact Figma values.

KIT_BOARD, KIT_CARD, KIT_CARD_2 = "#0a0c10", "#14171d", "#1d2129"
KIT_STROKE, KIT_ARROW, KIT_PAGE = "#34446b", "#6b93ff", "#26324f"
KIT_INDEX, KIT_DATA, KIT_HEAD, KIT_NOTE, KIT_PILL = "#f28ff0", "#30dc9a", "#6b93ff", "#4ed37a", "#2a3bff"
KIT_TXT, KIT_CAPTION = "#e6e8ea", "#9aa0a8"


def _lines(x: float, y: float, text: str, fill: str, size: float, family: str, anchor: str = "middle") -> str:
    """Multi-line text: '\\n' separates lines, line height 1.4."""
    return "".join(f'<text x="{x}" y="{y + i * size * 1.4:.1f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-family="{family}">{t}</text>'
                   for i, t in enumerate(text.split("\n")))


def kit_board(w: float, h: float, title: str = "") -> str:
    """The board: near-black, 8 px corners, optional mono title top left. Open the <svg> yourself."""
    out = f'<rect width="{w}" height="{h}" rx="8" fill="{KIT_BOARD}"/>'
    if title:
        out += f'<text x="40" y="64" fill="{KIT_TXT}" font-size="30" font-weight="500" font-family="{MONO}">{title}</text>'
    return out


def kit_box(x: float, y: float, w: float, h: float, label: str = "", nested: bool = False, dashed: bool = False) -> str:
    """A card with a 2 px stroke; nested cards take the second shade; dashed = optional group."""
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{KIT_CARD_2 if nested else KIT_CARD}" stroke="{KIT_STROKE}" stroke-width="2"{dash}/>'
    if label:
        n = label.count("\n") + 1
        out += _lines(x + w / 2, y + h / 2 - (n - 1) * 10.5 + 5, label, KIT_TXT, 15, MONO)
    return out


def kit_table(x: float, y: float, w: float, rows: int = 5, dense: bool = True) -> str:
    """The kit table glyph: header of blue dashes, then rows of a pink index dot and green data dashes."""
    rh, head = 18, 22
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{head + rows * rh + 6}" rx="4" fill="{KIT_CARD_2}"/>',
           f'<rect x="{x}" y="{y}" width="{w}" height="{head}" rx="4" fill="{KIT_PAGE}"/>']
    cx = x + 12
    while cx < x + w - 24:
        seg = 18 if (cx - x) % 3 else 8
        out.append(f'<rect x="{cx:.1f}" y="{y + 9}" width="{seg}" height="3" rx="1.5" fill="{KIT_HEAD}"/>')
        cx += seg + 6
    for r in range(rows):
        ry = y + head + 8 + r * rh
        out.append(f'<rect x="{x + 10}" y="{ry}" width="6" height="3" rx="1.5" fill="{KIT_INDEX}"/>')
        dx = x + 26
        i = 0
        while dx < x + w - 14:
            if dense or (i + r) % 3 != 1:
                out.append(f'<rect x="{dx:.1f}" y="{ry}" width="6" height="3" rx="1.5" fill="{KIT_DATA}"/>')
            dx += 11
            i += 1
        if r < rows - 1:
            out.append(f'<path d="M{x} {ry + 11} H{x + w}" stroke="{KIT_BOARD}" stroke-width="1"/>')
    return "".join(out)


def kit_file(x: float, y: float, ext: str) -> str:
    """The kit file glyph: a blank page with a rounded lower-right corner and a violet-blue extension pill."""
    pw = max(8 * len(ext) + 14, 40)
    return (f'<path d="M{x} {y} h46 v52 q0 12 -12 12 h-34 z" fill="{KIT_PAGE}"/>'
            f'<rect x="{x + 23 - pw / 2:.1f}" y="{y + 72}" width="{pw}" height="20" rx="10" fill="{KIT_PILL}"/>'
            f'<text x="{x + 23}" y="{y + 86}" fill="#ffffff" font-size="11.5" text-anchor="middle" font-family="{MONO}">{ext}</text>')


def kit_arrow(x1: float, y1: float, x2: float, y2: float, dashed: bool = False) -> str:
    """A 2 px straight line with an open line arrowhead (the kit's ARROW_LINES cap)."""
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx = [x2 - 11 * math.cos(a + s * 0.5) for s in (-1, 1)]
    hy = [y2 - 11 * math.sin(a + s * 0.5) for s in (-1, 1)]
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return (f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{KIT_ARROW}" stroke-width="2"{dash}/>'
            f'<path d="M{hx[0]:.1f} {hy[0]:.1f} L{x2} {y2} L{hx[1]:.1f} {hy[1]:.1f}" fill="none" stroke="{KIT_ARROW}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')


def kit_note(x: float, y: float, text: str) -> str:
    """Green mono annotation above a step: why the step matters."""
    return _lines(x, y, text, KIT_NOTE, 13, MONO)


def kit_label(x: float, y: float, text: str) -> str:
    """Mono label below a glyph, in normal case."""
    return _lines(x, y, text, KIT_TXT, 14, MONO)


def kit_caption(x: float, y: float, text: str) -> str:
    """Small grey sans caption: group names above glyphs."""
    return _lines(x, y, text, KIT_CAPTION, 13, SANS)


def kit_example_svg() -> str:
    """Worked example in the kit style: files into Rerun, a reader and lenses, a layered recording, the catalog."""
    w, h = 1120, 420
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Kit-style board: mp4, csv and json files go through a reader and lenses into a layered rrd recording, then into the catalog for queries and training." xmlns="http://www.w3.org/2000/svg">',
         kit_board(w, h, "From capture files to a queryable catalog")]
    p.append(kit_note(170, 128, "every source file,\nkept lossless"))
    p.append(kit_box(48, 168, 244, 172, nested=False))
    p.append(kit_caption(170, 194, "Capture"))
    for i, ext in enumerate([".mp4", ".csv", ".json"]):
        p.append(kit_file(74 + i * 74, 214, ext))
    p.append(kit_arrow(304, 254, 364, 254))
    p.append(kit_note(470, 128, "readers yield chunks,\nlenses reshape them"))
    p.append(kit_box(376, 168, 188, 172, "Reader\n+ lenses"))
    p.append(kit_arrow(576, 254, 636, 254))
    p.append(kit_note(770, 128, "one recording id,\nstacked layers"))
    p.append(kit_box(648, 168, 244, 172))
    p.append(kit_caption(770, 194, "Layered .rrd"))
    for i in range(3):
        p.append(kit_table(668 + i * 70, 204, 64, rows=4, dense=i != 0))
    p.append(kit_label(770, 328, "base · align · keypoints"))
    p.append(kit_arrow(904, 254, 964, 254))
    p.append(kit_box(976, 168, 112, 172, "Catalog", dashed=True))
    p.append(kit_label(1032, 382, "query\n& train"))
    p.append(kit_arrow(1032, 346, 1032, 362))
    p.append("</svg>")
    return "".join(p)

# --- Measured charts: bars and before/after pairs, with a phone version ------------------------------------------------
# Only real, measured numbers; the exact value is written on every bar; a before/after chart always has a key.
# chart_html(spec) returns both versions; rerun.css shows the wide one above 600 px and the large-type one below:
#     chart_html({"bars": {"title": "Train time", "unit": "min", "rows": [["Brush 7k", "0.50"], ["Hexels V1 7k", 3.61]]}})
# A value given as a string keeps its text on the bar ("0.50", not 0.5).
#     chart_html({"pairs": {"title": "Kernel time", "unit": "ms", "a": "CUDA", "b": "Mojo", "rows": [["solve", 7139, 4615]]}})
import html as _html
import re as _re

INK, DIM_C, LINE_C, BAR, BAR_A = TXT, DIM, "#2f3338", ROW, "#5e6672"
FONT = 'font-family="' + SANS + '"'
MONO_ATTR = 'font-family="' + MONO + '"'


def _caps(title: str) -> str:
    """A chart title is a label: mono UPPERCASE, and a measured panel says so (DESIGN.md: `MEASURED · <what>`)."""
    return (title if title.upper().startswith("MEASURED") else f"Measured · {title}").upper()
e = _html.escape
class Measured(float):
    """A number that keeps its measured text: chart_html() turns a string value ("0.50", "1,048") into one."""

    text: str

    def __new__(cls, text: str) -> "Measured":
        obj = super().__new__(cls, float(text.replace(",", "")))
        obj.text = text
        return obj


def _num(v: float) -> str:
    """The value as measured: a string value as written ("0.50"); integers plain (thousands separated); others with up to
    4 significant digits (18.98, 114.5, 0.035). Pass strings when trailing zeros matter."""
    if isinstance(v, Measured):
        return v.text
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def _fit(s: str, size: float, width: float) -> str:
    """Cut a label to about the width (Arial-like 0.55 em per character)."""
    n = max(4, int(width / (size * 0.55)))
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _wrap(s: str, size: float, width: float) -> list[str]:
    n, out, cur = max(6, int(width / (size * 0.55))), [], ""
    for w in s.split():
        if cur and len(cur) + 1 + len(w) > n:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return out + ([cur] if cur else [])


def _t(x, y, s, size, fill=INK, anchor="start", weight=400, mono=False):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
            f'{MONO_ATTR if mono else FONT}>{e(s)}</text>')


def _frame(w, h, body, title=""):
    label = f' role="img" aria-label="{e(title)}"' if title else ' aria-hidden="true"'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}"{label}>'
            f'<rect width="{w}" height="{h}" rx="8" fill="{BG}"/>{body}</svg>')


def _key(items: list[tuple[str, str]], x: float, y: float, fs: float) -> str:
    """A one-line key: a colour swatch and its word per series."""
    out = ""
    for col, word in items:
        out += (f'<rect x="{x:.1f}" y="{y - fs * 0.72:.1f}" width="{fs * 0.75:.1f}" height="{fs * 0.62:.1f}" rx="2" fill="{col}"/>'
                + _t(x + fs, y, word.upper(), fs * 0.8, DIM_C, mono=True))  # legend words are labels: mono uppercase
        x += fs * 2.2 + len(word) * fs * 0.8 * 0.62
    return out


def bars(d: dict, thumb: bool, phone: bool = False) -> str:
    """One bar per row, the exact value at its end. Thumbnails put each label above its bar, so no label is cut."""
    rows = d["rows"]
    vmax = d.get("max") or max(r[1] for r in rows) or 1
    txt = lambda r: r[2] if len(r) > 2 and r[2] else f'{_num(r[1])} {d.get("unit", "")}'.strip()  # noqa: E731
    if thumb or phone:
        w, h, pad, fs = 480, 300, 20, 24
        key = d.get("key") or (d.get("title") if phone else None)
        top = pad + (fs + 12 if key else 0)
        if phone:
            rh = 66.0
            h = int(top + rh * len(rows) + pad)
        else:
            rh = min(78.0, (h - top - pad) / len(rows))
            top += (h - top - pad - rh * len(rows)) / 2
        lab = _caps(key) if key and key == d.get("title") else (key or "").upper()
        body = _t(pad, pad + fs * 0.8, _fit(lab, fs * 0.7, w - 2 * pad), fs * 0.7, DIM_C, mono=True) if key else ""
        for i, r in enumerate(rows):
            y = top + i * rh
            body += _t(pad, y + fs * 0.9, _fit(r[0], fs, w - 2 * pad), fs, INK)
            bw = w - 2 * pad - 150
            L = max(3.0, bw * r[1] / vmax)
            by = y + fs * 1.15
            bh = max(10.0, min(20.0, rh - fs * 1.35))
            body += f'<rect x="{pad}" y="{by:.1f}" width="{L:.1f}" height="{bh:.1f}" rx="3" fill="{BAR}"/>'
            body += _t(pad + L + 10, by + bh * 0.85, txt(r), fs * 0.92, INK, mono=True)
        return _frame(w, h, body, d.get("title", ""))
    w, fs, rh, pad = 960, 15, 34, 24
    top = pad + (fs + 14 if d.get("title") else 0)
    h = top + rh * len(rows) + pad
    lab_w = w * 0.3
    bw = w - lab_w - pad * 2 - 140
    body = _t(pad, pad + fs, _caps(d["title"]), fs - 2, DIM_C, mono=True) if d.get("title") else ""
    for i, r in enumerate(rows):
        y = top + i * rh
        body += _t(pad, y + rh * 0.62, _fit(r[0], fs, lab_w - 10), fs, INK)
        L = max(2.0, bw * r[1] / vmax)
        body += f'<rect x="{pad + lab_w:.1f}" y="{y + rh * 0.2:.1f}" width="{L:.1f}" height="{rh * 0.56:.1f}" rx="3" fill="{BAR}"/>'
        body += _t(pad + lab_w + L + 10, y + rh * 0.62, txt(r), fs, INK, mono=True)
    return _frame(w, h, body, d.get("title", ""))


def pairs(d: dict, thumb: bool, phone: bool = False) -> str:
    """Before/after (or baseline/ours) per row: a grey bar over a blue bar, the exact values at their ends, and a key line."""
    rows = d["rows"]
    vmax = d.get("max") or max(max(r[1], r[2]) for r in rows) or 1
    unit = d.get("unit", "")
    a, b = d.get("a", "before"), d.get("b", "after")
    if thumb or phone:
        w, h, pad, fs = 480, 300, 18, 22
        body = _t(pad, pad + fs * 0.85, _fit(_caps(d["title"]), fs * 0.7, w - 2 * pad), fs * 0.7, DIM_C, mono=True) if phone and d.get("title") else ""
        ky = pad + fs * 0.85 + (fs + 12 if phone and d.get("title") else 0)
        body += _key([(BAR_A, a), (BAR, b)], pad, ky, fs)
        top = ky + 14
        rh = 84.0 if phone else (h - top - pad) / len(rows)
        if phone:
            h = int(top + rh * len(rows) + pad)
        for i, r in enumerate(rows):
            y = top + i * rh
            body += _t(pad, y + fs * 0.85, _fit(r[0], fs, w - 2 * pad), fs, INK)
            bh = max(8.0, min(16.0, (rh - fs * 1.2) / 2 - 2))
            for j, (v, col) in enumerate(((r[1], BAR_A), (r[2], BAR))):
                L = max(3.0, (w - 2 * pad - 150) * v / vmax)
                by = y + fs * 1.1 + j * (bh + 3)
                body += f'<rect x="{pad}" y="{by:.1f}" width="{L:.1f}" height="{bh:.1f}" rx="2" fill="{col}"/>'
                body += _t(pad + L + 8, by + bh * 0.95, f"{_num(v)} {unit}".strip(), min(fs * 0.85, bh + 6), INK if j else DIM, mono=True)
        return _frame(w, h, body, d.get("title", ""))
    w, fs, rh, pad = 960, 15, 52, 24
    top = pad + (fs + 14 if d.get("title") else 0) + fs + 16
    h = top + rh * len(rows) + pad
    lab_w = w * 0.3
    bw = w - lab_w - pad * 2 - 150
    body = _t(pad, pad + fs, _caps(d["title"]), fs - 2, DIM_C, mono=True) if d.get("title") else ""
    body += _key([(BAR_A, a), (BAR, b)], pad, pad + (fs + 14 if d.get("title") else 0) + fs, fs - 1)
    for i, r in enumerate(rows):
        y = top + i * rh
        body += _t(pad, y + rh * 0.55, _fit(r[0], fs, lab_w - 10), fs, INK)
        for j, (v, col) in enumerate(((r[1], BAR_A), (r[2], BAR))):
            L = max(2.0, bw * v / vmax)
            by = y + rh * (0.12 + 0.4 * j)
            body += f'<rect x="{pad + lab_w:.1f}" y="{by:.1f}" width="{L:.1f}" height="{rh * 0.32:.1f}" rx="3" fill="{col}"/>'
            body += _t(pad + lab_w + L + 8, by + rh * 0.27, f"{_num(v)} {unit}".strip(), fs - 2, INK if j else DIM, mono=True)
    return _frame(w, h, body, d.get("title", ""))




def chart_svg(spec: dict, phone: bool = False) -> str:
    kind = "bars" if "bars" in spec else "pairs"
    d = dict(spec[kind])
    n = 2 if kind == "bars" else 3  # bars: [label, value, optional text]; pairs: [label, a, b]
    d["rows"] = [[r[0], *(Measured(x) if isinstance(x, str) else x for x in r[1:n]), *r[n:]] for r in d["rows"]]
    return (bars if kind == "bars" else pairs)(d, False, phone=phone)


def chart_html(spec: dict) -> str:
    """The wide chart and its phone version; pair with the .rr-chart rules in rerun.css."""
    return f'<div class="rr-chart"><div class="rr-chart-w">{chart_svg(spec)}</div><div class="rr-chart-p">{chart_svg(spec, phone=True)}</div></div>'
