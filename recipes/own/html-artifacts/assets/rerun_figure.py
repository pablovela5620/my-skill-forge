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
"""

BG, PANEL, HEAD, LINE, TXT, DIM = "#0e0f10", "#1a1c1e", "#25282b", "#5b6066", "#e6e8ea", "#8a9097"
ROW, TIME, COMP = "#22b5e6", "#de7df3", "#6cefb4"
ROW_BG, TIME_BG, COMP_BG = "#11313c", "#3a2742", "#1d4536"
GREEN = "#3fe0a0"
MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"
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
