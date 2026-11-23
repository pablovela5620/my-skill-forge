"""Render a live progress page from its state file, in the Rerun house style, and update the state from the shell.

The page always has the same shape, top to bottom: the work-in-progress badge and the "needs you" list, the stage
diagram coloured by status, one chart per target metric against its goal, the newest evidence per stage, and a short
timeline. A finished state renders without the badge. See references/progress-pages.md.

Run with pixi (no install into any project):
    pixi exec -s python=3.13 -s pyserde -s orjson -s tomli-w -s tyro python <skill-dir>/assets/progress/progress_page.py render state.json index.html
"""

import datetime
import html
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

import progress_state as ps  # noqa: E402
from progress_state import Evidence, Sample, Stage, State, Target  # noqa: E402

ASSETS: Path = Path(__file__).resolve().parent.parent
# The bundled Tabler set exists in the built package; HTML_ARTIFACTS_TABLER points elsewhere (for example from the recipe tree).
TABLER: Path = Path(os.environ.get("HTML_ARTIFACTS_TABLER", str(ASSETS / "tabler-outline")))
e = html.escape

STATUS_GLYPH: dict[str, str] = {"todo": "circle-dashed", "running": "progress", "done": "circle-check", "blocked": "circle-half-2", "failed": "circle-x"}
STATUS_WORD: dict[str, str] = {"todo": "To do", "running": "Running", "done": "Done", "blocked": "Blocked", "failed": "Failed"}
KIND_GLYPH: dict[str, str] = {"info": "point", "result": "chart-bar", "problem": "alert-triangle", "decision": "git-branch"}
TIMELINE_SHOWN: int = 12


def glyph(name: str, size: int = 16) -> str:
    """A Tabler outline glyph from the bundled set, inline; "" when the set is not installed (the recipe tree)."""
    path: Path = TABLER / f"{name}.svg"
    if not path.exists():
        return ""
    src: str = path.read_text()
    body: str = src[src.index(">") + 1: src.rindex("</svg>")].replace('<path stroke="none" d="M0 0h24v24H0z" fill="none"/>', "")
    return (f'<svg aria-hidden="true" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


def when(t: str) -> str:
    return f"step {t[1:]}" if ps.is_step(t) else t.replace("T", " ").replace("Z", " UTC")


def minutes(t: str) -> float:
    """A sortable position: the step number for a replay, minutes since the epoch for a clock time."""
    if ps.is_step(t):
        return float(t[1:])
    return datetime.datetime.strptime(t.replace("Z", ""), "%Y-%m-%dT%H:%M" if t.count(":") == 1 else "%Y-%m-%dT%H:%M:%S").timestamp() / 60


def decimals(values: list[float]) -> int:
    """The decimals a target's numbers are written with: the most any value needs (4.5 -> 1), so 5.0 stays 5.0."""
    return max((len(repr(float(v)).split(".")[1].rstrip("0")) for v in values if "e" not in repr(float(v))), default=0)


def fmt(v: float, d: int | None = None) -> str:
    if d is None:
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    return f"{v:,.{d}f}"


# ---------------------------------------------------------------- blocks
def badge(state: State) -> str:
    if state.finished:
        return ""
    running: list[str] = [s.title for s in state.stages if s.status == "running"]
    now: str = f'<span class="rr-wip-now">Now: {e(", ".join(running))}</span>' if running else ""
    return (f'<div class="rr-wip" data-wip role="status">{glyph("progress")}Work in progress'
            f'<span>updated <time datetime="{e(state.updated)}">{e(when(state.updated))}</time></span>{now}</div>')


def needs(state: State) -> str:
    if not state.needs:
        body: str = f'<p class="pp-none">{glyph("circle-check")}Nothing waits on you.</p>'
    else:
        items: str = "".join(f'<li><span>{e(n.text)}</span><time>since {e(when(n.since))}</time></li>' for n in state.needs)
        body = f'<div class="rr-callout warn"><p class="rr-label on">Needs you</p><ul class="pp-needs">{items}</ul></div>'
    return f'<section id="needs" aria-label="Needs you">{body}</section>'


def stage_node(s: Stage) -> str:
    note: str = f"<small>{e(s.note)}</small>" if s.note else ""
    return (f'<li class="pp-stage" data-status="{s.status}"><b>{e(s.title)}</b>'
            f'<span class="pp-st">{glyph(STATUS_GLYPH[s.status], 14)}{STATUS_WORD[s.status]}</span>{note}</li>')


def stages(state: State) -> str:
    done: int = sum(s.status == "done" for s in state.stages)
    return (f'<section id="stages"><h2 class="rr-label">Stages · {done} of {len(state.stages)} done</h2>'
            f'<ol class="pp-flow">{"".join(stage_node(s) for s in state.stages)}</ol></section>')


def line_chart(target: Target, samples: list[Sample], phone: bool) -> str:
    """Measured values over time against the goal: points joined by a line, the goal dashed, the start dotted."""
    w, h, fs, pad = (480, 320, 22, 22) if phone else (960, 300, 15, 28)
    left, right, top, bottom = pad + (70 if phone else 60), w - pad - (40 if phone else 120), pad + fs * 2.6, h - pad - fs * 1.6
    vals: list[float] = [s.value for s in samples] + [target.goal, target.start]
    d: int = decimals(vals)
    lo, hi = min(vals), max(vals)
    span: float = (hi - lo) or 1.0
    lo, hi = lo - span * 0.08, hi + span * 0.08
    ts: list[float] = [minutes(s.t) for s in samples]
    t0, t1 = (min(ts), max(ts)) if ts else (0.0, 1.0)
    tspan: float = (t1 - t0) or 1.0

    def x(t: float) -> float:
        return left + (right - left) * ((t - t0) / tspan if len(ts) > 1 else 0.5)

    def y(v: float) -> float:
        return bottom - (bottom - top) * (v - lo) / (hi - lo)

    mono: str = "font-family=\"'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace\""
    title: str = f"MEASURED · {target.name} ({target.unit})".upper()
    out: list[str] = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{e(title)}"><rect width="{w}" height="{h}" rx="8" fill="#0e0f10"/>',
                      f'<text x="{pad}" y="{pad + fs * 0.6}" fill="#8a9097" font-size="{fs * 0.75}" {mono}>{e(title)}</text>']
    marks: list[tuple[float, str, str, str]] = [(target.goal, "GOAL", "8 6", "#3fe0a0")]
    if target.start == target.goal:
        marks = [(target.goal, "GOAL = START", "8 6", "#3fe0a0")]  # "no worse than where we began"
    else:
        marks.append((target.start, "START", "2 5", "#8a9097"))
    for v, word, dash, col in marks:
        out.append(f'<line x1="{left}" x2="{right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{col}" stroke-width="1.4" stroke-dasharray="{dash}"/>')
        out.append(f'<text x="{right + 8}" y="{y(v) + fs * 0.3:.1f}" fill="{col}" font-size="{fs * 0.7}" {mono}>{word} {e(fmt(v, d))}</text>' if not phone
                   else f'<text x="{left}" y="{y(v) - 6:.1f}" fill="{col}" font-size="{fs * 0.7}" {mono}>{word} {e(fmt(v, d))}</text>')
    out.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{bottom}" stroke="#2f3338"/><line x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}" stroke="#2f3338"/>')
    if samples:
        pts: str = " ".join(f"{x(t):.1f},{y(s.value):.1f}" for t, s in zip(ts, samples, strict=True))
        out.append(f'<polyline points="{pts}" fill="none" stroke="#22b5e6" stroke-width="2.2"/>')
        for i, (t, s) in enumerate(zip(ts, samples, strict=True)):
            last: bool = i == len(samples) - 1
            out.append(f'<circle cx="{x(t):.1f}" cy="{y(s.value):.1f}" r="{5 if last else 3.5}" fill="#22b5e6"/>')
            if last or not phone:
                out.append(f'<text x="{x(t):.1f}" y="{y(s.value) - 10:.1f}" fill="#e6e8ea" font-size="{fs * (0.85 if last else 0.7)}" '
                           f'text-anchor="middle" {mono}>{e(fmt(s.value, d))}</text>')
        out.append(f'<text x="{left}" y="{bottom + fs * 1.3}" fill="#8a9097" font-size="{fs * 0.7}" {mono}>{e(when(samples[0].t))}</text>')
        if len(samples) > 1:
            out.append(f'<text x="{right}" y="{bottom + fs * 1.3}" fill="#8a9097" font-size="{fs * 0.7}" text-anchor="end" {mono}>'
                       f'{e(when(samples[-1].t)[11:] if not ps.is_step(samples[-1].t) and samples[-1].t[:10] == samples[0].t[:10] else when(samples[-1].t))}</text>')
    out.append("</svg>")
    return "".join(out)


def metric(target: Target, samples: list[Sample]) -> str:
    mine: list[Sample] = sorted((s for s in samples if s.target == target.id), key=lambda s: minutes(s.t))
    latest: float = mine[-1].value if mine else target.start
    d: int = decimals([s.value for s in mine] + [target.start, target.goal])
    better: bool = latest >= target.goal if target.higher_is_better else latest <= target.goal
    tiles: str = (f'<div class="rr-stats"><div class="rr-stat"><b>{e(fmt(latest, d))}</b><span class="rr-label">now, {e(target.unit)}</span></div>'
                  f'<div class="rr-stat"><b>{e(fmt(target.goal, d))}</b><span class="rr-label">goal{" · reached" if better else ""}</span></div>'
                  f'<div class="rr-stat"><b>{e(fmt(target.start, d))}</b><span class="rr-label">start</span></div></div>')
    chart: str = (f'<div class="rr-chart"><div class="rr-chart-w">{line_chart(target, mine, False)}</div>'
                  f'<div class="rr-chart-p">{line_chart(target, mine, True)}</div></div>')
    return f'<figure class="pp-metric"><figcaption class="rr-label on">{e(target.name)}</figcaption>{tiles}{chart}</figure>'


def metrics(state: State) -> str:
    body: str = "".join(metric(t, state.samples) for t in state.targets) or '<p class="pp-none">No target metric.</p>'
    return f'<section id="metrics"><h2 class="rr-label">Metrics against the goal</h2>{body}</section>'


def evidence_item(ev: Evidence, stage: Stage) -> str:
    if ev.kind == "image":
        media: str = f'<img src="{e(ev.src)}" alt="{e(ev.caption)}" loading="lazy">'
    elif ev.kind == "video":
        poster: str = f' poster="{e(ev.poster)}"' if ev.poster else ""
        media = f'<video src="{e(ev.src)}"{poster} muted loop playsinline controls preload="none"></video>'
    elif ev.poster:
        media = f'<a href="{e(ev.src)}"><img src="{e(ev.poster)}" alt="{e(ev.caption)}" loading="lazy"></a>'
    else:
        media = f'<a class="pp-link" href="{e(ev.src)}">{glyph("file-3d" if ev.kind == "rerun" else "link", 20)}{e(ev.src)}</a>'
    return (f'<figure class="pp-ev" data-status="{stage.status}"><p class="rr-label">{e(stage.title)} · {e(when(ev.t))}</p>{media}'
            f'<figcaption>{e(ev.caption)}</figcaption></figure>')


def evidence(state: State) -> str:
    newest: dict[str, Evidence] = {}
    for ev in sorted(state.evidence, key=lambda v: minutes(v.t)):
        newest[ev.stage] = ev
    items: str = "".join(evidence_item(newest[s.id], s) for s in state.stages if s.id in newest)
    body: str = f'<div class="pp-evs">{items}</div>' if items else '<p class="pp-none">No evidence yet.</p>'
    return f'<section id="evidence"><h2 class="rr-label">Newest evidence per stage</h2>{body}</section>'


def timeline(state: State) -> str:
    titles: dict[str, str] = {s.id: s.title for s in state.stages}
    events = sorted(state.events, key=lambda v: minutes(v.t), reverse=True)

    def row(ev: ps.Event) -> str:
        stage: str = f'<span class="pp-tag">{e(titles[ev.stage])}</span>' if ev.stage else ""
        return (f'<li data-kind="{ev.kind}"><time>{e(when(ev.t))}</time><span class="pp-kind">{glyph(KIND_GLYPH[ev.kind], 14)}{ev.kind}</span>'
                f'{stage}<span class="pp-text">{e(ev.text)}</span></li>')

    shown: str = "".join(row(v) for v in events[:TIMELINE_SHOWN])
    older: str = (f'<details class="pp-older"><summary>{len(events) - TIMELINE_SHOWN} older events</summary><ol class="pp-tl">'
                  f'{"".join(row(v) for v in events[TIMELINE_SHOWN:])}</ol></details>') if len(events) > TIMELINE_SHOWN else ""
    return f'<section id="timeline"><h2 class="rr-label">Timeline</h2><ol class="pp-tl">{shown}</ol>{older}</section>'


CSS: str = """
.pp-wrap{max-width:1280px;margin:0 auto;padding:clamp(14px,3vw,40px)}
.pp-wrap>header h1{margin:6px 0 6px} .pp-goal{margin:0;color:var(--rr-text)} .pp-meta{margin:4px 0 0;color:var(--rr-dim);font:500 12px var(--rr-mono)}
.pp-wrap section{border-top:1px solid var(--rr-border);padding:22px 0 10px}
.pp-wrap section h2{margin:0 0 12px}
.pp-replay{margin:12px 0 0} .pp-replay p{margin:0}
.pp-none{display:flex;align-items:center;gap:6px;color:var(--rr-dim);margin:0}
.pp-needs{margin:6px 0 0;padding-left:18px} .pp-needs li{margin:4px 0} .pp-needs time{margin-left:8px;color:var(--rr-dim);font:500 12px var(--rr-mono)}
.pp-flow{list-style:none;margin:0;padding:0;display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));counter-reset:st}
.pp-stage{position:relative;border:1px solid var(--rr-border-strong);border-radius:8px;padding:10px 12px;display:grid;gap:4px;align-content:start;background:var(--rr-panel);counter-increment:st}
.pp-stage b{font:500 12px/1.33 var(--rr-mono);letter-spacing:.0125em;text-transform:uppercase;color:var(--rr-fg)}
.pp-stage b::before{content:counter(st) " · ";color:var(--rr-dim)}
.pp-stage small{color:var(--rr-text);font-size:.82rem}
.pp-st{display:inline-flex;align-items:center;gap:5px;font:500 11px var(--rr-mono);text-transform:uppercase;letter-spacing:.0125em}
[data-status=done]{--pp-k:var(--rr-good)} [data-status=running]{--pp-k:var(--rr-brand-blue)} [data-status=blocked]{--pp-k:var(--rr-warn)}
[data-status=failed]{--pp-k:var(--rr-bad)} [data-status=todo]{--pp-k:var(--rr-dim)}
.pp-stage{border-color:var(--pp-k)} .pp-stage[data-status=todo]{border-style:dashed} .pp-st{color:var(--pp-k)}
.pp-metric{margin:0 0 18px;display:grid;gap:10px} .pp-metric .rr-stats{max-width:620px;grid-template-columns:repeat(3,minmax(0,1fr))}
.pp-evs{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(min(100%,340px),1fr))}
.pp-ev{margin:0;background:var(--rr-figure);border-radius:8px;overflow:hidden;border-top:2px solid var(--pp-k)}
.pp-ev .rr-label{margin:8px 12px} .pp-ev img,.pp-ev video{display:block;width:100%;height:auto} .pp-ev figcaption{padding:8px 12px;font-size:.86rem}
.pp-link{display:flex;align-items:center;gap:8px;min-height:44px;padding:0 12px;overflow-wrap:anywhere}
.pp-tl{list-style:none;margin:0;padding:0} .pp-tl li{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline;padding:8px 0;border-bottom:1px solid var(--rr-border)}
.pp-tl time{font:500 12px var(--rr-mono);color:var(--rr-dim);min-width:150px}
.pp-kind{display:inline-flex;align-items:center;gap:4px;font:500 11px var(--rr-mono);text-transform:uppercase;color:var(--rr-dim);min-width:84px}
[data-kind=result] .pp-kind{color:var(--rr-good)} [data-kind=problem] .pp-kind{color:var(--rr-bad)} [data-kind=decision] .pp-kind{color:var(--rr-warn)}
.pp-tag{font:500 11px var(--rr-mono);text-transform:uppercase;border:1px solid var(--rr-border-strong);border-radius:999px;padding:1px 8px;color:var(--rr-text)}
.pp-text{flex:1 1 260px;min-width:0}
.pp-older summary{min-height:44px;display:flex;align-items:center;cursor:pointer;font:500 12px var(--rr-mono);text-transform:uppercase;color:var(--rr-dim)}
"""


def render(state: State) -> str:
    """The whole page as one HTML string."""
    css: str = (ASSETS / "rerun.css").read_text()
    status: str = "finished" if state.finished else "running"
    replay: str = (f'<div class="rr-callout pp-replay"><p class="rr-label on">Replay</p><p>{e(state.replay)}</p></div>' if state.replay else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,"><title>{e(state.title)}</title><style>{css}{CSS}</style></head>
<body><main class="pp-wrap">{badge(state)}
<header><p class="rr-label">Live progress · {status} · {e(state.owner)}</p><h1>{e(state.title)}</h1><p class="pp-goal">{e(state.goal)}</p>
<p class="pp-meta">started {e(when(state.started))} · updated {e(when(state.updated))}</p>{replay}</header>
{needs(state)}{stages(state)}{metrics(state)}{evidence(state)}{timeline(state)}</main></body></html>"""


# ---------------------------------------------------------------- command line
@dataclass(frozen=True, slots=True)
class Render:
    """Render the page from the state file."""

    state: Path
    """The state file."""
    out: Path
    """The HTML file to write."""


@dataclass(frozen=True, slots=True)
class StageCmd:
    """Set a stage's status (and its one-line note)."""

    state: Path
    """The state file."""
    stage: str
    """Stage id."""
    status: ps.Status
    """New status."""
    note: str | None = None
    """One-line note under the node; omit to keep the old one."""
    t: str | None = None
    """UTC time (2026-11-23T09:00Z) or, in a replay, a step number (#3); default now."""


@dataclass(frozen=True, slots=True)
class SampleCmd:
    """Add a measured value of a target."""

    state: Path
    """The state file."""
    target: str
    """Target id."""
    value: float
    """The measured value."""
    stage: str = ""
    """The stage that produced it."""
    t: str | None = None
    """UTC time (2026-11-23T09:00Z) or, in a replay, a step number (#3); default now."""


@dataclass(frozen=True, slots=True)
class EventCmd:
    """Add one line to the timeline."""

    state: Path
    """The state file."""
    text: str
    """What happened, one line."""
    kind: ps.EventKind = "info"
    """info, result, problem or decision."""
    stage: str = ""
    """Stage id."""
    t: str | None = None
    """UTC time (2026-11-23T09:00Z) or, in a replay, a step number (#3); default now."""


@dataclass(frozen=True, slots=True)
class EvidenceCmd:
    """Add a picture or link for a stage."""

    state: Path
    """The state file."""
    stage: str
    """Stage id."""
    kind: ps.EvidenceKind
    """image, video, rerun or link."""
    src: str
    """Path relative to the page, or a same-origin URL."""
    caption: str
    """How to read it, one line."""
    poster: str = ""
    """Still for a video or recording."""
    t: str | None = None
    """UTC time (2026-11-23T09:00Z) or, in a replay, a step number (#3); default now."""


@dataclass(frozen=True, slots=True)
class NeedCmd:
    """Replace the "needs you" list; no text clears it."""

    state: Path
    """The state file."""
    text: tuple[str, ...] = ()
    """The open asks, one line each."""
    t: str | None = None
    """UTC time (2026-11-23T09:00Z) or, in a replay, a step number (#3); default now."""


@dataclass(frozen=True, slots=True)
class FinishCmd:
    """Mark the task finished: the next render has no badge."""

    state: Path
    """The state file."""
    t: str | None = None
    """UTC time (2026-11-23T09:00Z) or, in a replay, a step number (#3); default now."""


Command = Render | StageCmd | SampleCmd | EventCmd | EvidenceCmd | NeedCmd | FinishCmd


def run(cmd: Command) -> None:
    match cmd:
        case Render():
            cmd.out.write_text(render(ps.load(cmd.state)))
            print(f"wrote {cmd.out}")
        case StageCmd():
            ps.set_stage(cmd.state, cmd.stage, cmd.status, cmd.note, t=cmd.t)
        case SampleCmd():
            ps.add_sample(cmd.state, cmd.target, cmd.value, t=cmd.t, stage=cmd.stage)
        case EventCmd():
            ps.add_event(cmd.state, cmd.text, kind=cmd.kind, stage=cmd.stage, t=cmd.t)
        case EvidenceCmd():
            ps.add_evidence(cmd.state, cmd.stage, cmd.kind, cmd.src, cmd.caption, poster=cmd.poster, t=cmd.t)
        case NeedCmd():
            ps.set_needs(cmd.state, list(cmd.text), t=cmd.t)
        case FinishCmd():
            ps.finish(cmd.state, t=cmd.t)


def main() -> None:
    import tyro

    commands: dict[str, type] = {"render": Render, "stage": StageCmd, "sample": SampleCmd, "event": EventCmd,
                                 "evidence": EvidenceCmd, "need": NeedCmd, "finish": FinishCmd}
    cmd: Command = tyro.extras.subcommand_cli_from_dict(commands)
    try:
        run(cmd)
    except ValueError as err:  # a rejected update: say why, leave the file as it was
        print(f"rejected: {err}", file=sys.stderr)
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
