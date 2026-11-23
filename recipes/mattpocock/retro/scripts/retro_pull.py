"""Query the agent-traces catalog for a retro window and write the leads as one Markdown summary.

The catalog is the store: this script keeps no copy of its tables. It runs one filtered reader per
entity family per dataset, analyses the rows in memory, and prints the summary; with `--out` it
also writes `summary.md` there, and nothing else. Every lead carries its dataset, segment ID and
UTC time, so a follow-up query can target it. Run it in the agent-traces pixi environment:

    cd ~/agent-traces-staging/em && pixi run --frozen -e agent-traces python -I \
        ~/.claude/skills/retro/scripts/retro_pull.py --since 2026-11-22 --until 2026-11-23 --out /tmp/retro

A bound without a UTC offset is UTC: `--since 2026-11-22T00:00-06:00` starts at local midnight.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import tyro

DATASETS: tuple[str, ...] = ("agent-traces-claude", "agent-traces-claude-work", "agent-traces-codex", "agent-traces-codex-work")
INJECTED: re.Pattern[str] = re.compile(
    r"^\s*(<|\[Request interrupted|Caveat:|This session is being continued|Base directory for this skill|Stop hook|\[SYSTEM|# AGENTS\.md)"
)
CORRECTION: re.Pattern[str] = re.compile(
    r"\b(wrong|broken|doesn'?t work|not working|still|again|why (did|is|are|does|don'?t)|you (didn'?t|forgot|missed)"
    r"|did you (even|actually|check|look)|you sure|i (said|asked|told)|stop|never|don'?t|no,|nope|too (much|many|long|big|small)"
    r"|wall of text|missing|not (what|right)|instead|should(n'?t)? (be|have)|looks? (bad|off|weird)|weird|confus|horrible|ugly)",
    re.I,
)
SCRIPT: re.Pattern[str] = re.compile(r"([\w.-]+\.(?:py|sh|js|ts|mjs))\b")
LAUNCHERS: frozenset[str] = frozenset({"python", "python3", "bash", "sh", "node", "pixi", "uv", "timeout", "nohup", "time", "env"})
HOME: re.Pattern[str] = re.compile(r"^(/home|/Users)/[^/]+")
TIME: str = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True, slots=True)
class Config:
    """Window and output for one run."""

    since: str
    """Window start: ISO date or datetime, UTC unless it carries an offset."""
    until: str
    """Window end (exclusive), same format as `since`."""
    out: Path | None = None
    """Directory for `summary.md`; without it the summary goes to stdout only."""
    catalog: str = "rerun+http://dgx-spark:9876"
    """Catalog URL."""
    datasets: tuple[str, ...] = DATASETS
    """Datasets to read; missing ones are reported and skipped."""
    cwd: str | None = None
    """Keep only sessions whose working directory contains this substring."""
    text_chars: int = 400
    """Characters kept from each tool input and result."""
    prompt_chars: int = 240
    """Characters of each human prompt shown in the summary."""


@dataclass(frozen=True, slots=True)
class Session:
    """One segment (session) with activity in the window."""

    dataset: str
    """Dataset name, for example `agent-traces-claude`."""
    segment: str
    """Segment (session) ID."""
    host: str
    """Machine the session ran on."""
    cwd: str
    """Working directory."""
    title: str
    """Session title, empty when unknown."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """One main-agent `/conversation/user` row in the window."""

    dataset: str
    """Dataset name."""
    segment: str
    """Segment (session) ID."""
    wall: datetime
    """Event time, naive UTC."""
    text: str
    """Full prompt text."""
    human: bool
    """True when a person typed it, not an injected or inter-agent message."""
    correction_hint: bool
    """True when the correction regex matches; a first pass, not a verdict."""


@dataclass(frozen=True, slots=True)
class ToolRow:
    """One `/tools` row: a model tool call or one part of its result."""

    dataset: str
    """Dataset name."""
    segment: str
    """Segment (session) ID."""
    agent_id: str
    """Empty string for the main agent."""
    call_id: str
    """Tool call ID shared by the call and its results."""
    tool: str
    """Tool name, for example `Bash` or `mcp/paseo/wait_for_agent`."""
    phase: str
    """`call` or `result`."""
    wall: datetime
    """Event time, naive UTC."""
    is_error: bool
    """True when the result reports an error."""
    elapsed_ms: float | None
    """Recorded call-to-result time, when known."""
    input: str
    """Truncated `input_json`."""
    result: str
    """Truncated `tool_use_result_json`."""


@dataclass(frozen=True, slots=True)
class Call:
    """A tool call joined with its results."""

    dataset: str
    """Dataset name."""
    segment: str
    """Segment (session) ID."""
    agent_id: str
    """Empty string for the main agent."""
    call_id: str
    """Tool call ID; empty for Codex native command failures."""
    tool: str
    """Tool name."""
    key: str
    """Tool name, plus the script or program for shell calls (`Bash:rrd_to_video.py`)."""
    start: datetime
    """Call time, naive UTC."""
    latency_s: float | None
    """Recorded elapsed time, else result time minus call time; None without a result."""
    is_error: bool
    """True when any result part reports an error."""
    input: str
    """Truncated call input."""
    result: str
    """Truncated result: the first error part when there is one."""


@dataclass(frozen=True, slots=True)
class Streak:
    """Consecutive failed calls of one key by one agent: a retry loop."""

    dataset: str
    """Dataset name."""
    segment: str
    """Segment (session) ID."""
    agent_id: str
    """Empty string for the main agent."""
    key: str
    """Call key that kept failing."""
    length: int
    """Number of failed calls in a row."""
    start: datetime
    """Time of the first failure."""
    recovered: bool
    """True when the next call of the same key succeeded."""
    first_error: str
    """Result text of the first failure."""


@dataclass(frozen=True, slots=True)
class Evidence:
    """Everything one window query returns, held in memory only."""

    since: datetime
    """Window start, naive UTC."""
    until: datetime
    """Window end (exclusive), naive UTC."""
    coverage: list[str]
    """One line per dataset, plus a `LAGGING` line when the catalog ends before the window does."""
    sessions: list[Session]
    """Sessions with activity in the window."""
    prompts: list[Prompt]
    """Main-agent user rows in the window."""
    calls: list[Call]
    """Tool calls in the window, subagents included."""


def parse_instant(text: str) -> datetime:
    """Parse an ISO date or datetime into naive UTC, the catalog's `wall` convention."""
    instant: datetime = datetime.fromisoformat(text)
    return instant.astimezone(UTC).replace(tzinfo=None) if instant.tzinfo else instant


def is_human(text: str | None) -> bool:
    """True for a prompt a person typed, not an injected, inter-agent or heartbeat message."""
    return bool(text) and not INJECTED.match(text) and "paseo-system" not in text[:200]


def is_correction(text: str) -> bool:
    """True when a prompt reads like pushback; a first-pass hint, not a verdict."""
    return len(text) < 4000 and CORRECTION.search(text) is not None


def command_key(tool: str, input_json: str) -> str:
    """Name a call by its tool and, for shell calls, the script or program it runs."""
    command: str | None = None
    try:
        parsed: object = json.loads(input_json)
        if isinstance(parsed, dict):
            value: object = parsed.get("command", parsed.get("cmd"))
            command = value[-1] if isinstance(value, list) and value else value if isinstance(value, str) else None
    except json.JSONDecodeError:
        found: re.Match[str] | None = re.search(r'"(?:command|cmd)"\s*:\s*"(.*)', input_json)
        command = found.group(1) if found else None
    if not command:
        return tool
    last: str = re.split(r"&&|;|\|\|", command, maxsplit=1)[-1] if command.lstrip().startswith("cd ") else command
    words: list[str] = [w.strip("()[]{};") for w in last.split() if "=" not in w.split("/")[0]]
    words = [w for w in words if w]
    if not words:
        return tool
    program: str = Path(words[0]).name
    script: re.Match[str] | None = SCRIPT.search(last)
    return f"{tool}:{script.group(1)}" if program in LAUNCHERS and script else f"{tool}:{program}"


def join_calls(rows: list[ToolRow]) -> list[Call]:
    """Join each call row with its result rows; results whose call is outside the window are dropped."""
    results: defaultdict[tuple[str, str, str, str], list[ToolRow]] = defaultdict(list)
    for r in rows:
        if r.phase == "result":
            results[(r.dataset, r.segment, r.agent_id, r.call_id)].append(r)
    calls: list[Call] = []
    for c in rows:
        if c.phase != "call":
            continue
        parts: list[ToolRow] = results.get((c.dataset, c.segment, c.agent_id, c.call_id), [])
        errors: list[ToolRow] = [p for p in parts if p.is_error]
        elapsed: list[float] = [p.elapsed_ms for p in parts if p.elapsed_ms is not None]
        latency: float | None = None
        if elapsed:
            latency = max(elapsed) / 1000.0
        elif parts:
            latency = (max(p.wall for p in parts) - c.wall).total_seconds()
        shown: ToolRow | None = errors[0] if errors else parts[0] if parts else None
        calls.append(
            Call(c.dataset, c.segment, c.agent_id, c.call_id, c.tool, command_key(c.tool, c.input), c.wall,
                 latency, bool(errors), c.input, shown.result if shown else "")
        )
    return sorted(calls, key=lambda x: x.start)


def error_streaks(calls: list[Call], min_length: int = 2) -> list[Streak]:
    """Find runs of `min_length` or more failed calls in a row of one key by one agent."""
    by_key: defaultdict[tuple[str, str, str, str], list[Call]] = defaultdict(list)
    for c in sorted(calls, key=lambda x: x.start):
        by_key[(c.dataset, c.segment, c.agent_id, c.key)].append(c)
    streaks: list[Streak] = []
    for (dataset, segment, agent_id, key), seq in by_key.items():
        i: int = 0
        while i < len(seq):
            j: int = i
            while j < len(seq) and seq[j].is_error:
                j += 1
            if j - i >= min_length:
                streaks.append(Streak(dataset, segment, agent_id, key, j - i, seq[i].start, j < len(seq), seq[i].result))
            i = max(j, i + 1)
    return sorted(streaks, key=lambda s: -s.length)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile of a non-empty list."""
    ordered: list[float] = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def one_line(text: str, limit: int) -> str:
    """Collapse whitespace, drop backticks, and cut to `limit` characters."""
    return re.sub(r"\s+", " ", text).replace("`", "'").strip()[:limit]


def locus(c: Call) -> str:
    """Where a call happened, in the terms a follow-up catalog query filters on."""
    agent: str = f" agent `{c.agent_id}`" if c.agent_id else ""
    call: str = f" call `{c.call_id}`" if c.call_id else ""
    return f"{c.start:{TIME}} `{c.dataset}` `{c.segment}`{agent}{call}"


def render(ev: Evidence, prompt_chars: int = 240) -> str:
    """Render the window's coverage, prompts, errors, retry loops and slow paths as Markdown."""
    lines: list[str] = [
        f"# Retro evidence, {ev.since:{TIME}} to {ev.until:{TIME}} UTC",
        "",
        "Times are UTC. Drill in with a targeted catalog query: the dataset, `filter_segments([segment])`, one entity, the needed columns.",
        "",
        "## Coverage",
        *(f"- {line}" for line in ev.coverage),
        "",
        f"## Sessions ({len(ev.sessions)})",
    ]
    by_session: dict[tuple[str, str], Session] = {(s.dataset, s.segment): s for s in ev.sessions}
    lines += [f"- `{s.dataset}` `{s.segment}` {s.host} {HOME.sub('~', s.cwd)} | {one_line(s.title, 80)}" for s in ev.sessions]

    human: list[Prompt] = sorted((p for p in ev.prompts if p.human), key=lambda p: (p.dataset, p.segment, p.wall))
    hints: int = sum(p.correction_hint for p in human)
    lines += ["", f"## Human prompts ({len(human)}; `!` marks {hints} correction hints, a broad regex: read them all)"]
    current: tuple[str, str] | None = None
    for p in human:
        if (p.dataset, p.segment) != current:
            current = (p.dataset, p.segment)
            s: Session | None = by_session.get(current)
            lines += ["", f"### `{p.dataset}` `{p.segment}` {s.host if s else ''} {HOME.sub('~', s.cwd) if s else ''}"]
        lines.append(f"- {p.wall:{TIME}} {'!' if p.correction_hint else '-'} {one_line(p.text, prompt_chars)}")

    errors: list[Call] = [c for c in ev.calls if c.is_error]
    totals: Counter[str] = Counter(c.key for c in ev.calls)
    lines += ["", f"## Tool errors ({len(errors)} of {len(ev.calls)} calls)", "", "| key | calls | errors |", "| --- | ---: | ---: |"]
    lines += [f"| `{key}` | {totals[key]} | {n} |" for key, n in Counter(c.key for c in errors).most_common(15)]
    lines += ["", "Top error texts (grouped by their first 60 characters up to `[` or `followed by`, digits as N), each with its first occurrence:"]
    first: dict[str, Call] = {}
    texts: Counter[str] = Counter()
    for c in errors:
        group: str = re.split(r" followed by|\[", one_line(re.sub(r"\d+", "N", c.result), 60))[0]
        texts[group] += 1
        first.setdefault(group, c)
    lines += [f"- {n}x {one_line(re.sub(r'\d+', 'N', first[g].result), 120)} | {locus(first[g])}" for g, n in texts.most_common(12)]

    lines += ["", "## Retry loops (same key failed 2+ times in a row)"]
    for s in error_streaks(ev.calls)[:15]:
        agent: str = f" agent `{s.agent_id}`" if s.agent_id else ""
        lines.append(f"- {s.length}x `{s.key}` {'recovered' if s.recovered else 'gave up'} | {s.start:{TIME}} `{s.dataset}` "
                     f"`{s.segment}`{agent} | {one_line(s.first_error, 120)}")

    lines += ["", "## Slow paths (keys with 3+ timed calls, by p90; 600 s is the tool timeout)", "",
              "| key | n | p50 s | p90 s | max s | slowest call |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    timed: defaultdict[str, list[Call]] = defaultdict(list)
    for c in ev.calls:
        if c.latency_s is not None:
            timed[c.key].append(c)
    ranked: list[tuple[str, list[Call]]] = sorted(
        ((k, v) for k, v in timed.items() if len(v) >= 3), key=lambda kv: -percentile([c.latency_s for c in kv[1]], 0.9)
    )
    for key, group in ranked[:15]:
        values: list[float] = [c.latency_s for c in group]
        slowest: Call = max(group, key=lambda c: c.latency_s)
        lines.append(f"| `{key}` | {len(values)} | {percentile(values, 0.5):.1f} | {percentile(values, 0.9):.1f} | "
                     f"{max(values):.1f} | {locus(slowest)} |")
    return "\n".join(lines) + "\n"


def write_summary(summary: str, out: Path | None) -> Path | None:
    """Write `summary.md` into `out`, the only file this script creates."""
    if out is None:
        return None
    out.mkdir(parents=True, exist_ok=True)
    path: Path = out / "summary.md"
    path.write_text(summary)
    return path


def fetch(cfg: Config) -> Evidence:
    """Query every dataset for the window: one schema call and one filtered reader per entity family."""
    from datafusion import col, lit
    from datafusion import functions as F
    from rerun.catalog import CatalogClient

    since: datetime = parse_instant(cfg.since)
    until: datetime = parse_instant(cfg.until)
    client = CatalogClient(cfg.catalog)
    names: set[str] = set(client.dataset_names())
    sessions: list[Session] = []
    prompts: list[Prompt] = []
    rows: list[ToolRow] = []
    exec_errors: list[Call] = []
    coverage: list[str] = []
    refreshed: datetime | None = None

    def first(frame, name: str, dtype: pa.DataType):
        """First element of a column, or a typed null when the dataset lacks it (README pattern)."""
        return col(name)[0] if name in frame.schema().names else lit(None).cast(dtype)

    def text(name: str):
        return F.substring(F.coalesce(col(name)[0], lit("")), lit(1), lit(cfg.text_chars))

    for name in cfg.datasets:
        if name not in names:
            coverage.append(f"`{name}`: not on the catalog, skipped")
            continue
        dataset = client.get_dataset(name)
        table = dataset.segment_table()
        segs: list[dict[str, object]] = table.select(
            col("rerun_segment_id").alias("segment"),
            first(table, "property:session:host", pa.string()).alias("host"),
            first(table, "property:session:cwd", pa.string()).alias("cwd"),
            first(table, "property:session:title", pa.string()).alias("title"),
            col("wall:start").alias("start"),
            col("wall:end").alias("end"),
            col("rerun_last_updated_at").cast(pa.timestamp("us")).alias("updated"),
        ).to_arrow_table().to_pylist()
        latest: datetime | None = max((s["end"] for s in segs if s["end"]), default=None)
        coverage.append(f"`{name}`: {len(segs)} sessions, latest event {latest:%Y-%m-%d %H:%M} UTC" if latest else f"`{name}`: empty")
        refreshed = max([refreshed, *(s["updated"] for s in segs if s["updated"])], key=lambda t: t or datetime.min)
        keep: dict[str, Session] = {
            s["segment"]: Session(name, s["segment"], s["host"] or "", s["cwd"] or "", s["title"] or "") for s in segs
            if s["end"] and s["end"] >= since and s["start"] < until and (cfg.cwd is None or cfg.cwd in (s["cwd"] or ""))
        }
        if not keep:
            continue
        scoped = dataset.filter_segments(sorted(keep)) if cfg.cwd else dataset

        def window(entity: str):
            reader = scoped.filter_contents([entity]).reader(index="event")
            return reader, reader.filter(col("wall") >= lit(since)).filter(col("wall") < lit(until))

        _, user = window("/conversation/user")
        for p in user.filter(col("/conversation/user:agent_id")[0] == lit("")).select(
            col("rerun_segment_id").alias("segment"), "wall", col("/conversation/user:TextLog:text")[0].alias("text")
        ).to_arrow_table().to_pylist():
            body: str = p["text"] or ""
            human: bool = is_human(body)
            prompts.append(Prompt(name, p["segment"], p["wall"], body, human, human and is_correction(body)))

        reader, tools = window("/tools")
        for r in tools.select(
            col("rerun_segment_id").alias("segment"), "wall",
            col("/tools:agent_id")[0].alias("agent_id"), col("/tools:call_id")[0].alias("call_id"),
            col("/tools:tool")[0].alias("tool"), col("/tools:phase")[0].alias("phase"),
            col("/tools:is_error")[0].alias("is_error"), first(reader, "/tools:elapsed_ms", pa.float64()).alias("elapsed_ms"),
            text("/tools:input_json").alias("input"), text("/tools:tool_use_result_json").alias("result"),
        ).filter(col("phase").is_not_null()).to_arrow_table().to_pylist():
            rows.append(ToolRow(name, r["segment"], r["agent_id"] or "", r["call_id"] or "", r["tool"] or "", r["phase"],
                                r["wall"], bool(r["is_error"]), r["elapsed_ms"], r["input"], r["result"]))

        reader, execs = window("/executions/command")
        if "/executions/command:is_error" in reader.schema().names:
            for e in execs.filter(col("/executions/command:is_error")[0]).select(
                col("rerun_segment_id").alias("segment"), "wall", col("/executions/command:agent_id")[0].alias("agent_id"),
                text("/executions/command:input_json").alias("input"), text("/executions/command:TextLog:text").alias("result"),
            ).to_arrow_table().to_pylist():
                tool: str = "executions/command"
                exec_errors.append(Call(name, e["segment"], e["agent_id"] or "", "", tool, command_key(tool, e["input"]),
                                        e["wall"], None, True, e["input"], e["result"]))
        sessions += list(keep.values())

    if refreshed is None or refreshed < until:
        stamp: str = f"{refreshed:%Y-%m-%d %H:%M}" if refreshed else "unknown"
        coverage.append(f"LAGGING: last registration {stamp} UTC is before the window end; later sessions are missing")
    calls: list[Call] = sorted(join_calls(rows) + exec_errors, key=lambda c: c.start)
    return Evidence(since, until, coverage, sessions, prompts, calls)


def main(cfg: Config) -> None:
    """Query the window, print the summary, and write `summary.md` when `--out` is given."""
    summary: str = render(fetch(cfg), cfg.prompt_chars)
    print(summary)
    write_summary(summary, cfg.out)


if __name__ == "__main__":
    main(tyro.cli(Config))
