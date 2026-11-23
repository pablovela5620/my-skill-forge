"""Pull a window of agent-traces catalog evidence for a retro into Parquet, and print a summary.

Writes `sessions.parquet`, `prompts.parquet` (main agent), `calls.parquet` (main agent and
subagents) and `summary.txt` to `--out`. One schema call and one filtered reader per entity
family per dataset; all analysis after the pull is local. Run it in the agent-traces pixi
environment:

    cd ~/agent-traces-staging/em && pixi run --frozen -e agent-traces python -I \
        ~/.claude/skills/retro/scripts/retro_pull.py --since 2026-11-22 --until 2026-11-23 --out /tmp/retro

A bound without a UTC offset is UTC: `--since 2026-11-22T00:00-06:00` starts at local midnight.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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


@dataclass(frozen=True, slots=True)
class Config:
    """Window and output for one pull."""

    since: str
    """Window start: ISO date or datetime, UTC unless it carries an offset."""
    until: str
    """Window end (exclusive), same format as `since`."""
    out: Path
    """Directory for the Parquet files and summary.txt."""
    catalog: str = "rerun+http://dgx-spark:9876"
    """Catalog URL."""
    datasets: tuple[str, ...] = DATASETS
    """Datasets to read; missing ones are reported and skipped."""
    cwd: str | None = None
    """Keep only sessions whose working directory contains this substring."""
    text_chars: int = 800
    """Characters kept from each tool input and result."""


@dataclass(frozen=True, slots=True)
class ToolRow:
    """One `/tools` row: a model tool call or one part of its result."""

    dataset: str
    """Dataset name without the `agent-traces-` prefix."""
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
    """Dataset name without the `agent-traces-` prefix."""
    segment: str
    """Segment (session) ID."""
    agent_id: str
    """Empty string for the main agent."""
    call_id: str
    """Tool call ID."""
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
    """Dataset name without the `agent-traces-` prefix."""
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


def pull(cfg: Config) -> None:
    """Read the window from every dataset, write Parquet and the summary."""
    from datafusion import col, lit
    from datafusion import functions as F
    from rerun.catalog import CatalogClient

    since: datetime = parse_instant(cfg.since)
    until: datetime = parse_instant(cfg.until)
    cfg.out.mkdir(parents=True, exist_ok=True)
    client = CatalogClient(cfg.catalog)
    names: set[str] = set(client.dataset_names())
    sessions: list[dict[str, object]] = []
    prompts: list[dict[str, object]] = []
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
            coverage.append(f"{name}: not on the catalog, skipped")
            continue
        short: str = name.removeprefix("agent-traces-")
        dataset = client.get_dataset(name)
        table = dataset.segment_table()
        segs: list[dict[str, object]] = table.select(
            col("rerun_segment_id").alias("segment"),
            first(table, "property:session:host", pa.string()).alias("host"),
            first(table, "property:session:cwd", pa.string()).alias("cwd"),
            first(table, "property:session:title", pa.string()).alias("title"),
            first(table, "property:session:n_turns", pa.int64()).alias("turns"),
            col("wall:start").alias("start"),
            col("wall:end").alias("end"),
            col("rerun_last_updated_at").cast(pa.timestamp("us")).alias("updated"),
        ).to_arrow_table().to_pylist()
        latest: datetime | None = max((s["end"] for s in segs if s["end"]), default=None)
        coverage.append(f"{name}: {len(segs)} sessions, latest event {latest:%Y-%m-%d %H:%M} UTC" if latest else f"{name}: empty")
        refreshed = max([refreshed, *(s["updated"] for s in segs if s["updated"])], key=lambda t: t or datetime.min)
        keep: dict[str, dict[str, object]] = {
            s["segment"]: s for s in segs
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
            human: bool = is_human(p["text"])
            prompts.append({"dataset": short, **p, "human": human, "correction_hint": human and is_correction(p["text"])})

        reader, tools = window("/tools")
        for r in tools.select(
            col("rerun_segment_id").alias("segment"), "wall",
            col("/tools:agent_id")[0].alias("agent_id"), col("/tools:call_id")[0].alias("call_id"),
            col("/tools:tool")[0].alias("tool"), col("/tools:phase")[0].alias("phase"),
            col("/tools:is_error")[0].alias("is_error"), first(reader, "/tools:elapsed_ms", pa.float64()).alias("elapsed_ms"),
            text("/tools:input_json").alias("input"), text("/tools:tool_use_result_json").alias("result"),
        ).filter(col("phase").is_not_null()).to_arrow_table().to_pylist():
            rows.append(ToolRow(dataset=short, **{**r, "agent_id": r["agent_id"] or "", "call_id": r["call_id"] or "",
                                                  "tool": r["tool"] or "", "is_error": bool(r["is_error"])}))

        reader, execs = window("/executions/command")
        if "/executions/command:is_error" in reader.schema().names:
            for e in execs.filter(col("/executions/command:is_error")[0]).select(
                col("rerun_segment_id").alias("segment"), "wall", col("/executions/command:agent_id")[0].alias("agent_id"),
                text("/executions/command:input_json").alias("input"), text("/executions/command:TextLog:text").alias("result"),
            ).to_arrow_table().to_pylist():
                tool: str = "executions/command"
                exec_errors.append(Call(short, e["segment"], e["agent_id"] or "", "", tool, command_key(tool, e["input"]),
                                        e["wall"], None, True, e["input"], e["result"]))
        sessions += [{"dataset": short, **{k: v for k, v in s.items() if k != "updated"}} for s in keep.values()]

    if refreshed is None or refreshed < until:
        coverage.append(f"LAGGING: last registration {f'{refreshed:%Y-%m-%d %H:%M}' if refreshed else 'unknown'} UTC is before the window end; later sessions are missing")
    calls: list[Call] = sorted(join_calls(rows) + exec_errors, key=lambda c: c.start)
    pq.write_table(pa.Table.from_pylist(sessions), cfg.out / "sessions.parquet")
    pq.write_table(pa.Table.from_pylist(prompts), cfg.out / "prompts.parquet")
    pq.write_table(pa.Table.from_pylist([asdict(c) for c in calls]), cfg.out / "calls.parquet")
    summary: str = summarize(since, until, coverage, sessions, prompts, calls)
    (cfg.out / "summary.txt").write_text(summary)
    print(summary)


def summarize(since: datetime, until: datetime, coverage: list[str], sessions: list[dict[str, object]],
              prompts: list[dict[str, object]], calls: list[Call]) -> str:
    """Render the window's coverage, corrections, errors, retry loops and slow paths as plain text."""
    where: dict[tuple[str, str], str] = {
        (s["dataset"], s["segment"]): f"{s['host']} {HOME.sub('~', s['cwd'] or '')}" for s in sessions
    }
    human: list[dict[str, object]] = [p for p in prompts if p["human"]]
    hints: list[dict[str, object]] = sorted((p for p in human if p["correction_hint"]), key=lambda p: p["wall"])
    lines: list[str] = [f"Window {since:%Y-%m-%d %H:%M} to {until:%Y-%m-%d %H:%M} UTC", "", "## Coverage", *coverage, ""]
    lines.append(f"## Sessions: {len(sessions)}  " + ", ".join(f"{h}={n}" for h, n in Counter(str(s['host']) for s in sessions).most_common()))
    lines += [f"## Prompts: {len(human)} human, {len(hints)} correction hints (read them; the hint is a regex)", ""]
    lines += [f"{p['wall']:%m-%d %H:%M} {where.get((p['dataset'], p['segment']), '')} {p['segment'][:8]} | "
              + re.sub(r"\s+", " ", str(p['text']))[:220] for p in hints[:60]]
    errors: list[Call] = [c for c in calls if c.is_error]
    lines += ["", f"## Tool calls: {len(calls)}, errors {len(errors)}  (key calls errors)"]
    totals: Counter[str] = Counter(c.key for c in calls)
    for key, n in Counter(c.key for c in errors).most_common(15):
        lines.append(f"{key:50s} {totals[key]:6d} {n:5d}")
    lines += ["", "## Error texts (top, digits as N)"]
    for message, n in Counter(re.sub(r"\d+", "N", c.result[:120]) for c in errors).most_common(12):
        lines.append(f"{n:4d} {message}")
    lines += ["", "## Retry loops (same key failed 2+ times in a row)"]
    for s in error_streaks(calls)[:15]:
        lines.append(f"{s.length:3d}x {s.key:40s} {'recovered' if s.recovered else 'gave up'} {s.start:%m-%d %H:%M} "
                     f"{where.get((s.dataset, s.segment), '')} {s.segment[:8]} | {s.first_error[:120]}")
    lines += ["", "## Slow paths (keys with 3+ timed calls, by p90 seconds)  (key n p50 p90 max)"]
    timed: defaultdict[str, list[float]] = defaultdict(list)
    for c in calls:
        if c.latency_s is not None:
            timed[c.key].append(c.latency_s)
    slow: list[tuple[str, list[float]]] = sorted(((k, v) for k, v in timed.items() if len(v) >= 3), key=lambda kv: -percentile(kv[1], 0.9))
    for key, values in slow[:15]:
        lines.append(f"{key:50s} {len(values):5d} {percentile(values, 0.5):7.1f} {percentile(values, 0.9):7.1f} {max(values):7.1f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    pull(tyro.cli(Config))
