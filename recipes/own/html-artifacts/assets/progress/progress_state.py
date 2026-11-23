"""The state file of a live progress page: typed, short, and rejected when it turns into prose.

An agent on a long task edits data, never HTML. The state holds the goal and its target metrics, the stages and their
status, metric samples over time, the newest evidence per stage, a "needs you" list, and one-line events. Every string
has a length limit and must be one line of at most two sentences: a paragraph fails the load. progress_page.py turns
the state into the page.

Update it with the commands of progress_page.py (they validate before they write, so a rejected update leaves the
file as it was):
    progress_page.py stage --state state.json --stage track --status done --note "14.5 ticks/s"
    progress_page.py sample --state state.json --target tps --value 26.5 --stage fit
    progress_page.py event --state state.json --text "Fit at 15 Hz: 26.5 ticks/s." --kind result --stage fit
    progress_page.py evidence --state state.json --stage fit --kind image --src media/fit.jpg --caption "Fit overlay, frame 246."
    progress_page.py need --state state.json --text "Approve the 15 Hz body rate."
    progress_page.py finish --state state.json
    progress_page.py render --state state.json --out index.html
"""

import dataclasses
import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from serde import SerdeError, serde, to_dict
from serde.json import from_json

Status: TypeAlias = Literal["todo", "running", "done", "blocked", "failed"]
EvidenceKind: TypeAlias = Literal["image", "video", "rerun", "link"]
EventKind: TypeAlias = Literal["info", "result", "problem", "decision"]

ID_RE: re.Pattern[str] = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")
TIME_RE: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$")
STEP_RE: re.Pattern[str] = re.compile(r"^#\d{1,4}$")  # a replay of recorded steps without clock times: #0, #1, ...
SENTENCE_RE: re.Pattern[str] = re.compile(r"[.!?](\s|$)")
MAX_SENTENCES: int = 2


def now() -> str:
    """The current UTC time in the state's format (2026-11-08T21:00Z)."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%MZ")


def _line(where: str, value: str, limit: int, allow_empty: bool = False) -> None:
    """One line, at most `limit` characters and two sentences: the rule that keeps a state file from becoming prose."""
    if not value and not allow_empty:
        raise ValueError(f"{where}: empty")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{where}: one line only, found a line break")
    if len(value) > limit:
        raise ValueError(f"{where}: {len(value)} characters, the limit is {limit}; write one line, put detail in evidence")
    if len(SENTENCE_RE.findall(value)) > MAX_SENTENCES:
        raise ValueError(f"{where}: more than {MAX_SENTENCES} sentences; one event per fact")


def _id(where: str, value: str) -> None:
    if not ID_RE.match(value):
        raise ValueError(f"{where}: {value!r} is not an id (lowercase letters, digits, '-', at most 24)")


def is_step(value: str) -> bool:
    return bool(STEP_RE.match(value))


def _time(where: str, value: str, steps: bool = True) -> None:
    """A UTC clock time, or (for a replay, when `steps`) a step number such as #3."""
    if steps and is_step(value):
        return
    if not TIME_RE.match(value):
        raise ValueError(f"{where}: {value!r} is not a UTC time like 2026-11-08T21:00Z")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Target:
    """A metric the task drives toward a goal, such as FPS 11 -> 60."""

    id: str
    """Short id that samples refer to (tps, fps, psnr)."""
    name: str
    """What is measured, as a label (Pipeline rate)."""
    unit: str
    """Unit written after each value (ticks/s, dB, ms)."""
    start: float
    """The value when the task began."""
    goal: float
    """The value that ends the task."""
    higher_is_better: bool = True
    """False for times and errors."""

    def __post_init__(self) -> None:
        _id("targets.id", self.id)
        _line(f"targets[{self.id}].name", self.name, 40)
        _line(f"targets[{self.id}].unit", self.unit, 12)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Stage:
    """One step of the plan; the stage diagram draws one node per stage, coloured by status."""

    id: str
    """Short id that samples, evidence and events refer to."""
    title: str
    """Node label, 1-5 words."""
    status: Status = "todo"
    """todo, running, done, blocked or failed."""
    note: str = ""
    """Optional one-line result or blocker, shown under the node (14.5 ticks/s; waits for GPU)."""

    def __post_init__(self) -> None:
        _id("stages.id", self.id)
        _line(f"stages[{self.id}].title", self.title, 32)
        _line(f"stages[{self.id}].note", self.note, 60, allow_empty=True)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Sample:
    """One measured value of a target at a time."""

    t: str
    """UTC time, 2026-11-08T21:00Z."""
    target: str
    """Target id."""
    value: float
    """The measured value, in the target's unit."""
    stage: str = ""
    """The stage that produced it, if any."""

    def __post_init__(self) -> None:
        _time("samples.t", self.t)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Evidence:
    """A picture or link that proves a stage's state: a Rerun shot or video, an image, a link."""

    stage: str
    """Stage id."""
    kind: EvidenceKind
    """image, video (an .mp4), rerun (an .rrd or viewer URL, with a poster) or link."""
    src: str
    """Path relative to the page, or a same-origin URL."""
    caption: str
    """How to read it, one line."""
    t: str
    """UTC time it was made."""
    poster: str = ""
    """Still image for a video or Rerun recording."""

    def __post_init__(self) -> None:
        _line(f"evidence[{self.stage}].src", self.src, 300)
        _line(f"evidence[{self.stage}].caption", self.caption, 140)
        _line(f"evidence[{self.stage}].poster", self.poster, 300, allow_empty=True)
        _time("evidence.t", self.t)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Need:
    """Something only the user can give: a decision, an approval, access."""

    text: str
    """The ask, one line."""
    since: str
    """UTC time it was raised."""

    def __post_init__(self) -> None:
        _line("needs.text", self.text, 120)
        _time("needs.since", self.since)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Event:
    """One line in the timeline."""

    t: str
    """UTC time."""
    text: str
    """What happened, one line."""
    stage: str = ""
    """Stage id, if any."""
    kind: EventKind = "info"
    """info, result, problem or decision."""

    def __post_init__(self) -> None:
        _time("events.t", self.t)
        _line("events.text", self.text, 120)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class State:
    """The whole progress page as data."""

    title: str
    """Page title."""
    goal: str
    """What done means, one line."""
    owner: str
    """The agent or person doing the work."""
    started: str
    """UTC start time."""
    updated: str
    """UTC time of the last update; the commands set it."""
    schema: int = 1
    """State format version."""
    finished: bool = False
    """True renders the page without the work-in-progress badge."""
    replay: str = ""
    """Set only when the page replays a recorded run (what, when, from which record); the page shows it as a banner.
    A replay may use step numbers (#0, #1, ...) in place of clock times."""
    targets: list[Target] = field(default_factory=list)
    """Metrics against their goals."""
    stages: list[Stage] = field(default_factory=list)
    """The plan, in order."""
    samples: list[Sample] = field(default_factory=list)
    """Metric values over time."""
    evidence: list[Evidence] = field(default_factory=list)
    """Pictures and links; the page shows the newest per stage."""
    needs: list[Need] = field(default_factory=list)
    """Open asks for the user; empty when nothing waits on them."""
    events: list[Event] = field(default_factory=list)
    """One-line timeline entries."""

    def __post_init__(self) -> None:
        if self.schema != 1:
            raise ValueError(f"schema {self.schema} is not supported (expected 1)")
        _line("title", self.title, 80)
        _line("goal", self.goal, 140)
        _line("owner", self.owner, 60)
        _time("started", self.started, steps=False)
        _time("updated", self.updated, steps=False)
        _line("replay", self.replay, 140, allow_empty=True)
        times: list[str] = ([s.t for s in self.samples] + [v.t for v in self.evidence] + [n.since for n in self.needs] + [v.t for v in self.events])
        if times and any(is_step(x) for x in times) and not all(is_step(x) for x in times):
            raise ValueError("times: do not mix step numbers (#3) and clock times in one state")
        if any(is_step(x) for x in times) and not self.replay:
            raise ValueError("replay: step numbers are only for a replay; say what is replayed in `replay`")
        stage_ids: set[str] = {s.id for s in self.stages}
        target_ids: set[str] = {t.id for t in self.targets}
        if len(stage_ids) != len(self.stages):
            raise ValueError("stages: duplicate id")
        for where, ref in ([("samples.stage", s.stage) for s in self.samples] + [("evidence.stage", e.stage) for e in self.evidence]
                           + [("events.stage", e.stage) for e in self.events]):
            if ref and ref not in stage_ids:
                raise ValueError(f"{where}: unknown stage {ref!r}")
        for s in self.samples:
            if s.target not in target_ids:
                raise ValueError(f"samples.target: unknown target {s.target!r}")


def load(path: Path) -> State:
    """Read and validate a state file; any problem is a ValueError naming the file and the field."""
    try:
        return from_json(State, path.read_text())
    except (SerdeError, json.JSONDecodeError, ValueError) as err:
        raise ValueError(f"{path}: {err}") from err


def save(path: Path, state: State) -> None:
    """Write atomically: a reader never sees half a file."""
    tmp: Path = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_dict(state), indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _update(path: Path, t: str | None, **changes: object) -> State:
    old: State = load(path)
    updated: str = t if t and not is_step(t) else now()  # a step number is not a time: "updated" stays the real write time
    state: State = dataclasses.replace(old, updated=updated, **changes)  # __post_init__ validates the result
    save(path, state)
    return state


def set_stage(path: Path, stage_id: str, status: Status, note: str | None = None, t: str | None = None) -> State:
    old: State = load(path)
    if stage_id not in {s.id for s in old.stages}:
        raise ValueError(f"unknown stage {stage_id!r}")
    stages: list[Stage] = [dataclasses.replace(s, status=status, note=s.note if note is None else note) if s.id == stage_id else s
                           for s in old.stages]
    return _update(path, t, stages=stages)


def add_sample(path: Path, target: str, value: float, t: str | None = None, stage: str = "") -> State:
    old: State = load(path)
    return _update(path, t, samples=[*old.samples, Sample(t=t or now(), target=target, value=float(value), stage=stage)])


def add_event(path: Path, text: str, kind: EventKind = "info", stage: str = "", t: str | None = None) -> State:
    old: State = load(path)
    return _update(path, t, events=[*old.events, Event(t=t or now(), text=text, stage=stage, kind=kind)])


def add_evidence(path: Path, stage: str, kind: EvidenceKind, src: str, caption: str, poster: str = "", t: str | None = None) -> State:
    old: State = load(path)
    item: Evidence = Evidence(stage=stage, kind=kind, src=src, caption=caption, t=t or now(), poster=poster)
    return _update(path, t, evidence=[*old.evidence, item])


def set_needs(path: Path, texts: list[str], t: str | None = None) -> State:
    """Replace the open asks (an empty list clears them); an ask that stays keeps its first time."""
    old: State = load(path)
    since: dict[str, str] = {n.text: n.since for n in old.needs}
    return _update(path, t, needs=[Need(text=x, since=since.get(x, t or now())) for x in texts])


def finish(path: Path, t: str | None = None) -> State:
    return _update(path, t, finished=True)
