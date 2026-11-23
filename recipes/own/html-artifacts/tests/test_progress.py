"""Tests for the live progress page: the state schema rejects prose, the renderer draws the fixed shape.

Run against the installed package (rattler-build test) or the recipe tree:
    PROGRESS_DIR=<skill-dir>/assets/progress pytest tests/test_progress.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

_DEFAULT = Path(__file__).resolve().parents[1] / "assets" / "progress"
sys.path.insert(0, os.environ.get("PROGRESS_DIR", str(_DEFAULT)))
sys.dont_write_bytecode = True

import progress_page  # noqa: E402
import progress_state  # noqa: E402


def _state(**over) -> dict:
    data: dict = {
        "schema": 1,
        "title": "MAMMA real time on the GB10",
        "goal": "Run the multiview SMPL-X pipeline at 30 ticks/s on one DGX Spark.",
        "owner": "mamma-realtime agent",
        "started": "2026-11-08T09:00Z",
        "updated": "2026-11-08T21:00Z",
        "finished": False,
        "targets": [{"id": "tps", "name": "Pipeline rate", "unit": "ticks/s", "start": 4.5, "goal": 30.0, "higher_is_better": True}],
        "stages": [
            {"id": "decode", "title": "AV1 decode", "status": "done"},
            {"id": "track", "title": "Tracking", "status": "running"},
            {"id": "fit", "title": "SMPL-X fit", "status": "todo"},
        ],
        "samples": [
            {"t": "2026-11-08T10:00Z", "target": "tps", "value": 4.5},
            {"t": "2026-11-08T15:00Z", "target": "tps", "value": 14.5, "stage": "track"},
        ],
        "evidence": [{"stage": "decode", "kind": "image", "src": "media/decode.jpg", "caption": "Decode at 120 fps.", "t": "2026-11-08T11:00Z"}],
        "needs": [{"text": "Approve the 15 Hz body rate.", "since": "2026-11-08T20:00Z"}],
        "events": [{"t": "2026-11-08T15:00Z", "text": "Tracking on TensorRT: 14.5 ticks/s.", "stage": "track", "kind": "result"}],
    }
    data.update(over)
    return data


def _write(tmp_path: Path, data: dict) -> Path:
    path: Path = tmp_path / "state.json"
    path.write_text(json.dumps(data))
    return path


def test_valid_state_loads(tmp_path: Path) -> None:
    state: progress_state.State = progress_state.load(_write(tmp_path, _state()))
    assert state.stages[1].status == "running"
    assert state.targets[0].goal == 30.0


def test_paragraph_is_rejected(tmp_path: Path) -> None:
    para: str = ("We spent the afternoon profiling the tracker and found that most of the time goes into the landmark "
                 "model, so we moved it to TensorRT, which helped a lot, and then we looked at decode as well. ") * 2
    data: dict = _state(events=[{"t": "2026-11-08T15:00Z", "text": para, "kind": "info"}])
    with pytest.raises(ValueError, match="events"):
        progress_state.load(_write(tmp_path, data))


def test_multiline_text_is_rejected(tmp_path: Path) -> None:
    data: dict = _state(goal="Line one.\nLine two.")
    with pytest.raises(ValueError, match="goal"):
        progress_state.load(_write(tmp_path, data))


def test_many_sentences_are_rejected(tmp_path: Path) -> None:
    data: dict = _state(needs=[{"text": "Look. At. This. Now.", "since": "2026-11-08T20:00Z"}])
    with pytest.raises(ValueError, match="needs"):
        progress_state.load(_write(tmp_path, data))


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    data: dict = _state(notes="free text that has no place")
    with pytest.raises(ValueError):
        progress_state.load(_write(tmp_path, data))


def test_unknown_stage_reference_is_rejected(tmp_path: Path) -> None:
    data: dict = _state(events=[{"t": "2026-11-08T15:00Z", "text": "Done.", "stage": "nope", "kind": "info"}])
    with pytest.raises(ValueError, match="nope"):
        progress_state.load(_write(tmp_path, data))


def test_bad_status_is_rejected(tmp_path: Path) -> None:
    data: dict = _state(stages=[{"id": "decode", "title": "AV1 decode", "status": "almost"}])
    with pytest.raises(ValueError):
        progress_state.load(_write(tmp_path, data))


def test_running_page_has_badge_and_fixed_shape(tmp_path: Path) -> None:
    html: str = progress_page.render(progress_state.load(_write(tmp_path, _state())))
    page: str = html.split("<style>")[0] + html.split("</style>")[-1]
    assert 'class="rr-wip"' in page and "data-wip" in page
    order: list[int] = [html.index(f'id="{sid}"') for sid in ("needs", "stages", "metrics", "evidence", "timeline")]
    assert order == sorted(order), "the five blocks keep their order"
    assert html.index('class="rr-wip"') < order[0]
    assert "Approve the 15 Hz body rate." in html
    assert 'data-status="running"' in html


def test_finished_page_has_no_badge(tmp_path: Path) -> None:
    data: dict = _state(finished=True, needs=[], stages=[{"id": "decode", "title": "AV1 decode", "status": "done"}],
                        samples=[{"t": "2026-11-08T10:00Z", "target": "tps", "value": 4.5}], evidence=[],
                        events=[{"t": "2026-11-08T15:00Z", "text": "Done.", "kind": "result"}])
    html: str = progress_page.render(progress_state.load(_write(tmp_path, data)))
    page: str = html.split("<style>")[0] + html.split("</style>")[-1]  # rerun.css documents the badge in a comment
    assert "data-wip" not in page and 'class="rr-wip"' not in page
    assert "Live progress · finished" in page


def test_text_is_escaped(tmp_path: Path) -> None:
    data: dict = _state(events=[{"t": "2026-11-08T15:00Z", "text": "<script>x</script> ok.", "kind": "info"}])
    html: str = progress_page.render(progress_state.load(_write(tmp_path, data)))
    assert "<script>x</script>" not in html


def test_update_commands_round_trip(tmp_path: Path) -> None:
    path: Path = _write(tmp_path, _state())
    progress_state.set_stage(path, "track", "done", t="2026-11-08T22:00Z")
    progress_state.add_sample(path, "tps", 26.5, t="2026-11-08T22:00Z", stage="fit")
    progress_state.add_event(path, "Fit at 15 Hz: 26.5 ticks/s.", kind="result", stage="fit", t="2026-11-08T22:00Z")
    state: progress_state.State = progress_state.load(path)
    assert state.stages[1].status == "done"
    assert state.samples[-1].value == 26.5
    assert state.events[-1].text.startswith("Fit at 15 Hz")
    assert state.updated == "2026-11-08T22:00Z"
    with pytest.raises(ValueError):
        progress_state.add_event(path, "x" * 400)
    assert progress_state.load(path).events[-1].text.startswith("Fit at 15 Hz"), "a rejected update leaves the file as it was"


def test_replay_uses_step_numbers_and_shows_a_banner(tmp_path: Path) -> None:
    data: dict = _state(replay="Replay of the 2026-11-08 run: 12 recorded steps, no clock times.",
                        samples=[{"t": "#0", "target": "tps", "value": 4.5}, {"t": "#11", "target": "tps", "value": 23.4}],
                        evidence=[{"stage": "decode", "kind": "image", "src": "media/decode.jpg", "caption": "Decode.", "t": "#1"}],
                        needs=[], events=[{"t": "#1", "text": "AV1 on NVDEC: 5.0 ticks/s.", "stage": "decode", "kind": "result"}])
    path: Path = _write(tmp_path, data)
    html: str = progress_page.render(progress_state.load(path))
    assert "Replay of the 2026-11-08 run" in html and "step 11" in html
    state: progress_state.State = progress_state.add_event(path, "Final quality mode: 13.2 ticks/s.", kind="result", t="#10")
    assert state.events[-1].t == "#10"
    assert progress_state.TIME_RE.match(state.updated) and not state.updated.startswith("#"), "updated stays the real publish time"


def test_steps_and_clock_times_do_not_mix(tmp_path: Path) -> None:
    data: dict = _state(samples=[{"t": "#0", "target": "tps", "value": 4.5}])
    with pytest.raises(ValueError, match="mix"):
        progress_state.load(_write(tmp_path, data))


def test_glyph_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "progress.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M1 1h2"/></svg>')
    monkeypatch.setattr(progress_page, "TABLER", tmp_path)
    assert 'd="M1 1h2"' in progress_page.glyph("progress")
    assert progress_page.glyph("missing-glyph") == ""


def test_values_keep_the_targets_decimals(tmp_path: Path) -> None:
    data: dict = _state(samples=[{"t": "2026-11-08T10:00Z", "target": "tps", "value": 5.0}, {"t": "2026-11-08T11:00Z", "target": "tps", "value": 6.5}])
    html: str = progress_page.render(progress_state.load(_write(tmp_path, data)))
    assert ">5.0<" in html and ">30.0<" in html, "5.0 is not shortened to 5 when the target's numbers carry a decimal"
