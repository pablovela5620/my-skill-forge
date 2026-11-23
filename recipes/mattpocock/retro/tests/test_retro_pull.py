"""Unit tests for the pure parts of retro_pull.py, without catalog access.

Run: python test_retro_pull.py [<scripts dir>]
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parents[1] / "scripts"))

from retro_pull import Call, ToolRow, command_key, error_streaks, is_correction, is_human, join_calls, parse_instant  # noqa: E402

T0: datetime = datetime(2026, 11, 22, 12, 0, 0)


def test_parse_instant_date_is_utc_midnight() -> None:
    assert parse_instant("2026-11-22") == datetime(2026, 11, 22)


def test_parse_instant_offset_converts_to_naive_utc() -> None:
    assert parse_instant("2026-11-22T00:00-06:00") == datetime(2026, 11, 22, 6)


def test_is_human_drops_injected_text() -> None:
    assert is_human("too much text, cut it down")
    assert not is_human('<paseo-system version="1" kind="agent-message">hi</paseo-system>')
    assert not is_human("<heartbeat>\n<automation_id>x</automation_id>")
    assert not is_human("[Request interrupted by user]")
    assert not is_human("")


def test_is_correction_flags_pushback() -> None:
    assert is_correction("did you even look at the page?")
    assert is_correction("Too much text")
    assert is_correction("Why don't you look instead")
    assert not is_correction("Progress report please")


def test_command_key_names_the_script_or_program() -> None:
    assert command_key("Bash", '{"command": "cd /x && pixi run python scripts/rrd_to_video.py --embed a.rrd"}') == "Bash:rrd_to_video.py"
    assert command_key("Bash", '{"command": "git status"}') == "Bash:git"
    assert command_key("Bash", '{"command": "(pixi run python shots.py)"}') == "Bash:shots.py"
    assert command_key("exec", '{"cmd": "sed -n 1,5p f.py"}') == "exec:sed"
    assert command_key("mcp/paseo/wait_for_agent", '{"agentId": "a"}') == "mcp/paseo/wait_for_agent"


def row(phase: str, call_id: str, seconds: float, is_error: bool = False, tool: str = "Bash") -> ToolRow:
    return ToolRow(
        dataset="claude", segment="s1", agent_id="", call_id=call_id, tool=tool, phase=phase,
        wall=T0 + timedelta(seconds=seconds), is_error=is_error, elapsed_ms=None,
        input='{"command": "git push"}' if phase == "call" else "", result="Error: denied" if is_error else "ok",
    )


def test_join_calls_pairs_call_and_result() -> None:
    calls: list[Call] = join_calls([row("call", "a", 0), row("result", "a", 3, is_error=True), row("call", "b", 5)])
    assert [c.call_id for c in calls] == ["a", "b"]
    assert calls[0].is_error and calls[0].latency_s == 3.0 and calls[0].result == "Error: denied"
    assert calls[0].key == "Bash:git"
    assert calls[1].latency_s is None and not calls[1].is_error


def test_error_streaks_find_repeated_failures_of_one_key() -> None:
    rows: list[ToolRow] = []
    for i, err in enumerate([True, True, True, False, True]):
        rows += [row("call", f"c{i}", i * 10), row("result", f"c{i}", i * 10 + 1, is_error=err)]
    streaks = error_streaks(join_calls(rows), min_length=2)
    assert len(streaks) == 1
    assert streaks[0].length == 3 and streaks[0].key == "Bash:git" and streaks[0].recovered


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
