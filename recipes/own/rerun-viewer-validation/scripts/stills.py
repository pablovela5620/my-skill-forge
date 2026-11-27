"""Batch stills: many screenshots of one recording from one headless viewer, in one call.

One `rerun --headless` viewer on a claimed port (headless_viewer.claim_port: a free port plus an
exclusive lock file) opens the recording (and the .rbl after it), then for each requested time sets
the cursor and saves a screenshot through `rerun mcp` (`viewer-mcp` on 0.42). Before the first shot it checks that
every recording in the viewer is one of ours (by recording id from `rerun rrd print`), and refuses to
shoot a viewer that holds anything else. Nothing is logged into the recording.

Adapted from the monorepo guide's kit/rrview.py and kit/shots.py (port lock, one recording per
capture, refusal on foreign recordings), which replaced one-at-a-time MCP screenshot loops.

Stdlib only; same requirements as rrd_to_video.py.

Usage:
  python stills.py --rrd run.rrd [--rbl run.rbl] --timeline video_time --at 10s --at 12.5s --at 14s \\
      --out-dir shots/ [--rerun-bin PATH] [--size 1600x900] [--view-id <uuid>] [--name walk]
Writes <out-dir>/<name>-<timeline>@<value>.png per time (value in the timeline's unit) and prints
one line per shot with its path and milliseconds, then TIMING. Look at every picture before you use it.
"""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` does not add the script dir
from headless_viewer import connect_mcp, rerun_version, start_viewer  # noqa: E402
from rrd_to_video import apply_blueprint, parse_time, pick_timeline, serve_local, wait_loaded  # noqa: E402

STORE_RE = re.compile(r'StoreId\(\s*Recording,\s*"[^"]*",\s*"([^"]+)"', re.S)


def recording_ids(rerun_bin: str, source: str, max_lines: int = 400) -> set[str]:
    """Recording ids announced by the StoreInfo blocks at the start of `rerun rrd print` (it streams, so this is fast)."""
    if "://" in source:
        return set()
    proc = subprocess.Popen([rerun_bin, "rrd", "print", source], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    assert proc.stdout is not None
    lines = []
    for i, line in enumerate(proc.stdout):
        lines.append(line)
        if i >= max_lines:
            break
    proc.kill()
    proc.wait()
    return set(STORE_RE.findall("".join(lines)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rrd", required=True, help="Recording: a path or an http(s) URL")
    ap.add_argument("--rbl", default=None, help="Blueprint (path or URL), applied after the recording loads")
    ap.add_argument("--timeline", required=True, help="Timeline of the --at times")
    ap.add_argument("--at", action="append", required=True, help="Time: integer in the timeline's unit or seconds ('12.5s'); repeat")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="shot", help="File name prefix")
    ap.add_argument("--view-id", default=None, help="Shoot one view (id from rerun_get_state 'views') instead of the window")
    ap.add_argument("--size", default="1600x900", help="Viewer window size WxH in logical points")
    ap.add_argument("--rerun-bin", default="rerun", help="rerun binary, 0.42+ (>= the recording's writer version)")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)  # rerun_save_screenshot does not create folders
    t0 = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="rrd-stills-"))
    viewer = mcp = server = None
    try:
        print(f"rerun: {rerun_version(args.rerun_bin)}")
        ours = recording_ids(args.rerun_bin, args.rrd)
        urls, server = serve_local([args.rrd, *([args.rbl] if args.rbl else [])], work / "files")
        viewer = start_viewer(args.rerun_bin, [urls[0]], work / "viewer.log", args.size)
        mcp = connect_mcp(args.rerun_bin, viewer)
        wait_loaded(mcp, viewer, None if args.rbl else args.timeline)
        if args.rbl:
            apply_blueprint(mcp, urls[1])
        state = wait_loaded(mcp, viewer, args.timeline)
        foreign = [r["store_id"] for r in state.get("recordings", []) if ours and not any(rid in r["store_id"] for rid in ours)]
        if foreign:
            raise SystemExit(f"refusing to shoot: the viewer on port {viewer.port} holds other recordings {foreign}")
        tl = pick_timeline(state, args.timeline)
        cursor = {"store_id": tl["store_id"], "timeline": {"name": tl["name"]}, "play_state": "PLAY_STATE_PAUSED"}
        extra = {"view_id": args.view_id} if args.view_id else {}
        ready = time.monotonic()
        mcp.call_tool("rerun_set_time_cursor", {**cursor, "time": {"time": parse_time(args.at[0], tl["kind"])}})
        mcp.call_tool("rerun_save_screenshot", {"file_path": str(work / "warmup.png"), "max_size": 0, **extra})
        for at in args.at:
            t = parse_time(at, tl["kind"])
            path = out_dir / f"{args.name}-{tl['name']}@{t}.png"
            a = time.monotonic()
            mcp.call_tool("rerun_set_time_cursor", {**cursor, "time": {"time": t}})
            mcp.call_tool("rerun_save_screenshot", {"file_path": str(path), "max_size": 0, **extra})
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"screenshot at {at} is missing or empty ({path})")
            print(f"{path} {1000 * (time.monotonic() - a):.0f} ms")
        done = time.monotonic()
        print(f"TIMING startup_s={ready - t0:.2f} shots_s={done - ready:.2f} shots={len(args.at)} total_s={done - t0:.2f}")
    finally:
        if mcp is not None:
            mcp.close()
        if viewer is not None:
            viewer.stop()
        if server is not None:
            server.shutdown()
            server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
