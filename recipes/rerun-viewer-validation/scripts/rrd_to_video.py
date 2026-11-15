"""Render an .rrd recording to video by sweeping a timeline in a headless Rerun viewer.

One `rerun --headless` viewer loads the recording; `rerun viewer-mcp` (stdio JSON-RPC, the 0.42
`rerun_*` tools) drives it: set the time cursor -> short settle -> save a screenshot, per frame,
then ffmpeg encodes the PNGs. Frame times are sampled evenly across the timeline's range, or
across --start..--end. Nothing is logged into the recording.

Outputs:
  --out sweep.mp4     one H.264 MP4 (CRF 20), for reports and PRs.
  --embed site/run    run.av1.mp4 (SVT-AV1, screen content, keyframe every 2 s), run.h264.mp4
                      (960 px fallback) and run.poster.jpg: the video set for a NAS embed.js figure.

--rbl applies a blueprint after the recording has loaded and waits until its views appear. Local
files are served over 127.0.0.1 HTTP: the native viewer registers a local .rrd with its Viewer
catalog under a random application id, and a blueprint opened after that does not apply.

Stdlib only. Requires a rerun binary (0.42+) with the `viewer-mcp` subcommand, ffmpeg (libx264;
libsvtav1 for --embed) and a GPU or software rasterizer.

Usage:
  python rrd_to_video.py --rrd rec.rrd --out sweep.mp4 [--rerun-bin PATH] [--timeline NAME]
      [--frames 150] [--fps 15] [--settle-ms 30] [--size WxH] [--collapse-panels]
  python rrd_to_video.py --rrd run.rrd --rbl run.rbl --embed site/run --timeline video_time \\
      --start 26s --end 28s --poster-at 26.8s --seconds 15
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import itertools
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` does not add the script dir
from headless_viewer import HeadlessViewer, McpStdioClient, connect_mcp, rerun_version, start_viewer  # noqa: E402

LOAD_TIMEOUT_S = 180.0
FFMPEG_TIMEOUT_S = 600.0
KINDS = {"TIME_TYPE_SEQUENCE": "sequence", "TIME_TYPE_DURATION_NS": "duration", "TIME_TYPE_TIMESTAMP_NS": "timestamp"}


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass


def serve_local(sources: list[str], folder: Path) -> tuple[list[str], ThreadingHTTPServer | None]:
    """HTTP URLs for the sources; local paths are served from 127.0.0.1 (see the module docstring)."""
    if all("://" in s for s in sources):
        return sources, None
    folder.mkdir()
    for s in sources:
        if "://" not in s:
            (folder / Path(s).name).symlink_to(Path(s).resolve())
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(folder)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return [s if "://" in s else f"{base}/{Path(s).name}" for s in sources], server


def timelines(state: Any) -> list[dict[str, Any]]:
    """Ranged timelines of the active recording as {store_id, name, kind, start, end}.

    Proto JSON omits a zero start. The boot wait and the selection use this one predicate, so a
    timeline that is visible but not yet ranged cannot pass boot and then fail selection.
    """
    recordings = state.get("recordings", []) if isinstance(state, dict) else []
    if not recordings:
        return []
    rec = next((r for r in recordings if r.get("store_id") == state.get("active_store_id")), recordings[0])
    out = []
    for tl in rec.get("timelines", []):
        span = tl.get("time_range")
        if span:
            out.append({
                "store_id": rec["store_id"],
                "name": tl.get("timeline", {}).get("name"),
                "kind": KINDS.get(tl.get("time_type", ""), "timestamp"),
                "start": int(span.get("start", 0)),
                "end": int(span.get("end", 0)),
            })
    return out


def pick_timeline(state: Any, requested: str | None) -> dict[str, Any]:
    """The requested timeline, else the first that is not log_time, else the first."""
    ranged = timelines(state)
    if not ranged:
        raise SystemExit(f"no timelines with a range in the viewer state: {json.dumps(state)[:2000]}")
    if requested is not None:
        for tl in ranged:
            if tl["name"] == requested:
                return tl
        raise SystemExit(f"timeline {requested!r} not found; have {[t['name'] for t in ranged]}")
    non_log = [t for t in ranged if t["name"] != "log_time"]
    return non_log[0] if non_log else ranged[0]


def parse_time(value: str, kind: str) -> int:
    """'805', '26.8s' (temporal timelines only) or a nanosecond integer, in the timeline's unit."""
    if value.endswith("s"):
        if kind == "sequence":
            raise SystemExit(f"{value!r}: seconds only apply to temporal timelines")
        return round(float(value[:-1]) * 1e9)
    return int(value)


def compute_frame_times(lo: int, hi: int, n_req: int, is_sequence: bool) -> list[int]:
    """Evenly sample up to n_req points across [lo, hi], inclusive of both ends.

    Sequence timelines have discrete ticks, so n is capped at the tick count. Temporal timelines
    (nanoseconds) are continuous, but rounding can collide on a tiny range, so duplicates are
    collapsed (the effective frame count is len() of the result).
    """
    if hi <= lo:
        return [lo]
    n = min(n_req, hi - lo + 1) if is_sequence else n_req
    if n <= 1:
        return [lo]
    times = [lo + round(i * (hi - lo) / (n - 1)) for i in range(n)]
    return list(dict.fromkeys(times))


def wait_loaded(mcp: McpStdioClient, viewer: HeadlessViewer, require: str | None) -> Any:
    """Poll the state until nothing is loading and the (requested) timeline has a range."""
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    seen_at: float | None = None
    while time.monotonic() < deadline:
        viewer.check_alive()
        state = mcp.tool_json("rerun_get_state")
        names = {t["name"] for t in timelines(state)}
        if names and not state.get("loading"):
            if require is None or require in names:
                return state
            seen_at = seen_at or time.monotonic()
            if time.monotonic() - seen_at > 5.0:
                raise SystemExit(f"timeline {require!r} not found; recording has {sorted(names)}")
        time.sleep(0.1)
    raise SystemExit(f"recording did not load within {LOAD_TIMEOUT_S:.0f}s; see {viewer.log_path}")


def apply_blueprint(mcp: McpStdioClient, rbl_url: str) -> None:
    """Open the .rbl after the recording has loaded and wait until its views replace the automatic ones."""
    def view_ids() -> set[str]:
        return {v["view_id"] for v in mcp.tool_json("rerun_get_state").get("views", [])}

    before = view_ids()
    mcp.call_tool("rerun_open", {"url": rbl_url, "navigate": False})
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        after = view_ids()
        if after and after != before:
            return
        time.sleep(0.25)
    raise SystemExit(f"blueprint {rbl_url} did not replace the automatic layout within 30s")


def ffmpeg(*arguments: str) -> None:
    """Run ffmpeg with a deadline; fail with the tail of its output."""
    try:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-hide_banner", "-y", *arguments],
            check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S,
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ffmpeg failed (rc={e.returncode}):\n{e.stderr.decode(errors='replace')[-2000:]}") from e


def encode(frames_dir: Path, fps: int, out: Path | None, embed: Path | None, av1_crf: int, h264_width: int) -> list[Path]:
    """Encode the frames: one H.264 MP4 (--out), and/or the embed set (--embed)."""
    source = ["-framerate", str(fps), "-i", str(frames_dir / "%06d.png")]
    written: list[Path] = []
    if out is not None:
        ffmpeg(*source, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out))
        written.append(out)
    if embed is not None:
        container = ["-g", str(2 * fps), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        av1, h264, poster = (Path(f"{embed}{s}") for s in (".av1.mp4", ".h264.mp4", ".poster.jpg"))
        ffmpeg(*source, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libsvtav1", "-preset", "7",
               "-crf", str(av1_crf), "-svtav1-params", "tune=0:scm=1", *container, str(av1))
        ffmpeg(*source, "-vf", f"scale={h264_width}:-2", "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
               "-preset", "slow", "-crf", "28", "-tune", "animation", *container, str(h264))
        ffmpeg("-i", str(frames_dir / "poster.png"), "-q:v", "3", str(poster))
        written += [av1, h264, poster]
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rrd", required=True, help="Recording: a path or an http(s) URL")
    ap.add_argument("--out", default=None, help="Write one H.264 MP4 here")
    ap.add_argument("--embed", default=None, help="Output stem for <stem>.av1.mp4, <stem>.h264.mp4, <stem>.poster.jpg")
    ap.add_argument("--rbl", default=None, help="Blueprint (path or URL), applied after the recording loads")
    ap.add_argument("--rerun-bin", default="rerun", help="rerun binary with viewer-mcp (same version as the viewer that will show it)")
    ap.add_argument("--timeline", default=None, help="Timeline to sweep (default: first ranged timeline that is not log_time)")
    ap.add_argument("--start", default=None, help="First time: integer in the timeline's unit, or seconds with an 's' suffix")
    ap.add_argument("--end", default=None, help="Last time, same format")
    ap.add_argument("--poster-at", default=None, help="Poster time for --embed, same format (default: first frame)")
    ap.add_argument("--frames", type=int, default=None, help="Points sampled evenly across the range (default: seconds*fps)")
    ap.add_argument("--seconds", type=float, default=None, help="Video length; sets --frames (default 10 for --out, 15 for --embed)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--settle-ms", type=int, default=30,
                    help="Wait after each time change before the screenshot; raise to 100-400 for overlay-heavy or video scenes")
    ap.add_argument("--size", default=None, help="Viewer window size WxH in logical points (default 1280x800 for --embed)")
    ap.add_argument("--collapse-panels", action="store_true", help="Collapse the blueprint, time and selection panels")
    ap.add_argument("--allow-static", action="store_true", help="Accept a mostly unchanging sweep instead of failing")
    ap.add_argument("--av1-crf", type=int, default=38, help="SVT-AV1 CRF for --embed; higher is smaller")
    ap.add_argument("--h264-width", type=int, default=960, help="Width of the --embed H.264 fallback")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()
    if not args.out and not args.embed:
        ap.error("give --out and/or --embed")

    # SIGTERM/SIGINT must run the finally below (killing the viewer's process group).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))

    for source in (args.rrd, args.rbl):
        if source and "://" not in source and not Path(source).is_file():
            raise SystemExit(f"not found: {source}")
    size = args.size or ("1280x800" if args.embed else None)
    seconds = args.seconds or (15.0 if args.embed else 10.0)
    n_req = args.frames or max(2, round(seconds * args.fps))
    out = Path(args.out).resolve() if args.out else None
    embed = Path(args.embed).resolve() if args.embed else None
    for target in (out, embed):
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="rrd-video-"))
    frames_dir = work / "frames"
    frames_dir.mkdir()
    log_path = work / "viewer.log"
    viewer: HeadlessViewer | None = None
    mcp: McpStdioClient | None = None
    server: ThreadingHTTPServer | None = None
    try:
        print(f"rerun: {rerun_version(args.rerun_bin)}")
        urls, server = serve_local([args.rrd, *([args.rbl] if args.rbl else [])], work / "files")
        viewer = start_viewer(args.rerun_bin, [urls[0]], log_path, size)
        mcp = connect_mcp(args.rerun_bin, viewer)
        wait_loaded(mcp, viewer, None if args.rbl else args.timeline)
        if args.rbl:
            apply_blueprint(mcp, urls[1])
        state = wait_loaded(mcp, viewer, args.timeline)
        tl = pick_timeline(state, args.timeline)
        lo = parse_time(args.start, tl["kind"]) if args.start else tl["start"]
        hi = parse_time(args.end, tl["kind"]) if args.end else tl["end"]
        times = compute_frame_times(lo, hi, n_req, tl["kind"] == "sequence")
        n = len(times)
        if n < n_req:
            print(f"note: timeline range only supports {n} distinct frames (requested {n_req})")
        print(f"timeline {tl['name']!r} ({tl['kind']}) {lo}..{hi}: {n} frames")

        if args.collapse_panels:
            for label in ("Blueprint panel toggle", "Time panel toggle", "Selection panel toggle"):
                with contextlib.suppress(RuntimeError):  # already collapsed, or the label moved
                    mcp.call_tool("click", {"label_contains": label})

        time.sleep(1.0)  # let textures, tiles and the panel animation settle before the first frame
        startup_done = time.monotonic()
        cursor = {"store_id": tl["store_id"], "timeline": {"name": tl["name"]}, "play_state": "PLAY_STATE_PAUSED"}

        def capture(t: int, path: Path) -> bytes:
            mcp.call_tool("rerun_set_time_cursor", {**cursor, "time": {"time": int(t)}})
            if args.settle_ms > 0:
                time.sleep(args.settle_ms / 1000.0)
            mcp.call_tool("rerun_save_screenshot", {"file_path": str(path), "max_size": 0})
            data = path.read_bytes() if path.exists() else b""
            if not data:
                raise SystemExit(f"screenshot at {t} is empty ({path})")
            return data

        if embed is not None:
            capture(parse_time(args.poster_at, tl["kind"]) if args.poster_at else lo, frames_dir / "poster.png")
        hashes = [hashlib.md5(capture(t, frames_dir / f"{i:06d}.png")).hexdigest() for i, t in enumerate(times)]
        cap_done = time.monotonic()

        # Stale-frame tripwire: identical consecutive frames are legal for a static scene, but a
        # mostly duplicate sweep usually means --settle-ms is too low.
        dupes = sum(a == b for a, b in itertools.pairwise(hashes))
        if n > 1 and dupes / (n - 1) > 0.5 and not args.allow_static:
            raise SystemExit(
                f"{dupes}/{n - 1} consecutive frames are byte-identical — likely stale frames; "
                "retry with a larger --settle-ms, or pass --allow-static if the scene truly is static"
            )

        enc_start = time.monotonic()
        for path in encode(frames_dir, args.fps, out, embed, args.av1_crf, args.h264_width):
            print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
        enc_done = time.monotonic()
        print(
            f"TIMING startup_s={startup_done - t0:.2f} capture_s={cap_done - startup_done:.2f} "
            f"encode_s={enc_done - enc_start:.2f} total_s={enc_done - t0:.2f} "
            f"ms/frame={(cap_done - startup_done) / n * 1000.0:.1f} (frames={n})"
        )
    finally:
        if mcp is not None:
            mcp.close()
        if viewer is not None:
            viewer.stop()
        if server is not None:
            server.shutdown()
            server.server_close()
        if args.keep_frames:
            print(f"frames kept in {frames_dir}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
