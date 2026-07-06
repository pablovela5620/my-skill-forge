"""Render an .rrd recording to an .mp4 by sweeping a timeline in a headless Rerun viewer.

One `rerun --headless` viewer loads the recording; `rerun viewer-mcp` (stdio JSON-RPC)
drives it: set_time -> short settle -> screenshot(save_path) per frame, then one ffmpeg
pass over the PNGs. Frame times are sampled EVENLY across the timeline's full range.

Stdlib only (no rerun-sdk import). Requires a rerun binary with the `viewer-mcp` subcommand (0.34+) and ffmpeg.
120 frames at 1080p take ~10s; raise --settle-ms to 100-400 for overlay-heavy or
slow-decoding scenes (the stale-frame tripwire catches it if you forget).

Usage:
  python rrd_to_video.py --rrd rec.rrd --out out.mp4 --rerun-bin PATH \
      [--frames 150] [--fps 15] [--timeline NAME] [--settle-ms 30] \
      [--size WxH] [--pixels-per-point N] [--collapse-panels] \
      [--allow-static] [--keep-frames]
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import itertools
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

RPC_TIMEOUT_S = 90.0  # per-call bound; a wedged viewer-mcp must fail loudly, not hang forever
SPAWN_ATTEMPTS = 3  # retries against the free-port race (port stolen between probe and viewer bind)


class McpStdioClient:
    """Minimal JSON-RPC client for an MCP server speaking newline-delimited stdio."""

    def __init__(self, command: list[str]) -> None:
        stderr_f = tempfile.NamedTemporaryFile(prefix="viewer-mcp-stderr-", suffix=".log", delete=False)
        self.stderr_path = stderr_f.name
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_f,
        )
        stderr_f.close()  # the child owns a dup'd fd; we only keep the name for diagnostics
        self._buf = bytearray()
        self._id = 0
        self._keep_stderr = False  # set when an error message points the user at the log
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "rrd-to-video", "version": "0.5"},
            },
        )
        self._notify("notifications/initialized")

    def _send(self, msg: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def _readline(self, deadline: float) -> bytes:
        """One complete line from the server.

        The pipe is read in binary chunks into our own buffer, and buffered complete
        lines are consumed before ever select()ing — a stdlib TextIOWrapper.readline
        would over-read into a buffer select() can't see, and would block past the
        deadline on a partial line.
        """
        assert self.proc.stdout is not None
        scanned = 0  # resume the newline scan where the previous chunk ended
        while True:
            nl = self._buf.find(b"\n", scanned)
            if nl >= 0:
                line = bytes(self._buf[:nl])
                del self._buf[: nl + 1]
                return line
            scanned = len(self._buf)
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.proc.stdout], [], [], remaining)[0]:
                self._keep_stderr = True
                raise RuntimeError(f"MCP read timed out after {RPC_TIMEOUT_S:.0f}s (stderr: {self.stderr_path})")
            chunk = os.read(self.proc.stdout.fileno(), 1 << 16)
            if not chunk:
                self._keep_stderr = True
                raise RuntimeError(f"MCP server closed its stdout (stderr: {self.stderr_path})")
            self._buf += chunk

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        deadline = time.monotonic() + RPC_TIMEOUT_S
        while True:
            msg = json.loads(self._readline(deadline))
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method} failed: {msg['error']}")
                return msg["result"]

    def _notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            texts = [c.get("text", "") for c in result.get("content", [])]
            raise RuntimeError(f"tool {name} errored: {' '.join(texts)}")
        return result

    def tool_json(self, name: str) -> Any:
        result = self.call_tool(name)
        for c in result.get("content", []):
            if c.get("type") == "text":
                return json.loads(c["text"])
        return None

    def close(self) -> None:
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        except Exception:
            pass  # never let teardown mask the primary failure
        if not self._keep_stderr:
            with contextlib.suppress(OSError):
                os.unlink(self.stderr_path)


def free_port() -> int:
    """Grab a free TCP port via bind(:0), then release it for the viewer to reuse."""
    with socket.create_server(("127.0.0.1", 0)) as s:
        return s.getsockname()[1]


def kill_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the whole process group — the rerun launcher forks the real viewer as a child."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), sig)


def ranged_timelines(state: Any) -> list[dict[str, Any]]:
    """Timelines usable for a sweep: present in viewer_state AND carrying a min/max range.

    Both the boot readiness wait and the timeline selection go through this one
    predicate, so a timeline that is visible but not yet ranged cannot pass boot
    and then fail selection.
    """
    recordings = state.get("recordings", []) if isinstance(state, dict) else []
    return [
        t
        for rec in recordings
        for t in rec.get("timelines", [])
        if t.get("min") is not None and t.get("max") is not None
    ]


def pick_timeline(state: Any, requested: str | None) -> dict[str, Any]:
    """Choose the timeline to sweep: requested name, else first non-log_time, else first."""
    timelines = ranged_timelines(state)
    if not timelines:
        raise SystemExit(f"no timelines with a range in viewer_state: {json.dumps(state)[:2000]}")
    if requested is not None:
        for tl in timelines:
            if tl["timeline"] == requested:
                return tl
        raise SystemExit(f"timeline {requested!r} not found; have {[t['timeline'] for t in timelines]}")
    non_log = [t for t in timelines if t["timeline"] != "log_time"]
    return non_log[0] if non_log else timelines[0]


def compute_frame_times(lo: int, hi: int, n_req: int, is_sequence: bool) -> list[int]:
    """Evenly sample up to n_req points across [lo, hi], inclusive of both ends.

    Sequence timelines have discrete integer ticks, so n is capped at the tick count.
    Temporal timelines (values are nanoseconds) are continuous — do NOT treat the
    numeric range as a frame count — but rounding can still collide on a tiny range,
    so duplicates are collapsed (the effective frame count is len() of the result).
    """
    if hi <= lo:
        return [lo]
    n = min(n_req, hi - lo + 1) if is_sequence else n_req
    if n <= 1:
        return [lo]
    times = [lo + round(i * (hi - lo) / (n - 1)) for i in range(n)]
    return list(dict.fromkeys(times))  # nondecreasing, so this drops exactly the collisions


class PortStolen(RuntimeError):
    """Our viewer lost its port to another process before binding (free_port TOCTOU)."""


def viewer_death_error(returncode: int | None, port: int, log_path: Path) -> BaseException:
    """Diagnose a viewer that died during boot, from its captured output.

    Only a genuine port steal (the one nondeterministic cause) returns the retryable
    PortStolen; everything else — corrupt rrd, bad flag, missing GPU — is deterministic
    and fails immediately with the viewer's own words.
    """
    tail = (log_path.read_text(errors="replace")[-1500:] if log_path.exists() else "").strip()
    if "address already in use" in tail.lower():
        return PortStolen(f"viewer lost port {port} to another process (address already in use)")
    detail = tail or f"(no viewer output captured; see {log_path})"
    return SystemExit(f"viewer exited (rc={returncode}) during startup:\n{detail}")


def boot_viewer(
    rerun_bin: str,
    rrd_path: str,
    size: str | None,
    log_path: Path,
    require_timeline: str | None,
) -> tuple[subprocess.Popen, McpStdioClient, Any]:
    """Spawn a headless viewer + MCP client and connect, guarded against the port race.

    free_port() releases the port before the viewer binds it, so another process can
    steal it; if that happens our viewer dies with "address in use" and a bare connect
    would silently dial the impostor. Guard: OUR viewer process must stay alive through
    connect and load. ONLY a diagnosed port steal is retried on a fresh port —
    deterministic failures raise immediately with the viewer's own error output.
    """
    last_err: Exception | None = None
    for _attempt in range(SPAWN_ATTEMPTS):
        port = free_port()
        cmd = [rerun_bin, "--headless", "--port", str(port), "--hide-welcome-screen", "--expect-data-soon"]
        if size is not None:
            cmd += ["--window-size", size]
        cmd.append(rrd_path)
        with open(log_path, "a") as log_f:
            viewer_proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
        mcp: McpStdioClient | None = None
        try:
            mcp = McpStdioClient([rerun_bin, "viewer-mcp"])
            connect_deadline = time.monotonic() + 30.0
            while True:
                if viewer_proc.poll() is not None:
                    raise viewer_death_error(viewer_proc.returncode, port, log_path)
                try:
                    mcp.call_tool("connect", {"endpoint": f"http://127.0.0.1:{port}"})
                    break
                except RuntimeError as e:
                    if mcp.proc.poll() is not None:
                        raise SystemExit(f"viewer-mcp died during connect: {e}") from e
                    if time.monotonic() > connect_deadline:
                        raise SystemExit(f"could not connect to the viewer on port {port} within 30s: {e}") from e
                    time.sleep(0.05)  # a refused local connect fails in ~1ms; retry tightly
            # Poll viewer_state until the recording's (ranged) timelines appear.
            deadline = time.monotonic() + 90.0
            names_seen_at: float | None = None
            while time.monotonic() < deadline:
                if viewer_proc.poll() is not None:
                    raise viewer_death_error(viewer_proc.returncode, port, log_path)
                state = mcp.tool_json("viewer_state")
                names = {t["timeline"] for t in ranged_timelines(state)}
                if names:
                    if require_timeline is None or require_timeline in names:
                        return viewer_proc, mcp, state
                    # Visible but not the requested one; short grace for stragglers,
                    # then fail listing what the recording actually has.
                    names_seen_at = names_seen_at or time.monotonic()
                    if time.monotonic() - names_seen_at > 5.0:
                        raise SystemExit(f"timeline {require_timeline!r} not found; recording has {sorted(names)}")
                time.sleep(0.1)
            raise SystemExit(f"recording never showed timelines in viewer_state; see {log_path}")
        except BaseException as e:
            if mcp is not None:
                mcp.close()
            kill_group(viewer_proc, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                viewer_proc.wait(timeout=5)  # reap; don't leave a zombie across retries
            if not isinstance(e, PortStolen):
                raise
            last_err = e
    raise SystemExit(f"viewer boot failed after {SPAWN_ATTEMPTS} attempts: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rrd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rerun-bin", required=True, help="Path to the project env's rerun binary (needs the viewer-mcp subcommand)")
    ap.add_argument("--frames", type=int, default=150, help="Points sampled EVENLY across the range")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--timeline", default=None, help="Timeline to sweep (default: auto)")
    ap.add_argument("--settle-ms", type=int, default=30,
                    help="Wait after set_time before screenshot; raise to 100-400 for overlay-heavy or slow-decoding scenes")
    ap.add_argument("--pixels-per-point", type=float, default=1.0)
    ap.add_argument("--size", default=None, help="Viewer window size WxH (logical points)")
    ap.add_argument("--collapse-panels", action="store_true")
    ap.add_argument("--allow-static", action="store_true",
                    help="Accept a mostly-unchanging sweep (genuinely static scene) instead of failing")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    # SIGTERM/SIGINT must run the finally below (killing the viewer's process group);
    # the default handler would end the process without any cleanup.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))

    if "://" in args.rrd:
        rrd_path = args.rrd
    else:
        rrd = Path(args.rrd).resolve()
        if not rrd.exists():
            raise SystemExit(f"rrd not found: {rrd}")
        rrd_path = str(rrd)
    t0 = time.monotonic()
    frames_dir = Path(tempfile.mkdtemp(prefix="rrd-video-frames-"))

    # Boot deliberately sits OUTSIDE the try/finally: its failure messages point at
    # viewer.log inside frames_dir, which the finally below would delete.
    viewer_proc, mcp, state = boot_viewer(
        args.rerun_bin, rrd_path, args.size, frames_dir / "viewer.log", require_timeline=args.timeline
    )
    try:
        tl = pick_timeline(state, args.timeline)
        timeline_name = tl["timeline"]
        lo, hi = int(tl["min"]), int(tl["max"])
        times = compute_frame_times(lo, hi, args.frames, tl.get("type") == "sequence")
        n = len(times)
        if n < args.frames:
            print(f"note: timeline range only supports {n} distinct frames (requested {args.frames})")
        print(f"timeline {timeline_name!r} ({tl.get('type')}) {lo}..{hi}: {n} frames")

        if args.collapse_panels:
            for label in ("Blueprint panel toggle", "Time panel toggle", "Selection panel toggle"):
                try:
                    mcp.call_tool("click", {"label_contains": label})
                except RuntimeError:
                    pass  # panel already collapsed / label drift is non-fatal for framing

        # Settle before the first frame so initial textures/tiles are resident
        # (this also covers the panel-collapse animation above).
        time.sleep(1.0)
        startup_done = time.monotonic()

        hashes: list[str] = []
        for gidx, t in enumerate(times):
            frame_path = frames_dir / f"{gidx:06d}.png"
            try:
                mcp.call_tool("set_time", {"timeline": timeline_name, "time": int(t)})
                if args.settle_ms > 0:
                    time.sleep(args.settle_ms / 1000.0)
                result = mcp.call_tool(
                    "screenshot", {"save_path": str(frame_path), "pixels_per_point": args.pixels_per_point}
                )
            except RuntimeError as e:
                raise SystemExit(f"frame {gidx}/{n}: {e}") from e
            data = frame_path.read_bytes() if frame_path.exists() else b""
            if not data:  # save_path failed or raced; fall back to the PNG embedded in the reply
                img_b64 = next((c["data"] for c in result.get("content", []) if c.get("type") == "image"), "")
                data = base64.b64decode(img_b64)
                if not data:
                    raise SystemExit(f"frame {gidx} empty after screenshot RPC")
                frame_path.write_bytes(data)
            hashes.append(hashlib.md5(data).hexdigest())
        cap_done = time.monotonic()

        # Stale-frame tripwire: byte-identical consecutive frames are legal for a truly
        # static scene, but a mostly-duplicate sweep usually means settle-ms is too low.
        dupes = sum(a == b for a, b in itertools.pairwise(hashes))
        if n > 1 and dupes / (n - 1) > 0.5 and not args.allow_static:
            raise SystemExit(
                f"{dupes}/{n - 1} consecutive frames are byte-identical — likely stale frames; "
                f"retry with a larger --settle-ms, or pass --allow-static if the scene truly is static"
            )

        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        enc_start = time.monotonic()
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-framerate", str(args.fps),
                    "-i", str(frames_dir / "%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr_tail = e.stderr.decode(errors="replace")[-2000:]
            raise SystemExit(f"ffmpeg failed (rc={e.returncode}):\n{stderr_tail}") from e
        enc_done = time.monotonic()

        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        print(
            f"TIMING startup_s={startup_done - t0:.2f} capture_s={cap_done - startup_done:.2f} "
            f"encode_s={enc_done - enc_start:.2f} total_s={enc_done - t0:.2f} "
            f"ms/frame={(cap_done - startup_done) / n * 1000.0:.1f} (frames={n})"
        )
    finally:
        mcp.close()
        kill_group(viewer_proc, signal.SIGTERM)
        try:
            viewer_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kill_group(viewer_proc, signal.SIGKILL)
        if args.keep_frames:
            print(f"frames kept in {frames_dir}")
        else:
            shutil.rmtree(frames_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
