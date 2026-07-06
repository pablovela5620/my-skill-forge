"""Render an .rrd to an .mp4 by sweeping the timeline in a headless Rerun viewer.

Spawns a headless viewer, loads the recording, then drives `rerun viewer-mcp`
over stdio JSON-RPC: `set_time` -> `screenshot(save_path)` per sampled frame,
finally ffmpeg-encodes the frames. Stdlib-only apart from rerun-sdk itself.

Usage:
  python rrd_to_video.py --rrd recording.rrd --out out.mp4 \
      [--rerun-bin PATH] [--port 9878] [--timeline frame] [--frames 150] \
      [--fps 15] [--settle-ms 100] [--keep-frames]
"""

from __future__ import annotations

import argparse
import base64
import json
import select
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

RPC_TIMEOUT_S = 120.0  # per-call bound; a wedged viewer-mcp must fail, not hang the run


class McpStdioClient:
    """Minimal JSON-RPC client for an MCP server speaking newline-delimited stdio."""

    def __init__(self, command: list[str]) -> None:
        self._stderr = tempfile.NamedTemporaryFile(prefix="viewer-mcp-stderr-", suffix=".log", delete=False)
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
        )
        self._id = 0
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "rrd-to-video", "version": "0.1"},
        })
        self._notify("notifications/initialized")

    def _send(self, msg: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _rpc(self, method: str, params: dict[str, Any], timeout: float = RPC_TIMEOUT_S) -> dict[str, Any]:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        assert self.proc.stdout is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{method} timed out after {timeout:.0f}s (MCP stderr: {self._stderr.name})")
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 1.0))
            if not ready:
                continue
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server closed its stdout (stderr: {self._stderr.name})")
            msg = json.loads(line)
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

    def tool_json(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool and parse its first text content as JSON."""
        result = self.call_tool(name, arguments)
        for c in result.get("content", []):
            if c.get("type") == "text":
                return json.loads(c["text"])
        return None

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


def pick_timeline(state: Any, requested: str | None) -> dict[str, Any]:
    """Choose the timeline to sweep: requested name, else active, else first non-log_time."""
    recordings = state.get("recordings", []) if isinstance(state, dict) else []
    timelines: list[dict[str, Any]] = []
    for rec in recordings:
        timelines.extend(rec.get("timelines", []))
    timelines = [t for t in timelines if t.get("min") is not None and t.get("max") is not None]
    if not timelines:
        raise SystemExit(f"no timelines with a range in viewer_state: {json.dumps(state)[:2000]}")
    if requested is not None:
        for tl in timelines:
            if tl["timeline"] == requested:
                return tl
        raise SystemExit(f"timeline {requested!r} not found; have {[t['timeline'] for t in timelines]}")
    non_log = [t for t in timelines if t["timeline"] != "log_time"]
    return non_log[0] if non_log else timelines[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rrd", required=True, help="Path or URL of the recording to render")
    ap.add_argument("--out", required=True, help="Output .mp4 path")
    ap.add_argument("--rerun-bin", default="rerun", help="Path to a >=0.34 rerun binary")
    ap.add_argument("--port", type=int, default=9878)
    ap.add_argument("--timeline", default=None, help="Timeline to sweep (default: auto)")
    ap.add_argument("--frames", type=int, default=150, help="Max frames to sample across the range")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--settle-ms", type=int, default=100, help="Wait after set_time before screenshot")
    ap.add_argument("--pixels-per-point", type=float, default=1.0)
    ap.add_argument("--collapse-panels", action="store_true",
                    help="Collapse the blueprint/time/selection panels before sweeping (clean, chrome-free frames)")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    from rerun.experimental import ViewerClient

    viewer = ViewerClient.spawn(
        headless=True,
        port=args.port,
        hide_welcome_screen=True,
        detach_process=False,
        executable_path=args.rerun_bin if args.rerun_bin != "rerun" else None,
        executable_name=args.rerun_bin if args.rerun_bin == "rerun" else "rerun",
    )
    frames_dir = Path(tempfile.mkdtemp(prefix="rrd-video-frames-"))
    mcp = None
    try:
        mcp = McpStdioClient([args.rerun_bin, "viewer-mcp"])
        mcp.call_tool("connect", {"endpoint": f"http://127.0.0.1:{args.port}"})
        rrd = args.rrd if "://" in args.rrd else str(Path(args.rrd).resolve())
        mcp.call_tool("open_url", {"url": rrd})

        # Wait for the recording to load and its timelines to appear.
        deadline = time.time() + 120.0
        state: Any = None
        while time.time() < deadline:
            state = mcp.tool_json("viewer_state")
            recs = state.get("recordings", []) if isinstance(state, dict) else []
            if any(r.get("timelines") for r in recs):
                break
            time.sleep(0.5)
        else:
            raise SystemExit("recording never showed timelines in viewer_state")

        if args.collapse_panels:
            # The viewer's top bar exposes one labeled toggle per panel; a fresh
            # viewer starts with panels expanded, so one click each collapses.
            for label in ("Blueprint panel toggle", "Time panel toggle", "Selection panel toggle"):
                mcp.call_tool("click", {"label_contains": label})
                time.sleep(0.2)

        tl = pick_timeline(state, args.timeline)
        lo, hi = int(tl["min"]), int(tl["max"])
        n = min(args.frames, max(2, hi - lo + 1))
        times: list[int] = [lo + round(i * (hi - lo) / (n - 1)) for i in range(n)]
        print(f"sweeping timeline {tl['timeline']!r} ({tl.get('type')}) {lo}..{hi} in {n} frames")

        time.sleep(2.0)  # let first textures/tiles settle before frame 0
        for i, t in enumerate(times):
            mcp.call_tool("set_time", {"timeline": tl["timeline"], "time": t})
            time.sleep(args.settle_ms / 1000.0)
            frame_path = frames_dir / f"{i:05d}.png"
            result = mcp.call_tool("screenshot", {
                "save_path": str(frame_path),
                "pixels_per_point": args.pixels_per_point,
            })
            if not frame_path.exists():  # fall back to the inline payload
                for c in result.get("content", []):
                    if c.get("type") == "image":
                        frame_path.write_bytes(base64.b64decode(c["data"]))
            print(f"\r  frame {i + 1}/{n} @ {t}", end="", flush=True)
        print()

        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(frames_dir / "%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
            check=True, capture_output=True,
        )
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    finally:
        if mcp is not None:
            mcp.close()
        viewer.close()
        if args.keep_frames:
            print(f"frames kept in {frames_dir}")
        else:
            shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
