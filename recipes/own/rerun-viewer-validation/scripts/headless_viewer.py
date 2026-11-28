"""Start and stop a headless Rerun viewer, and talk to it through `rerun mcp` (0.44+) or `rerun viewer-mcp` (0.42).

One helper for every branch that needs a native viewer (rrd_to_video.py imports it):
- picks a free local port (never 9876, Rerun's default), holds an exclusive lock file for it while the viewer
  runs (/tmp/rerun-viewer-ports/<port>.lock), and retries if another process steals it. The lock comes from the
  monorepo guide's kit/rrview.py: parallel captures can no longer drive each other's viewer,
- waits until the viewer's port answers, and fails with the viewer's own log if it dies,
- runs the MCP server (`mcp`, or `viewer-mcp` on 0.42) from the same rerun binary and connects it with `endpoint=`,
- stops the viewer by process group (the PyPI `rerun` launcher forks the real viewer).

After the viewer stops, an SDK that was connected to it logs
`ERROR re_sdk::recording_stream: Failed to flush previous sink: gRPC connection severed`.
That line is expected at teardown and does not mean a failure.

CLI (stdlib only):
  python headless_viewer.py start [--rerun-bin rerun] [--size 1280x800] [--log viewer.log] FILE_OR_URL...
      prints PID=, PORT=, ENDPOINT=, LOG=, VERSION= and leaves the viewer running
  python headless_viewer.py stop PID
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RPC_TIMEOUT_S = 90.0  # per-call bound; a wedged MCP server must fail loudly, not hang forever
BIND_TIMEOUT_S = 60.0
SPAWN_ATTEMPTS = 3  # retries against the free-port race (port stolen between probe and viewer bind)
RERUN_DEFAULT_PORT = 9876
PORT_LOCKS = Path(tempfile.gettempdir()) / "rerun-viewer-ports"
# The canonical viewer (examples-monorepo packages/canonical-viewer: stock rerun plus every viewer extension, so splat and
# depth-video recordings render) when it is on PATH, else stock rerun.
DEFAULT_RERUN_BIN = "rerun-canonical" if shutil.which("rerun-canonical") else "rerun"


class McpStdioClient:
    """Minimal JSON-RPC client for an MCP server speaking newline-delimited stdio."""

    def __init__(self, command: list[str]) -> None:
        stderr_f = tempfile.NamedTemporaryFile(prefix="viewer-mcp-stderr-", suffix=".log", delete=False)
        self.stderr_path = stderr_f.name
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_f)
        stderr_f.close()  # the child owns a dup'd fd; we only keep the name for diagnostics
        self._buf = bytearray()
        self._id = 0
        self._keep_stderr = False  # set when an error message points the user at the log
        try:
            self._rpc(
                "initialize",
                {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "headless-viewer", "version": "1.0"}},
            )
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            # 0.44 renamed the viewer tools from rerun_* to viewer_*; callers keep the rerun_* names.
            tools: list[str] = [tool["name"] for tool in self._rpc("tools/list", {}).get("tools", [])]
            self._viewer_prefix: str = "viewer_" if "viewer_get_state" in tools else "rerun_"
        except BaseException:
            self.close()
            raise

    def _send(self, msg: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def _readline(self, deadline: float) -> bytes:
        """One complete line, read in binary chunks into our own buffer (select() cannot see a TextIOWrapper's)."""
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
            line = self._readline(deadline)
            if not line.lstrip().startswith(b"{"):
                continue  # not a JSON-RPC message
            msg = json.loads(line)
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method} failed: {msg['error']}")
                return msg["result"]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if name.startswith("rerun_"):
            name = self._viewer_prefix + name.removeprefix("rerun_")
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            texts = [c.get("text", "") for c in result.get("content", [])]
            raise RuntimeError(f"tool {name} errored: {' '.join(texts)}")
        return result

    def tool_json(self, name: str) -> Any:
        """A tool's JSON answer; the viewer appends its log lines after the JSON text."""
        result = self.call_tool(name)
        if isinstance(result.get("structuredContent"), dict):
            return result["structuredContent"]
        text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        return json.JSONDecoder().raw_decode(text.lstrip())[0]

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
    """A free local TCP port (bind :0 and release it), never Rerun's default 9876."""
    while True:
        with socket.create_server(("127.0.0.1", 0)) as s:
            port = s.getsockname()[1]
        if port != RERUN_DEFAULT_PORT:
            return port


def claim_port() -> tuple[int, Any]:
    """A free port this process alone owns: an exclusive non-blocking flock on PORT_LOCKS/<port>.lock, and nothing
    answering on the port. Keep the returned file open while the viewer runs; closing it releases the claim."""
    PORT_LOCKS.mkdir(exist_ok=True)
    for _ in range(50):
        port = free_port()
        lock = open(PORT_LOCKS / f"{port}.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            continue
        if port_answers(port):
            lock.close()
            continue
        return port, lock
    raise SystemExit("no free viewer port could be claimed")


def port_answers(port: int) -> bool:
    """Whether something accepts TCP connections on 127.0.0.1:port."""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the whole process group — the rerun launcher forks the real viewer as a child."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), sig)


def rerun_version(rerun_bin: str) -> str:
    """The first line of `rerun --version`."""
    out = subprocess.run([rerun_bin, "--version"], capture_output=True, text=True, timeout=60)
    return (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "unknown"


class PortStolen(RuntimeError):
    """Our viewer lost its port to another process before binding (free_port TOCTOU)."""


def viewer_death_error(returncode: int | None, port: int, log_path: Path) -> BaseException:
    """Diagnose a viewer that died: only a port steal is retryable; the rest fails with the viewer's words."""
    tail = (log_path.read_text(errors="replace")[-1500:] if log_path.exists() else "").strip()
    if "address already in use" in tail.lower():
        return PortStolen(f"viewer lost port {port} to another process (address already in use)")
    return SystemExit(f"viewer exited (rc={returncode}):\n{tail or f'(no viewer output; see {log_path})'}")


@dataclass
class HeadlessViewer:
    """A running headless viewer."""

    proc: subprocess.Popen
    port: int
    log_path: Path
    port_lock: Any = None

    @property
    def endpoint(self) -> str:
        """The viewer's gRPC control endpoint, for the MCP server / `rerun_connect`."""
        return f"http://127.0.0.1:{self.port}"

    def check_alive(self) -> None:
        """Raise with the viewer's log if it has exited."""
        if self.proc.poll() is not None:
            raise viewer_death_error(self.proc.returncode, self.port, self.log_path)

    def stop(self) -> None:
        """Stop the viewer's process group: SIGTERM, then SIGKILL after 5 s."""
        kill_group(self.proc, signal.SIGTERM)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kill_group(self.proc, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=5)
        if self.port_lock is not None:
            self.port_lock.close()
            self.port_lock = None


def start_viewer(rerun_bin: str, sources: list[str], log_path: Path, size: str | None = None) -> HeadlessViewer:
    """Spawn `rerun --headless` on a free port and wait until the port answers.

    Only a diagnosed port steal is retried on a fresh port; other failures raise at once with the
    viewer's own output.
    """
    last_err: Exception | None = None
    for _attempt in range(SPAWN_ATTEMPTS):
        port, lock = claim_port()
        cmd = [rerun_bin, "--headless", "--port", str(port), "--hide-welcome-screen", "--expect-data-soon"]
        if size is not None:
            cmd += ["--window-size", size]
        cmd += sources
        with open(log_path, "a") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
        viewer = HeadlessViewer(proc, port, log_path, lock)
        try:
            deadline = time.monotonic() + BIND_TIMEOUT_S
            while not port_answers(port):
                viewer.check_alive()
                if time.monotonic() > deadline:
                    raise SystemExit(f"viewer did not open port {port} within {BIND_TIMEOUT_S:.0f}s; see {log_path}")
                time.sleep(0.1)
            viewer.check_alive()
            return viewer
        except BaseException as e:
            viewer.stop()
            if not isinstance(e, PortStolen):
                raise
            last_err = e
    raise SystemExit(f"viewer start failed after {SPAWN_ATTEMPTS} attempts: {last_err}")


def mcp_subcommand(rerun_bin: str) -> str:
    """`mcp` on 0.44+, `viewer-mcp` on 0.42, read from the binary's help (an unknown subcommand would open a file)."""
    help_text: str = subprocess.run([rerun_bin, "--help"], capture_output=True, text=True, timeout=60).stdout
    return "mcp" if re.search(r"^\s+mcp\s", help_text, re.MULTILINE) else "viewer-mcp"


def connect_mcp(rerun_bin: str, viewer: HeadlessViewer, timeout_s: float = 30.0) -> McpStdioClient:
    """Run the MCP server from the same rerun binary and connect it to the viewer."""
    mcp = McpStdioClient([rerun_bin, mcp_subcommand(rerun_bin)])
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            viewer.check_alive()
            mcp.call_tool("rerun_connect", {"endpoint": viewer.endpoint})
            return mcp
        except RuntimeError as e:
            if mcp.proc.poll() is not None or time.monotonic() > deadline:
                mcp.close()
                raise SystemExit(f"could not connect the MCP server to {viewer.endpoint}: {e}") from e
            time.sleep(0.1)
        except BaseException:
            mcp.close()
            raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    start = sub.add_parser("start", help="start a headless viewer and leave it running")
    start.add_argument("sources", nargs="*", help="files or URLs to open")
    start.add_argument("--rerun-bin", default=DEFAULT_RERUN_BIN)
    start.add_argument("--size", default=None, help="window size WxH in logical points")
    start.add_argument("--log", default=None, help="viewer log path (default: a temp file)")
    stop = sub.add_parser("stop", help="stop a viewer started with `start`")
    stop.add_argument("pid", type=int)
    args = ap.parse_args()
    if args.cmd == "stop":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(args.pid, signal.SIGTERM)
        return 0
    log_path = Path(args.log or tempfile.mkstemp(prefix="rerun-headless-", suffix=".log")[1]).resolve()
    viewer = start_viewer(args.rerun_bin, args.sources, log_path, args.size)
    print(f"PID={viewer.proc.pid}\nPORT={viewer.port}\nENDPOINT={viewer.endpoint}\nLOG={log_path}")
    print(f"VERSION={rerun_version(args.rerun_bin)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
