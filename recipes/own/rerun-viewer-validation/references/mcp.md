# Viewer MCP: tools, settle and recovery

Use the MCP for UI state (clicks, panels, selection, reading widgets). For stills and videos use `scripts/stills.py`
and `scripts/rrd_to_video.py`: they drive the same tools in one session, without an agent turn per picture.

## Get the tools

1. `mcp__rerun__*` tools in your surface: use them.
2. Otherwise drive `rerun viewer-mcp` over stdio yourself (newline-delimited JSON-RPC: `initialize`,
   `notifications/initialized`, `tools/call`); `McpStdioClient` in `scripts/headless_viewer.py` does this.
3. To register it for later sessions: `claude mcp add rerun -- <env>/bin/rerun viewer-mcp`. Crossing to another
   agent CLI needs the user's OK.

Run `tools/list` on the binary you use; names below are 0.42.

## Connect and load

The MCP never starts a viewer. Start one (`headless_viewer.py start`), then `rerun_connect
{"endpoint": "http://127.0.0.1:<port>"}`. One MCP process holds one connection; after a viewer restart or a port
change, `rerun_disconnect`, then connect again.

The UI tools are local-only: connect over `127.0.0.1` on the viewer's own host. For a viewer on another machine, run
`rerun viewer-mcp` (or the scripts) there over SSH, or tunnel its port (`ssh -L <port>:127.0.0.1:<port> <host>`) and
connect to `127.0.0.1`. A tailnet name (`http://pablo-m5-pro:9876`) fails with "Only connections from the local
machine are allowed".

Open with `rerun_open {"url": "<http url or path>", "navigate": false}`. **Loaded** means `rerun_get_state` lists the
intended recording, `loading` is empty, and the timeline you need has a `time_range`. Poll with a deadline; an
accepted open is not a loaded recording. A local `.rrd` opened by path gets a random application id, so a `.rbl`
opened later does not apply: serve files over `http://127.0.0.1` (the scripts do) or pass both on the viewer's
command line.

| Purpose | 0.42 tool and arguments |
|---|---|
| State | `rerun_get_state {}`: `recordings[].store_id`, `timelines[].timeline.name`, `time_type`, `time_range.start/end` (a zero start is omitted), `views[].view_id`, `loading` |
| Time cursor | `rerun_set_time_cursor {"store_id": …, "timeline": {"name": "frame"}, "time": {"time": 42}, "play_state": "PLAY_STATE_PAUSED"}` (sequence: index; duration and timestamp: nanoseconds) |
| Screenshot to a file | `rerun_save_screenshot {"file_path": "/abs/x.png", "max_size": 0, "view_id": …}`: written on the viewer host; `mkdir -p` the folder first; returns metadata, not the image |
| Screenshot inline | `screenshot {"pixels_per_point": 1.0}`: returns PNG content |
| Blueprint | `rerun_get_blueprint`, `rerun_set_blueprint`; view ids change when a blueprint loads, so read them again |
| Logs | `rerun_get_logs` |

## Settle: `wait_for`

`wait_for` returns when at least `min_steps` frames have rendered and, with a filter, at least `min_matches` visible
widgets match it. It needs a filter or a nonzero `min_steps`.

- To let frames render (after a load, a time change, a click): `{"min_steps": 3}`. This is the pause; it ends as soon
  as the frames are drawn.
- To wait for a widget: a filter you saw in `widget_tree` (`content_contains`, `label_contains`, `value_contains`,
  `role`), with `timeout_secs` that fits the scene, and check that the result's `matched` is nonempty.
- `wait_for` sees widget text, not rendered pixels: a title drawn inside a view is not a widget.

## UI actions

Read `widget_tree`, act on a fresh `id` (or a `role` plus one text filter; `pos` in logical points only when the tree
cannot name the target), then verify the state changed: `clicked_id: null` proves nothing. `batch` takes only egui
tools (`click`, `wait_for`, `screenshot`, …); time, recording and connection calls go outside it. A batch stops at
the first error, after earlier actions took effect.

## Recovery

- Give a screenshot up to 90 s; a timed-out one may still be rendering. Settle with `min_steps`, retry once, then read
  `rerun_get_logs` and the viewer log instead of repeating.
- TCP errors: check the viewer process and its port; restart only a viewer you own, connect again, and pass the
  loaded check again. "The viewer did not paint a frame" on connect means the viewer has no GPU or has stalled: check
  the adapter line (SKILL.md step 2).
- gRPC image-size errors: `pixels_per_point: 1.0` or a smaller window, settle, retry once.
- Profile capture (`rerun_capture_profile_trace {"file_path": …, "num_frames": N}`): start one capture with the
  timeline playing, and keep `num_frames` near what the viewer renders in about 10 s. "Deadline expired" means the RPC
  timed out while the capture keeps running: wait for the `.puffin` file to appear, then read it. A second call fails
  with "Another profile capture is still running", and UI tools stall with "did not paint a frame" until the capture
  ends.
