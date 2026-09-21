# Viewer MCP: schemas and recovery

## Discover and connect

Run `tools/list` against the actual project's `rerun viewer-mcp` binary. Tool names and argument shapes vary by release; do not infer them from the installed skill version. The examples below were checked against 0.37.0 and 0.38.1. Use the returned schema if it differs.

Spawn a viewer separately with `rerun --headless --port <port>`. Its control endpoint is `http://127.0.0.1:<port>` from the MCP host. The SDK sink `rerun+http://…/proxy` and the web viewer port are different endpoints. Over SSH, run MCP beside the viewer or use a tunnel to the control port.

Each MCP process has one viewer connection. A legacy disconnect can return `not connected` on a fresh process; treat that specific response as already disconnected and continue to connect. Other errors still need inspection. Disconnect, then connect before opening data; repeat after reconnecting a session, changing ports, or restarting the viewer. On newer servers a repeated connect can reuse the connection, but disconnect → connect also works with legacy servers.

| Purpose | Legacy schema (0.37.0) | New schema (0.38.1) |
|---|---|---|
| Disconnect | `disconnect {}` | `rerun_disconnect {}` |
| Connect | `connect {"endpoint":"http://127.0.0.1:<port>"}` | `rerun_connect`, same arguments |
| Open | `open_url {"url":"/absolute/recording.rrd"}` | `rerun_open_url`, same arguments |
| State | `viewer_state {}` | `rerun_get_viewer_state {}` |
| Time cursor | `set_time {"time":42,"timeline":"frame","play":false}` | `rerun_set_time_cursor {"time":{"time":42},"timeline":{"name":"frame"},"play":false}` |

Optional `store_id` is an object in the legacy schema and an opaque string in the new schema. Copy it from state when selecting among recordings. Copy timeline names and ranges from state too: legacy timelines expose `timeline`, `type`, `min`, `max`; newer timelines expose `timeline.name`, `time_type`, `time_range.start/end`. Sequence time is an index; duration and timestamp time is nanoseconds.

## Loaded: gate actions on recording state

After opening, poll state with a deadline (for example 30 seconds, adjusted for known large inputs). **Loaded** means the intended recording is listed and its selected timeline has a range. Empty recordings, an empty tree, or “no active recording” mean this gate has not passed. On deadline, inspect the open response, viewer stderr, and available viewer logs; stop and report the missing condition. Do not keep issuing time or UI calls against an empty viewer.

## Locator: observe → act → verify

1. Query the current tree and use the target's fresh `id`.
2. If needed, use `role` plus one observed text predicate (`label_contains`, `value_contains`, or `content_contains`). Narrow ambiguous matches before acting.
3. Use `pos: {"x":100,"y":200}` only when the tree cannot identify the target. Coordinates are logical points; at `pixels_per_point: 1.0` they match screenshot pixels.

After a click, check `clicked_id` and verify the intended state change. `clicked_id: null` does not prove a target was activated. Refresh the tree after layout changes; stale IDs and guessed labels lead to misses.

Common UI arguments (consult `tools/list` for the rest):

| Tool | Arguments to provide |
|---|---|
| `query_tree` | Optional `role`, one text predicate, `visible_only`, `limit` |
| `get_node` | Required `id` |
| `click` | A locator or `pos` object |
| `scroll` | Required `delta: {"x":0,"y":300}`; optional target locator |
| `press_key` | Required `key` |
| `type_text` | Required `text`; target locator or an already focused widget |
| `resize` | Required integer `width`, `height` in logical points |
| `wait_for` | A tree filter, nonzero `min_steps`, or both |
| `batch` | Required `actions`, each with `name` and optional `args` |
| `screenshot` | Optional `pixels_per_point` (default 1.0), `save_path` |

## Settle: rendering frames versus widget presence

**Settle** before every screenshot, after a load, time change, UI action, or failed capture: `wait_for {"min_steps":3}`. This advances rendering; inspect the image to decide whether slow video or overlays need more settling.

For **presence**, use a role/text filter observed in `query_tree`, optionally with `min_steps`. `wait_for` sees accessible widget labels and values, not rendered pixels. A scene title or video timeline name visible in an image may not be accessible text. Check that returned `matched` is nonempty when presence was requested; `ok: true` alone is insufficient. A filter-free settle call need not return matches. Provide `timeout_secs` when the default five seconds is unsuitable.

`batch` accepts only egui UI tools. Timeline, recording, connection, and viewer-state calls belong outside it, regardless of their version-specific names. After observing a real widget ID, an egui batch can be:

```json
{"actions":[
  {"name":"click","args":{"id":"<fresh-id-from-query_tree>"}},
  {"name":"wait_for","args":{"min_steps":3}},
  {"name":"screenshot","args":{"pixels_per_point":1.0}}
]}
```

Inspect each action result. A batch stops on the first error; an earlier action may already have taken effect.

## Capture and bounded recovery

The egui `screenshot` tool returns inline PNG content; optional `save_path` also writes on the **MCP server host**. Newer servers additionally offer `rerun_save_screenshot {"file_path":"/absolute/capture.png"}` with optional `view_id`; this writes on the **viewer host** and returns metadata, not inline image proof. Copy/read the saved image and inspect it. Use a visible view ID from state for per-view captures.

Allow a realistic RPC timeout (for example 90 seconds); complex scenes have taken over a minute. A timed-out capture may still be rendering. Settle, then retry once. If it fails again, inspect viewer stderr/logs and rendering state instead of repeating the same call. Headed windows must remain visible; headless avoids occlusion. Use `rerun_get_viewer_logs` if discovery lists it, with its discovered schema.

For TCP failures, verify the viewer process and control port. Restart only a viewer owned by this validation, then disconnect → connect and pass the loaded gate again. Do not terminate shared catalog servers or unrelated viewers.

For gRPC image-size errors, set `pixels_per_point: 1.0` or resize the viewport smaller, settle, and retry once. Preserve the chosen viewport for subsequent comparisons. Record the versions, endpoint, timeline/range, settle settings, evidence paths, and any failed condition in the validation notes.
