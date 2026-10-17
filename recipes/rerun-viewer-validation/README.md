# rerun-viewer-validation

Agent skill for proving what the Rerun viewer rendered — pixel evidence over logs.
The skill itself lives in [`SKILL.md`](SKILL.md) (with `scripts/`, `references/`,
`agents/`); this README holds maintainer notes that don't belong in the packaged
skill.

## Packaging

- `recipe.yaml` builds the `agent-skill-rerun-viewer-validation` noarch conda
  package. Package **version tracks the Rerun line the skill targets** (parity
  with `run_constraints`); skill-only iterations bump `build.number`;
  retargeting a new Rerun line bumps the version and resets the number.
- `run:` carries real dependencies (ffmpeg); `run_constraints:` expresses the
  Rerun compatibility floor without forcing an install.

## Catalog embeds: history

### 2026-10-17: works on 0.38.1 (promoted to `references/web.md`)

Tested end to end on a live OSS catalog (`rerun server` 0.38.1 with
`--cors-allow-origin`, behind `tailscale serve --https`), hosted
`app.rerun.io/version/0.38.1` and a self-hosted copy, iframed in an HTTPS page,
headless Chrome 152–155 on an RTX 5090, 2× GTX 1080 Ti and an M5 Pro. Every
embed rendered the game video and paused at the requested `#when`.

- Bytes and memory for one 288 MB hour: catalog 8–11 MB moved, 45–130 MB WASM
  memory; full file 288 MB, 640–720 MB.
- Found on the way: the SDK's `segment_url()` writes durations with a thin
  space (`+2.525 964s`) that the viewer's `#when` parser drops silently; the
  Python `rr.server.Server` has no CORS parameter; `*://rerun.io` does not
  match `app.rerun.io`; each server restart gives the dataset a new id.
- Evidence (pablo-ubuntu): `/tmp/rerun-embed-study/` (source study, claims,
  two adversarial verdict files) and `/tmp/rerun-embed-validate/`
  (screenshots, metrics, scripts).

### 2026-07-06: blocked on 0.34.0 (historical)

A 751 MB recording registered on an OSS catalog loaded in the native viewer,
but the WASM viewer failed connection verification with
`missing grpc-status trailer` in every topology tried (plain HTTP same-machine,
`--serve-web`, the so100-hackathon JS-API pattern). The bytes reached the page
intact, so the fault was in the viewer's gRPC-web parsing; the 0.34.1 and 0.35.0
changelogs carry trailer-related fixes. `tailscale serve` also truncated gRPC
streams then; on 0.38.1 with Tailscale 1.102 the gRPC-web path through
`tailscale serve --https` works.
