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

## Open investigations

### Large-recording embeds via OSS catalog server (blocked upstream, 2026-07-06)

The web-embed branch gates recordings at a few hundred MB because the WASM
viewer holds the whole `.rrd` in memory. A possible way out is registering the
recording on an OSS catalog server and pointing the viewer at the segment URL,
so it streams instead of downloading (pattern from
[rerun-io/so100-hackathon](https://github.com/rerun-io/so100-hackathon)'s
`tools/apps/dataset_collector.py`):

```bash
rerun server --cors-allow-origin '<viewer-origin>'   # default allows only localhost + rerun.io
```

```python
client = rr.catalog.CatalogClient("rerun+http://127.0.0.1:51234")  # RERUN_INSECURE_SKIP_HOST_CHECK=1
dataset = client.create_dataset("recordings", exist_ok=True)
segs = dataset.register([path.as_uri()]).wait().segment_ids
url = dataset.segment_url(segs[0])
```

Status from a 751 MB test (assembly101 episode, rerun 0.34.0):

- Registration is instant (~1.3 s — lazy indexing, no upfront ingest).
- The **native** viewer connects and browses the dataset/segment fine.
- The **WASM** viewer fails connection verification with
  `missing grpc-status trailer` in *every* topology — including same-machine
  plain-http against the rerun binary's own `--serve-web` viewer — while curl
  shows byte-perfect grpc-web framing including the trailer frame. The break is
  in the viewer/browser grpc-web read path, i.e. **upstream**, not
  infrastructure. No matching rerun-io/rerun issue as of 2026-07-06.
- Even once fixed, two more hurdles: `tailscale serve` truncates gRPC streams
  in both tcp+TLS and https-proxy modes (tailnet transport would need a
  gRPC-aware proxy), and the OSS server is in-memory (1.6 GB RSS after serving
  the 751 MB recording), so it has its own ceiling.

When the upstream issue is fixed and the path is validated end-to-end, promote
the recipe into `references/web.md` and lift the size guidance accordingly.
