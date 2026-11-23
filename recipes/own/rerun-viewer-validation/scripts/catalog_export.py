"""Export one catalog segment to files: <out>.rrd (every layer, one recording store) and <out>.rbl (a blueprint).

  python catalog_export.py --catalog rerun+http://127.0.0.1:51234 --dataset NAME --segment ID --out DIR/run
      [--blueprint FILE.rbl] [--app-id ID]

The server joins the layers, so <out>.rrd holds one recording store and no blueprint store, even when a layer file
embeds its own blueprint. <out>.rbl is the dataset's default blueprint (or --blueprint), routed to the same
application id, so a viewer applies it to the recording. The default blueprint is read from its file:// storage URL:
run this on the catalog's host, or pass --blueprint. Needs the rerun SDK with the catalog extra (the project env's
python); the CLI runs as `python -m rerun`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


def default_blueprint(dataset: object) -> Path:
    """Local path of the dataset's default blueprint, from its file:// storage URL."""
    blueprint_id: str | None = dataset.default_blueprint()
    blueprints = dataset.blueprint_dataset()
    if not blueprint_id or blueprints is None:
        sys.exit("the dataset has no default blueprint; pass --blueprint <file.rbl>")
    rows: list[dict] = blueprints.segment_table().to_arrow_table().to_pylist()
    urls: list[str] = next(r["rerun_storage_urls"] for r in rows if r["rerun_segment_id"] == blueprint_id)
    local: list[Path] = [Path(unquote(urlparse(u).path)) for u in urls if u.startswith("file://")]
    found: list[Path] = [p for p in local if p.exists()]
    if not found:
        sys.exit(f"default blueprint {blueprint_id} is stored at {list(urls)}, not readable here; pass --blueprint")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", required=True, help="rerun+http://host:port")
    parser.add_argument("--dataset", required=True, help="dataset name")
    parser.add_argument("--segment", required=True, help="segment id (rerun_segment_id)")
    parser.add_argument("--out", required=True, type=Path, help="output prefix: writes <out>.rrd and <out>.rbl")
    parser.add_argument("--blueprint", type=Path, default=None, help=".rbl to use instead of the dataset default")
    parser.add_argument("--app-id", default=None, help="application id of both files (default: the dataset name)")
    args = parser.parse_args()

    os.environ.setdefault("RERUN_INSECURE_SKIP_HOST_CHECK", "1")  # plain-http catalog servers on the tailnet
    from rerun.catalog import CatalogClient

    dataset = CatalogClient(args.catalog).get_dataset(name=args.dataset)
    rows: list[dict] = dataset.segment_table().to_arrow_table().to_pylist()
    row: dict | None = next((r for r in rows if r["rerun_segment_id"] == args.segment), None)
    if row is None:
        first: list[str] = [r["rerun_segment_id"] for r in rows[:5]]
        sys.exit(f"segment {args.segment!r} is not in {args.dataset!r}; the first ids are {first}")

    blueprint: Path = args.blueprint or default_blueprint(dataset)
    app_id: str = args.app_id or args.dataset
    rrd: Path = Path(f"{args.out}.rrd")
    rbl: Path = Path(f"{args.out}.rbl")
    rrd.parent.mkdir(parents=True, exist_ok=True)
    dataset.segment_store(args.segment).write_rrd(rrd, application_id=app_id, recording_id=args.segment)
    subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "route", "--application-id", app_id, str(blueprint), "-o", str(rbl)],
        check=True,
        capture_output=True,
    )
    print(f"layers: {list(row['rerun_layer_names'])}")
    print(f"{rrd} ({rrd.stat().st_size / 1e6:.1f} MB) and {rbl} (from {blueprint}), application id {app_id!r}")


if __name__ == "__main__":
    main()
