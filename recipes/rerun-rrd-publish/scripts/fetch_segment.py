#!/usr/bin/env python
"""Stage 1 (fetch): one Rerun catalog segment -> ``segment.rrd`` + ``blueprint.rbl`` that share one application id.

``segment.rrd`` holds every layer of the segment joined into a single recording store, because the
catalog does the join server-side. This is the only clean route: merging the on-disk layer files
needs ``rerun rrd route``, which panics on several inputs when a dataforge base is among them, and
every dataforge base .rrd embeds its own blueprint whose activation wins after a merge.

Run with a Python that has ``rerun-sdk[catalog] >= 0.37`` (the project env's python); the rerun CLI is
invoked as ``python -m rerun`` from the same env, so no binary lookup is needed.

Usage:
  python fetch_segment.py --catalog-url rerun+http://127.0.0.1:9988 --dataset <name> --segment <segment-id> \
      --out-dir /tmp/rrd-publish/<name> [--app-id <id>] [--blueprint <file.rbl>] [--force]

The dataset's default blueprint is read from its storage URL, which must be a ``file://`` path reachable
from this machine (same host as the catalog server, or a shared mount). Otherwise pass ``--blueprint``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn
from urllib.parse import unquote, urlparse


def fail(message: str) -> NoReturn:
    print(f"fetch_segment: {message}", file=sys.stderr)
    sys.exit(1)


def rerun_cli(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "rerun", *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog-url", required=True, help="rerun+http://host:port of the catalog server")
    parser.add_argument("--dataset", required=True, help="dataset name in the catalog")
    parser.add_argument("--segment", required=True, help="segment id to export")
    parser.add_argument("--out-dir", required=True, type=Path, help="where segment.rrd and blueprint.rbl are written")
    parser.add_argument("--app-id", default=None, help="application id for both stores (default: the dataset name)")
    parser.add_argument("--blueprint", type=Path, default=None, help=".rbl to use instead of the dataset's default blueprint")
    parser.add_argument("--no-assets", action="store_true", help="skip the dataset's registered assets")
    parser.add_argument("--force", action="store_true", help="overwrite segment.rrd / blueprint.rbl if they are already in --out-dir")
    args = parser.parse_args()

    # Plain-http catalog servers need this; it is the fleet default and harmless for https.
    os.environ.setdefault("RERUN_INSECURE_SKIP_HOST_CHECK", "1")
    from rerun.catalog import CatalogClient

    client = CatalogClient(args.catalog_url)
    try:
        dataset = client.get_dataset(name=args.dataset)
    except Exception as error:  # noqa: BLE001 - the SDK raises a plain Exception for unknown names
        fail(f"dataset {args.dataset!r} not found ({error}); available: {sorted(client.dataset_names())}")
    rows = dataset.segment_table().to_arrow_table().to_pylist()
    row = next((r for r in rows if r["rerun_segment_id"] == args.segment), None)
    if row is None:
        fail(f"segment {args.segment!r} is not in {args.dataset!r}; first ids: {[r['rerun_segment_id'] for r in rows[:10]]}")
    layers: list[str] = list(row["rerun_layer_names"])

    if args.blueprint is not None:
        blueprint_src = args.blueprint
        blueprint_id = None
    else:
        blueprint_id = dataset.default_blueprint()
        if not blueprint_id:
            fail("the dataset has no default blueprint; pass --blueprint <file.rbl> (a file without one opens with an auto layout)")
        blueprint_dataset = dataset.blueprint_dataset()
        if blueprint_dataset is None:
            fail(f"the dataset names default blueprint {blueprint_id} but has no blueprint dataset to read it from; pass --blueprint <file.rbl>")
        blueprint_rows = blueprint_dataset.segment_table().to_arrow_table().to_pylist()
        blueprint_row = next((r for r in blueprint_rows if r["rerun_segment_id"] == blueprint_id), None)
        urls: list[str] = list(blueprint_row["rerun_storage_urls"]) if blueprint_row else []
        local = [Path(unquote(urlparse(u).path)) for u in urls if u.startswith("file://")]
        local = [p for p in local if p.exists()]
        if not local:
            fail(f"default blueprint {blueprint_id} lives at {urls}, not reachable from here; pass --blueprint <file.rbl>")
        blueprint_src = local[0]
    if not blueprint_src.exists():
        fail(f"blueprint file {blueprint_src} does not exist")

    app_id = args.app_id or args.dataset
    args.out_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.out_dir / "segment.rrd"
    blueprint_path = args.out_dir / "blueprint.rbl"
    # A segment export costs minutes and gigabytes; never silently replace one that is already there.
    existing = [p for p in (segment_path, blueprint_path) if p.exists()]
    if existing and not args.force:
        fail(f"{', '.join(str(p) for p in existing)} already exist(s); pass --force to overwrite, or choose another --out-dir")
    dataset.segment_store(args.segment, include_assets=not args.no_assets).write_rrd(
        segment_path, application_id=app_id, recording_id=args.segment
    )
    rerun_cli("rrd", "route", "--application-id", app_id, str(blueprint_src), "-o", str(blueprint_path))

    summary = {
        "segment": str(segment_path),
        "segment_bytes": segment_path.stat().st_size,
        "blueprint": str(blueprint_path),
        "blueprint_source": str(blueprint_src),
        "blueprint_id": blueprint_id,
        "application_id": app_id,
        "recording_id": args.segment,
        "layers": layers,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
