#!/usr/bin/env python
"""Export one catalog segment: segment.rrd (all layers, one store) + blueprint.rbl on the same application id.

usage: fetch_segment.py --catalog-url URL --dataset NAME --segment ID --out-dir DIR [--app-id ID] [--blueprint FILE]

The default blueprint is read from its file:// storage URL, so run this where the catalog's files are reachable,
or pass --blueprint. Needs rerun-sdk >= 0.37 with the catalog extra.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog-url", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--app-id", default=None, help="application id for both files (default: the dataset name)")
    parser.add_argument("--blueprint", type=Path, default=None, help=".rbl to use instead of the dataset's default")
    args = parser.parse_args()

    os.environ.setdefault("RERUN_INSECURE_SKIP_HOST_CHECK", "1")  # plain-http catalog servers
    from rerun.catalog import CatalogClient

    dataset = CatalogClient(args.catalog_url).get_dataset(name=args.dataset)
    rows = dataset.segment_table().to_arrow_table().to_pylist()
    row = next((r for r in rows if r["rerun_segment_id"] == args.segment), None)
    if row is None:
        sys.exit(f"segment {args.segment!r} not in {args.dataset!r}; ids start: {[r['rerun_segment_id'] for r in rows[:5]]}")

    blueprint = args.blueprint
    if blueprint is None:
        blueprint_id = dataset.default_blueprint()
        blueprints = dataset.blueprint_dataset()
        if not blueprint_id or blueprints is None:
            sys.exit("the dataset has no default blueprint; pass --blueprint <file.rbl>")
        bp_row = next(r for r in blueprints.segment_table().to_arrow_table().to_pylist() if r["rerun_segment_id"] == blueprint_id)
        local = [Path(unquote(urlparse(u).path)) for u in bp_row["rerun_storage_urls"] if u.startswith("file://")]
        local = [p for p in local if p.exists()]
        if not local:
            sys.exit(f"default blueprint {blueprint_id} is at {list(bp_row['rerun_storage_urls'])}, not reachable here; pass --blueprint")
        blueprint = local[0]

    app_id = args.app_id or args.dataset
    args.out_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.out_dir / "segment.rrd"
    dataset.segment_store(args.segment).write_rrd(segment_path, application_id=app_id, recording_id=args.segment)
    subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "route", "--application-id", app_id, str(blueprint), "-o", str(args.out_dir / "blueprint.rbl")],
        check=True, capture_output=True,
    )
    print(f"layers: {list(row['rerun_layer_names'])}")
    print(f"{segment_path} ({segment_path.stat().st_size / 1e6:.0f} MB) and {args.out_dir / 'blueprint.rbl'}, application id {app_id!r}")


if __name__ == "__main__":
    main()
