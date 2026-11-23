"""Checks before a recording goes into a page: writer version, slice integrity, and the size gate.

  python rrd_check.py version FILE...
      Writer SDK version from the RRD header (bytes 0-3 'RRF2', 4-7 major minor patch meta). Stdlib only.
  python rrd_check.py slice --full run.rrd --slice run.slice.rrd
      The slice keeps every static entity of the full recording and its blueprint store, and prints both
      time ranges. Needs the rerun SDK (rerun.chunk.RrdReader), so run it in the project env.
  python rrd_check.py restore --full run.rrd --slice part.rrd --out static.rrd
      Writes the full recording's static chunks under the slice's recording id, for
      `rerun rrd merge static.rrd part.rrd -o merged.rrd` (static first) when `slice` reports missing static data.
  python rrd_check.py gate [--slice F] [--full F] [--video F...] [--poster F] [--allow-large REASON]
      Fails (exit 1) when a file is over its budget for a 10 Mbps reader: poster 150 KB, video 2 MB,
      slice 25 MB, full 300 MB. --allow-large REASON lets it pass and prints the reason. Stdlib only.

Exit code 0 when every check passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BUDGET_BYTES = {"poster": 150_000, "video": 2_000_000, "slice": 25_000_000, "full": 300_000_000}
MBPS = 10.0
META = {0b00: "", 0b01: "-rc.{}", 0b10: "-alpha.{}", 0b11: "-alpha.{}+dev"}


def writer_version(path: Path) -> str:
    """'0.38.1' (or '0.42.0-rc.2') from the first 8 bytes of an .rrd or .rbl."""
    with path.open("rb") as f:
        head = f.read(8)
    if head[:4] != b"RRF2":
        raise SystemExit(f"{path}: not an RRF2 file (starts with {head[:4]!r})")
    major, minor, patch, meta = head[4:8]
    suffix = META[meta >> 6].format(meta & 0x3F) if meta else ""
    return f"{major}.{minor}.{patch}{suffix}"


def store_summary(path: Path) -> tuple[set[str], int, dict[str, tuple[int, int]]]:
    """Static entity paths, number of blueprint stores, and {timeline: (min, max)} of the first recording."""
    import pyarrow as pa
    import pyarrow.compute as pc
    from rerun.chunk import RrdReader

    reader = RrdReader(str(path))
    static: set[str] = set()
    ranges: dict[str, tuple[int, int]] = {}
    for chunk in reader.stream(store=reader.recordings()[0]):
        if chunk.is_static:
            static.add(str(chunk.entity_path))
            continue
        batch = chunk.to_record_batch()
        for name in chunk.timeline_names:
            col = batch.column(name)
            if not pa.types.is_integer(col.type):
                col = col.cast(pa.int64())
            lo, hi = pc.min(col).as_py(), pc.max(col).as_py()
            old = ranges.get(name, (lo, hi))
            ranges[name] = (min(lo, old[0]), max(hi, old[1]))
    return static, len(reader.blueprints()), ranges


def check_slice(full: Path, sliced: Path) -> bool:
    full_static, full_bp, full_ranges = store_summary(full)
    slice_static, slice_bp, slice_ranges = store_summary(sliced)
    ok = True
    missing = sorted(full_static - slice_static)
    if missing:
        ok = False
        print(f"FAIL static entities missing from the slice: {missing} (restore them: SKILL.md, Slice)")
    if full_bp and not slice_bp:
        ok = False
        print("FAIL the full recording has a blueprint store and the slice has none")
    for name, (lo, hi) in sorted(slice_ranges.items()):
        print(f"timeline {name}: slice {lo}..{hi}, full {full_ranges.get(name)}")
    print(f"size: full {full.stat().st_size / 1e6:.1f} MB, slice {sliced.stat().st_size / 1e6:.1f} MB")
    print("PASS slice keeps static data and blueprint; now compare pixels at one time" if ok else "slice check failed")
    return ok


def restore_static(full: Path, sliced: Path, out: Path) -> None:
    from rerun.chunk import RrdReader

    source = RrdReader(str(full))
    target = RrdReader(str(sliced)).recordings()[0]
    stream = source.stream(store=source.recordings()[0]).filter(is_static=True)
    stream.write_rrd(str(out), application_id=target.application_id, recording_id=target.recording_id)
    print(f"wrote {out}; next: rerun rrd merge {out} {sliced} -o merged.rrd && rerun rrd optimize merged.rrd -o <slice>.rrd")


def check_gate(files: dict[str, list[Path]], allow_large: str | None) -> bool:
    ok = True
    for kind, paths in files.items():
        for path in paths:
            size = path.stat().st_size
            seconds = size * 8 / (MBPS * 1e6)
            over = size > BUDGET_BYTES[kind]
            mark = "OVER" if over else "ok"
            print(f"{mark:4} {kind:6} {size / 1e6:8.2f} MB  budget {BUDGET_BYTES[kind] / 1e6:g} MB  {seconds:6.1f} s at {MBPS:g} Mbps  {path}")
            ok &= not over
    if not ok and allow_large:
        print(f"allow-large: {allow_large}")
        return True
    if not ok:
        print("FAIL over budget: slice shorter, drop big static entities (rerun rrd filter --drop-entity), lower the video CRF "
              "or length, or pass --allow-large '<why the reader needs it>'")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("version")
    v.add_argument("files", nargs="+", type=Path)
    s = sub.add_parser("slice")
    s.add_argument("--full", required=True, type=Path)
    s.add_argument("--slice", required=True, type=Path)
    r = sub.add_parser("restore")
    r.add_argument("--full", required=True, type=Path)
    r.add_argument("--slice", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    g = sub.add_parser("gate")
    for kind in ("slice", "full", "poster"):
        g.add_argument(f"--{kind}", type=Path, action="append", default=[])
    g.add_argument("--video", type=Path, nargs="+", default=[])
    g.add_argument("--allow-large", metavar="REASON", default=None)
    args = ap.parse_args()
    if args.cmd == "version":
        for f in args.files:
            print(f"{writer_version(f)}  {f}")
        return 0
    if args.cmd == "restore":
        restore_static(args.full, args.slice, args.out)
        return 0
    if args.cmd == "slice":
        return 0 if check_slice(args.full, args.slice) else 1
    files = {"poster": args.poster, "video": args.video, "slice": args.slice, "full": args.full}
    return 0 if check_gate(files, args.allow_large) else 1


if __name__ == "__main__":
    sys.exit(main())
