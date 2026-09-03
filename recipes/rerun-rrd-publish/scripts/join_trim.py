#!/usr/bin/env python
"""Join a recording .rrd with a blueprint .rbl into one .rrd, optionally clipped.

usage: join_trim.py --recording R.rrd --blueprint B.rbl --out OUT.rrd [--clip-seconds S] [--timeline video_time]
                    [--loop | --no-loop] [--keep-work]

A clip ends on the first video keyframe at or after the requested length, so its last GoP is complete. Static
chunks that `rrd split` drops are written back. Clips loop unless --no-loop. Needs rerun-sdk >= 0.37; the CLI is
called as `python -m rerun`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def rerun(*args: str) -> None:
    result = subprocess.run([sys.executable, "-m", "rerun", *args], capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"rerun {args[0]} {args[1]} failed:\n{result.stderr[-1500:]}")


def stores(path: Path) -> tuple[list, list]:
    import rerun.experimental as ex

    reader = ex.RrdReader(str(path))
    return list(reader.recordings()), list(reader.blueprints())


def keyframes(path: Path, timeline: str) -> tuple[int, list[int]]:
    """Timeline start and sorted keyframe times (ns) over every video, from the sparse is_keyframe rows."""
    import pyarrow as pa
    import rerun.experimental as ex

    start: int | None = None
    times: set[int] = set()
    for chunk in ex.RrdReader(str(path)).stream().filter(components="VideoStream:is_keyframe").to_chunks():
        batch = chunk.to_record_batch()
        if timeline not in batch.schema.names:
            sys.exit(f"{chunk.entity_path} has no timeline {timeline!r}; it has {chunk.timeline_names}")
        column = batch.column(timeline)
        if not (pa.types.is_duration(column.type) or pa.types.is_timestamp(column.type)):
            sys.exit(f"timeline {timeline!r} is {column.type}; clips need a duration or timestamp timeline")
        stamps = column.cast(pa.int64()).to_pylist()
        flags = batch.column(next(n for n in batch.schema.names if n.endswith("is_keyframe"))).to_pylist()
        start = min(stamps) if start is None else min(start, *stamps)
        times.update(t for f, t in zip(flags, stamps, strict=True) if (any(f) if isinstance(f, list) else bool(f)))
    if start is None or not times:
        sys.exit(f"no VideoStream keyframes on {timeline!r}; omit --clip-seconds")
    return start, sorted(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--blueprint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clip-seconds", type=float, default=None)
    parser.add_argument("--timeline", default="video_time")
    loop = parser.add_mutually_exclusive_group()
    loop.add_argument("--loop", dest="loop", action="store_true", default=None)
    loop.add_argument("--no-loop", dest="loop", action="store_false")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    import rerun.blueprint as rrb
    import rerun.experimental as ex

    recordings, embedded = stores(args.recording)
    _, blueprints = stores(args.blueprint)
    if len(recordings) != 1 or embedded:
        sys.exit(f"{args.recording}: need exactly one recording store and no blueprint store; found {recordings} {embedded} (a dataforge base file? export from the catalog)")
    if len(blueprints) != 1 or blueprints[0].application_id != recordings[0].application_id:
        sys.exit(f"blueprint stores {blueprints} must be exactly one with application id {recordings[0].application_id!r} (rerun rrd route --application-id)")
    app_id, recording_id, blueprint_id = recordings[0].application_id, recordings[0].recording_id, blueprints[0].recording_id

    work = args.out.parent / f"{args.out.stem}.work"
    shutil.rmtree(work, ignore_errors=True)
    (work / "splits").mkdir(parents=True)
    parts: list[Path] = []

    if args.clip_seconds is not None:
        fixed = work / "fixed.rrd"  # raw dataforge exports mark is_keyframe true for frame 0 only
        rerun("rrd", "optimize", "--fix-keyframe", str(args.recording), "-o", str(fixed))
        start, times = keyframes(fixed, args.timeline)
        wanted = start + int(round(args.clip_seconds * 1e9))
        cut = next((t for t in times if t >= wanted), None)
        if cut is None:
            sys.exit(f"--clip-seconds {args.clip_seconds} reaches past the last keyframe at {(times[-1] - start) / 1e9:.3f} s; shorten or omit it")
        clip_id = f"{recording_id}-clip-"  # split names parts <prefix><index>
        rerun("rrd", "split", "--output-dir", str(work / "splits"), "--timeline", args.timeline, "--time", str(cut), "--recording-id", clip_id, str(fixed))
        first = [p for p in (work / "splits").glob("*.rrd") if any(e.recording_id == f"{clip_id}0" for e in stores(p)[0])]
        if len(first) != 1:
            sys.exit(f"expected one first part in {work / 'splits'}, found {first}")
        static = work / "static.rrd"  # split drops static chunks (pinholes, transforms); write them back
        ex.RrdReader(str(fixed)).stream().filter(is_static=True).write_rrd(static, application_id=app_id, recording_id=f"{clip_id}0")
        parts += [first[0], static]
        print(f"clip: {(cut - start) / 1e9:.3f} s (asked {args.clip_seconds} s), ends on the keyframe at {cut / 1e9:.6f} s")
    else:
        parts.append(args.recording)

    if args.clip_seconds is not None if args.loop is None else args.loop:
        fragment = work / "loop-src.rbl"  # its newer RowIds override the blueprint's own /time_panel
        rrb.Blueprint(rrb.TimePanel(loop_mode="All", state=rrb.PanelState.Collapsed), auto_layout=False, auto_views=False).save(app_id, str(fragment))
        rerun("rrd", "route", "--recording-id", blueprint_id, str(fragment), "-o", str(work / "loop.rbl"))
        parts.append(work / "loop.rbl")
    parts.append(args.blueprint)

    rerun("rrd", "merge", *map(str, parts), "-o", str(work / "merged.rrd"))
    rerun("rrd", "optimize", "--fix-keyframe", str(work / "merged.rrd"), "-o", str(args.out))
    rerun("rrd", "verify", str(args.out))
    out_recordings, out_blueprints = stores(args.out)
    if len(out_recordings) != 1 or len(out_blueprints) != 1:
        sys.exit(f"{args.out} holds recordings={out_recordings} blueprints={out_blueprints}; expected one of each")
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{args.out} ({args.out.stat().st_size / 1e6:.1f} MB): recordings=1 blueprints=1, application id {app_id!r}")


if __name__ == "__main__":
    main()
