#!/usr/bin/env python
"""Stage 2 (join + trim): a recording ``.rrd`` and a blueprint ``.rbl`` -> one standalone ``.rrd``.

Optionally cuts the recording to a clip. ``--clip-seconds`` is measured from the first video stamp on the
timeline, not from zero, and the cut snaps forward to the first video keyframe at or after that point, so
the clip ends on a whole GoP. Observed once (2026-09-03, app.rerun.io 0.37.0) and not re-tested since: a
truncated final GoP stayed blank in the web viewer once playback parked at the end, while mid-clip frames
rendered. Clips also get a looping, collapsed time panel unless ``--no-loop`` is given, so the viewer
never parks at all.

Run with a Python that has ``rerun-sdk >= 0.37``; the rerun CLI is invoked as ``python -m rerun``.

Usage:
  python join_trim.py --recording segment.rrd --blueprint blueprint.rbl --out final.rrd \
      [--clip-seconds 5] [--timeline video_time] [--loop | --no-loop] [--time-panel collapsed] [--keep-work]

What it guarantees on success: exactly one recording store and one blueprint store, both with the same
application id; ``rerun rrd verify`` passes; static chunks (pinholes, transforms, annotation contexts)
survive the cut, which ``rerun rrd split`` alone drops.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn


def fail(message: str) -> NoReturn:
    print(f"join_trim: {message}", file=sys.stderr)
    sys.exit(1)


def rerun_cli(*args: str, quiet: bool = True) -> None:
    result = subprocess.run([sys.executable, "-m", "rerun", *args], capture_output=quiet, text=True)
    if result.returncode != 0:
        tail = (result.stderr or "")[-2000:] if quiet else ""
        fail(f"`rerun {' '.join(args[:2])}` failed with code {result.returncode}\n{tail}")


def keyframe_times(recording: Path, timeline: str) -> tuple[dict[str, list[int]], int]:
    """Keyframe times in ns per video entity, plus the first video stamp on that timeline.

    The keyframes are the ``VideoStream:is_keyframe`` rows that are true; the start is the smallest stamp of
    any of those rows, keyframe or not, and is what ``--clip-seconds`` is measured from.

    Read this from an ``optimize --fix-keyframe`` copy, never from a raw dataforge export: that export marks
    only frame 0 as a keyframe and every later frame as ``false``, so the raw file yields one keyframe.
    """
    import pyarrow as pa
    import rerun.experimental as ex

    times: dict[str, list[int]] = {}
    stamps_seen: list[int] = []
    seen_timelines: set[str] = set()
    for chunk in ex.RrdReader(str(recording)).stream().filter(components="VideoStream:is_keyframe").to_chunks():
        batch = chunk.to_record_batch()
        names = batch.schema.names
        seen_timelines.update(chunk.timeline_names)
        if timeline not in names:
            continue
        stamp_type = batch.schema.field(timeline).type
        if pa.types.is_duration(stamp_type):
            in_ns = pa.duration("ns")
        elif pa.types.is_timestamp(stamp_type):
            in_ns = pa.timestamp("ns", tz=stamp_type.tz)
        else:
            fail(f"timeline {timeline!r} is a {stamp_type} column, not a duration or a timestamp; --clip-seconds cannot cut a sequence timeline")
        # Cast instead of to_pylist(): a duration column comes back as timedeltas, which round away nanoseconds.
        stamps = batch.column(timeline).cast(in_ns).cast(pa.int64()).to_pylist()
        flag_column = next(n for n in names if n.endswith("is_keyframe"))
        flags = batch.column(flag_column).to_pylist()
        entity = str(chunk.entity_path)
        for flag, stamp in zip(flags, stamps, strict=True):
            stamps_seen.append(stamp)
            # Values arrive as [True]/[False] lists or plain bools; dataforge logs explicit False rows.
            truthy = any(flag) if isinstance(flag, list) else bool(flag)
            if truthy:
                times.setdefault(entity, []).append(stamp)
    if not times:
        seen = f" (timelines on the video: {sorted(seen_timelines)})" if seen_timelines else ""
        fail(f"no VideoStream keyframes on timeline {timeline!r}: nothing to snap to; omit --clip-seconds{seen}")
    return {k: sorted(set(v)) for k, v in times.items()}, min(stamps_seen)


def snap_cut(start_ns: int, requested_ns: int, keyframes: dict[str, list[int]]) -> tuple[int, str]:
    """Cut at the earliest keyframe at or after ``start_ns + requested_ns``, across all videos.

    Split puts the frame at the cut time in the second part, so the first part ends on a whole GoP and split has
    no cutoff to revise. Any other time makes it log ``revising cutoff time to match video keyframe``: later in the
    GoP part 0 keeps the first frame of the next GoP, earlier the revision snaps back to the GoP start. The test is
    ``>=``, so asking for exactly a keyframe time cuts there instead of carrying a whole extra GoP.
    """
    end_ns = start_ns + requested_ns
    next_keyframes = [k for times in keyframes.values() for k in [next((t for t in times if t >= end_ns), None)] if k is not None]
    if not next_keyframes:
        last_ns = max(times[-1] for times in keyframes.values())
        fail(f"--clip-seconds {requested_ns / 1e9:g} reaches past the last keyframe at {(last_ns - start_ns) / 1e9:.3f} s; shorten it or omit it")
    cut_ns = min(next_keyframes)
    return cut_ns, f"snapped {requested_ns / 1e9:.3f}s -> {(cut_ns - start_ns) / 1e9:.3f}s (the first keyframe at or after the requested length)"


def stores(path: Path) -> tuple[list, list]:
    import rerun.experimental as ex

    reader = ex.RrdReader(str(path))
    return list(reader.recordings()), list(reader.blueprints())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--blueprint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clip-seconds", type=float, default=None, help="cut this long after the timeline's first video stamp (snapped to a GoP end)")
    parser.add_argument("--timeline", default="video_time", help="timeline the cut and keyframes are read on")
    loop = parser.add_mutually_exclusive_group()
    loop.add_argument("--loop", dest="loop", action="store_true", default=None, help="add a looping time panel (default when clipping)")
    loop.add_argument("--no-loop", dest="loop", action="store_false")
    parser.add_argument("--time-panel", choices=["collapsed", "expanded", "hidden"], default="collapsed", help="time panel state when looping")
    parser.add_argument("--keep-work", action="store_true", help="keep the intermediate files; the summary names the work directory")
    args = parser.parse_args()

    import rerun.blueprint as rrb
    import rerun.experimental as ex

    recordings, blueprints_in_recording = stores(args.recording)
    if len(recordings) != 1:
        fail(f"{args.recording} must hold exactly one recording store, found {recordings}")
    if blueprints_in_recording:
        fail(f"{args.recording} embeds a blueprint store {blueprints_in_recording}; export the segment from the catalog instead (stage 1)")
    _, blueprint_stores = stores(args.blueprint)
    if len(blueprint_stores) != 1:
        fail(f"{args.blueprint} must hold exactly one blueprint store, found {blueprint_stores}")
    recording_store, blueprint_store = recordings[0], blueprint_stores[0]
    if recording_store.application_id != blueprint_store.application_id:
        fail(
            f"application ids differ: recording {recording_store.application_id!r} vs blueprint {blueprint_store.application_id!r}; "
            f"run `python -m rerun rrd route --application-id {recording_store.application_id} {args.blueprint} -o <routed.rbl>`"
        )
    app_id = recording_store.application_id
    do_loop = args.clip_seconds is not None if args.loop is None else args.loop

    args.out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{args.out.stem}.work-", dir=args.out.parent))
    inputs: list[Path] = []
    summary: dict[str, object] = {"out": str(args.out), "application_id": app_id, "loop": do_loop}

    if args.clip_seconds is not None:
        requested_ns = int(round(args.clip_seconds * 1e9))
        # Rebuild the keyframe markers before reading them: a dataforge export marks frame 0 true and every
        # later frame false, so keyframes and the GoP structure only exist in the fixed copy (~10 s per GB).
        fixed = work / "fixed.rrd"
        rerun_cli("rrd", "optimize", "--fix-keyframe", str(args.recording), "-o", str(fixed))
        keyframes, start_ns = keyframe_times(fixed, args.timeline)
        cut_ns, note = snap_cut(start_ns, requested_ns, keyframes)
        clip_id = f"{recording_store.recording_id}-clip-"
        splits = work / "splits"
        splits.mkdir()  # split writes part 0 without creating its output directory
        rerun_cli(
            "rrd", "split", "--output-dir", str(splits), "--timeline", args.timeline,
            "--time", str(cut_ns), "--recording-id", clip_id, str(fixed),
        )
        # Split names its parts `<stem>_<data-start>__<end>.rrd`, so part 0 is only called `*_0ns__*.rrd` when the
        # timeline starts at 0. Its recording id is always `<prefix>0`, so select the part on that instead.
        part_id = f"{clip_id}0"
        written = sorted(splits.glob("*.rrd"))
        parts = [p for p in written if any(store.recording_id == part_id for store in ex.RrdReader(str(p)).recordings())]
        if len(parts) != 1:
            fail(f"expected exactly one split part with recording id {part_id!r} in {splits}, found {parts} among {written}")
        for later_part in written:
            if later_part != parts[0]:
                later_part.unlink()
        static_path = work / "static.rrd"
        ex.RrdReader(str(fixed)).stream().filter(is_static=True).write_rrd(static_path, application_id=app_id, recording_id=part_id)
        inputs += [parts[0], static_path]
        summary.update({
            "clip_requested_s": args.clip_seconds,
            "clip_start_s": start_ns / 1e9,
            "clip_end_s": cut_ns / 1e9,
            "clip_actual_s": (cut_ns - start_ns) / 1e9,
            "clip_note": note,
            "keyframes_s": {k: [t / 1e9 for t in v[:8]] for k, v in keyframes.items()},
        })
    else:
        inputs.append(args.recording)

    if do_loop:
        loop_src = work / "loop-src.rbl"
        state = {"collapsed": rrb.PanelState.Collapsed, "expanded": rrb.PanelState.Expanded, "hidden": rrb.PanelState.Hidden}[args.time_panel]
        # auto_layout/auto_views must stay False: this fragment is merged over a blueprint that has a real container.
        rrb.Blueprint(rrb.TimePanel(loop_mode="All", state=state), auto_layout=False, auto_views=False).save(app_id, str(loop_src))
        loop_path = work / "loop.rbl"
        rerun_cli("rrd", "route", "--recording-id", blueprint_store.recording_id, str(loop_src), "-o", str(loop_path))
        # This fragment overrides the blueprint's own /time_panel because its chunks carry newer RowIds, which is
        # what the viewer resolves on. The order of the inputs to `rrd merge` does not decide it.
        inputs.append(loop_path)
    inputs.append(args.blueprint)

    merged = work / "merged.rrd"
    rerun_cli("rrd", "merge", *map(str, inputs), "-o", str(merged))
    rerun_cli("rrd", "optimize", "--fix-keyframe", str(merged), "-o", str(args.out))
    rerun_cli("rrd", "verify", str(args.out))

    out_recordings, out_blueprints = stores(args.out)
    if len(out_recordings) != 1 or len(out_blueprints) != 1:
        fail(f"{args.out} ended with recordings={out_recordings} blueprints={out_blueprints}; expected one of each")
    summary.update({
        "out_bytes": args.out.stat().st_size,
        "recording_id": out_recordings[0].recording_id,
        "blueprint_id": out_blueprints[0].recording_id,
    })
    if args.keep_work:
        summary["work_dir"] = str(work)
    else:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
