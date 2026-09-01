#!/usr/bin/env python3
"""Pull frames and short clips out of a dataset that already exists.

The repo could preview what the factory WOULD generate (``sim_preview.py``) and
re-render what it did (``sim_rerender.py``), but nothing read a shipped dataset
back. Looking at delivered data meant hand-rolling ffmpeg and rediscovering how
episodes map onto video files — which is not obvious: LeRobot v3 concatenates
many episodes into each mp4, so a frame is addressed by
``(videos/<cam>/file_index, from_timestamp)`` out of ``meta/episodes/*.parquet``.
That mapping lives here now instead of in a scratch script.

Captions and pixels come from the SAME metadata row, so a sample's prompt cannot
drift away from the frames under it.

Point it at one dataset, or at a directory of shards — for a family it takes one
episode from each of several shards rather than several from one, because a
shard is a seed and a domain-randomization draw, and five episodes out of one
shard show a single lighting rig and pad height.

    cd python/tatbot_sim && uv run python ../../scripts/sim_dataset_samples.py \
        --path ~/tatbot-sim/datasets/overnight-20260827/paper-draw-s01 --out /tmp/samples
    # a whole night, sampled across shards:
    uv run python ../../scripts/sim_dataset_samples.py \
        --path ~/tatbot-sim/datasets/overnight-20260827 --out /tmp/samples --samples 6

Writes WebP stills, an animated WebP of the episode's last seconds, the pigment
field when the run saved one, and a manifest.json describing every sample.
"""

from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro

UPPER = "observation.images.wrist_upper"
LOWER = "observation.images.wrist_lower"


@dataclass
class Args:
    path: Path
    """A single dataset, or a directory holding several shards."""
    out: Path
    """Where to write stills, clips and manifest.json."""
    samples: int = 5
    """Episodes per dataset (or, for a family, shards to draw one episode from)."""
    stills: tuple[float, ...] = (0.05, 0.35, 0.65, 0.97)
    """Fractions of each episode to grab stills at — a time axis, left to right."""
    clip_seconds: float = 4.0
    """Length of the animated clip, taken from the END of the episode: the first
    seconds are approach, the last are the drawing or the erasure."""
    clip_fps: int = 10
    width: int = 320
    """Clip width, px."""
    still_width: int = 384
    lower_camera: bool = True
    """Also grab one still from the lower wrist camera."""
    quality: int = 82
    """WebP quality for stills. These are photographic renders and a caller may
    be inlining them into a page, so they are not PNG."""


def _shards(root: Path) -> list[Path]:
    if (root / "meta/info.json").exists():
        return [root]
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith(".")
                  and (p / "meta/info.json").exists())


def _episodes(ds: Path) -> pd.DataFrame:
    cols = ["episode_index", "tasks", "length"]
    for cam in (UPPER, LOWER):
        cols += [f"videos/{cam}/{s}" for s in
                 ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    files = sorted(glob.glob(str(ds / "meta/episodes/*/*.parquet")))
    if not files:
        raise SystemExit(f"no episode metadata under {ds}")
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                   ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def _ffmpeg(args: list[str]) -> bool:
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args],
                       capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode()[-600:] + "\n")
    return r.returncode == 0


def _video(ds: Path, cam: str, chunk: int, fidx: int) -> Path:
    return ds / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.mp4"


def _task_of(row) -> str:
    t = row.tasks
    return str(t if isinstance(t, str) else t[0])


def _extract(ds: Path, row, out: Path, prefix: str, a: Args) -> dict:
    ep = int(row.episode_index)
    t0 = float(row[f"videos/{UPPER}/from_timestamp"])
    t1 = float(row[f"videos/{UPPER}/to_timestamp"])
    src = _video(ds, UPPER, int(row[f"videos/{UPPER}/chunk_index"]),
                 int(row[f"videos/{UPPER}/file_index"]))
    dur = t1 - t0
    entry = {"shard": ds.name, "episode": ep, "task": _task_of(row),
             "length": int(row.length), "seconds": round(dur, 2),
             "stills": [], "clip": None, "lower": None, "field": None}

    for k, frac in enumerate(a.stills):
        name = f"{prefix}ep{ep:04d}_s{k}.webp"
        if _ffmpeg(["-ss", f"{t0 + frac * dur:.3f}", "-i", str(src), "-frames:v", "1",
                    "-vf", f"scale={a.still_width}:-2:flags=lanczos",
                    "-quality", str(a.quality), str(out / name)]):
            entry["stills"].append(name)

    cs = min(a.clip_seconds, dur)
    name = f"{prefix}ep{ep:04d}.webp"
    if _ffmpeg(["-ss", f"{max(t0, t1 - cs):.3f}", "-t", f"{cs:.3f}", "-i", str(src),
                "-vf", f"fps={a.clip_fps},scale={a.width}:-2:flags=lanczos",
                "-loop", "0", "-q:v", "62", "-compression_level", "6",
                str(out / name)]):
        entry["clip"] = name

    if a.lower_camera:
        lsrc = _video(ds, LOWER, int(row[f"videos/{LOWER}/chunk_index"]),
                      int(row[f"videos/{LOWER}/file_index"]))
        l0 = float(row[f"videos/{LOWER}/from_timestamp"])
        l1 = float(row[f"videos/{LOWER}/to_timestamp"])
        name = f"{prefix}ep{ep:04d}_lower.webp"
        if _ffmpeg(["-ss", f"{l0 + 0.9 * (l1 - l0):.3f}", "-i", str(lsrc),
                    "-frames:v", "1",
                    "-vf", f"scale={a.still_width}:-2:flags=lanczos",
                    "-quality", str(a.quality), str(out / name)]):
            entry["lower"] = name

    # Ground truth for what ended up on the surface, when the run saved it
    # (--save-field-snapshots). Worth carrying next to the frames: the wrist
    # views are mostly filled by the tool, and this is the picture that shows
    # whether the prompt was actually drawn.
    fld = ds / "meta/fields" / f"episode_{ep:06d}.png"
    if fld.exists():
        name = f"{prefix}ep{ep:04d}_field.png"
        shutil.copy(fld, out / name)
        entry["field"] = name
    return entry


def _spread(n: int, total: int) -> list[int]:
    """Evenly spaced indices, so a sample is not all one end of the run."""
    n = min(n, total)
    return sorted({round(i * (total - 1) / max(n - 1, 1)) for i in range(n)})


def main(a: Args) -> None:
    root = a.path.expanduser()
    shards = _shards(root)
    if not shards:
        raise SystemExit(f"no finished datasets under {root}")
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    samples = []
    if len(shards) == 1:
        ds = shards[0]
        df = _episodes(ds)
        for i in _spread(a.samples, len(df)):
            samples.append(_extract(ds, df.iloc[i], out, "", a))
    else:
        # One episode per shard, spread across the run: each shard is its own
        # seed and DR draw, so this varies lighting and pad height too.
        for k, idx in enumerate(_spread(a.samples, len(shards))):
            ds = shards[idx]
            df = _episodes(ds)
            row = df.iloc[(k * 37 + 11) % len(df)]     # a different slot each time
            samples.append(_extract(ds, row, out, f"{ds.name}_", a))

    for s in samples:
        print(f"{s['shard']:24s} ep{s['episode']:04d} {s['seconds']:6.2f}s  "
              f"{s['task'][:64]}")
    (out / "manifest.json").write_text(json.dumps(
        {"source": str(root), "shards": len(shards), "samples": samples}, indent=2))
    print(f"\nwrote {len(samples)} samples to {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
