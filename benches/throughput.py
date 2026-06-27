#!/usr/bin/env python3
"""Reproducible throughput harness for the PyRoboFrames dataloader.

Generates a synthetic LeRobotDataset v3.0 (tabular always; camera video too if ``ffmpeg`` is on
PATH), then measures **frames/s** for a full epoch across several ``num_workers`` settings — so the
off-GIL prefetch pipeline can be compared against the synchronous path on the same data.

Honesty note: this is synthetic data on a single machine; treat the numbers as a relative
sync-vs-prefetch signal, not an absolute benchmark vs other libraries.

Run:  python benches/throughput.py                 # auto-detects ffmpeg for the video case
      python benches/throughput.py --episodes 8 --length 200 --batch-size 64 --no-video
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import pyroboframes as prf

CAM = "observation.images.top"


def make_dataset(root: str, episodes: int, length: int, with_video: bool, vw: int, vh: int) -> int:
    total = episodes * length
    os.makedirs(f"{root}/meta/episodes/chunk-000")
    os.makedirs(f"{root}/data/chunk-000")

    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": episodes,
        "total_frames": total,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            CAM: {"dtype": "video", "shape": [vh, vw, 3]},
            "observation.state": {"dtype": "float32", "shape": [7]},
            "action": {"dtype": "float32", "shape": [7]},
        },
    }
    with open(f"{root}/meta/info.json", "w") as fh:
        json.dump(info, fh)

    ep = pa.table(
        {
            "episode_index": pa.array(range(episodes), pa.int64()),
            "length": pa.array([length] * episodes, pa.int64()),
            "dataset_from_index": pa.array([i * length for i in range(episodes)], pa.int64()),
            "dataset_to_index": pa.array([(i + 1) * length for i in range(episodes)], pa.int64()),
            "data/chunk_index": pa.array([0] * episodes, pa.int64()),
            "data/file_index": pa.array([0] * episodes, pa.int64()),
            f"videos/{CAM}/chunk_index": pa.array([0] * episodes, pa.int64()),
            f"videos/{CAM}/file_index": pa.array([0] * episodes, pa.int64()),
            f"videos/{CAM}/from_timestamp": pa.array(
                [i * length / 30 for i in range(episodes)], pa.float64()
            ),
            f"videos/{CAM}/to_timestamp": pa.array(
                [(i + 1) * length / 30 for i in range(episodes)], pa.float64()
            ),
        }
    )
    pq.write_table(ep, f"{root}/meta/episodes/chunk-000/file-000.parquet")

    rng = np.random.default_rng(0)
    data = pa.table(
        {
            "observation.state": pa.array(
                rng.standard_normal((total, 7)).tolist(), pa.list_(pa.float32())
            ),
            "action": pa.array(rng.standard_normal((total, 7)).tolist(), pa.list_(pa.float32())),
        }
    )
    pq.write_table(data, f"{root}/data/chunk-000/file-000.parquet")

    if with_video:
        vdir = f"{root}/videos/{CAM}/chunk-000"
        os.makedirs(vdir)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={vw}x{vh}:rate=30",
             "-frames:v", str(total), "-pix_fmt", "yuv420p", f"{vdir}/file-000.mp4"],
            check=True,
        )
    return total


def time_epoch(
    ds,
    total: int,
    *,
    cameras=None,
    batch_size: int,
    num_workers: int = 0,
    output: str = "numpy",
    delta_timestamps=None,
    chunk_size: int = 0,
) -> float:
    """Frames/s for one full epoch (best of 3 runs)."""
    best = 0.0
    for _ in range(3):
        loader = ds.loader(
            batch_size=batch_size,
            shuffle=True,
            cameras=cameras,
            num_workers=num_workers,
            output=output,
            delta_timestamps=delta_timestamps,
            chunk_size=chunk_size,
        )
        t0 = time.perf_counter()
        n = 0
        for batch in loader:
            n += len(batch["episode_index"])
        dt = time.perf_counter() - t0
        best = max(best, n / dt if dt > 0 else 0.0)
    return best


def available_outputs() -> list[str]:
    """Output frameworks installed in this environment, NumPy first."""
    outs = ["numpy"]
    for name, mod in (("torch", "torch"), ("mlx", "mlx.core"), ("jax", "jax")):
        try:
            __import__(mod)
            outs.append(name)
        except ImportError:
            pass
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--length", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4])
    ap.add_argument("--video-size", type=int, nargs=2, default=[64, 48], metavar=("W", "H"))
    ap.add_argument("--no-video", action="store_true", help="skip the camera-decode benchmark")
    args = ap.parse_args()

    have_ffmpeg = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    with_video = have_ffmpeg and not args.no_video

    root = tempfile.mkdtemp(prefix="prf_bench_")
    try:
        vw, vh = args.video_size
        total = make_dataset(root, args.episodes, args.length, with_video, vw, vh)
        ds = prf.RoboFrameDataset.from_path(root)

        print(f"\nPyRoboFrames throughput — {total} frames "
              f"({args.episodes} ep × {args.length}), batch={args.batch_size}")
        print(f"ffmpeg: {'yes' if have_ffmpeg else 'no'}  | video case: {'on' if with_video else 'off'}\n")

        cases = [("tabular (state/action)", None)]
        if with_video:
            cases.append((f"+ camera {vw}x{vh} (FFmpeg decode)", [CAM]))

        for label, cameras in cases:
            print(f"== {label} ==")
            print(f"{'num_workers':>12} | {'frames/s':>12} | {'speedup':>8}")
            print("-" * 38)
            base = None
            for w in args.workers:
                fps = time_epoch(ds, total, cameras=cameras, batch_size=args.batch_size, num_workers=w)
                base = base or fps
                tag = "sync" if w == 0 else f"{w}"
                print(f"{tag:>12} | {fps:>12,.0f} | {fps / base:>7.2f}x")
            print()

        # Output-framework comparison (tabular): how much the per-batch conversion to MLX / Torch /
        # JAX costs vs the native NumPy form. MLX is the Apple-Silicon path of interest.
        outs = available_outputs()
        print(f"== output framework (tabular, batch={args.batch_size}) ==")
        print(f"{'output':>12} | {'frames/s':>12} | {'vs numpy':>8}")
        print("-" * 38)
        base = None
        for out in outs:
            fps = time_epoch(ds, total, batch_size=args.batch_size, output=out)
            base = base or fps
            print(f"{out:>12} | {fps:>12,.0f} | {fps / base:>7.2f}x")
        print(f"   (available: {', '.join(outs)})\n")

        # Sequence batching: temporal-window + episode-chunking sampler, producing
        # [batch, steps, dim] sequences, in NumPy vs MLX (the MLX sequence-model feed path).
        steps = [-0.2, -0.1, 0.0]
        deltas = {"observation.state": steps, "action": steps}
        seq_outs = [o for o in outs if o in ("numpy", "mlx")]
        print(f"== sequence batching ({len(steps)}-step window, episode-chunked) ==")
        print(f"{'output':>12} | {'frames/s':>12} | {'vs numpy':>8}")
        print("-" * 38)
        base = None
        for out in seq_outs:
            fps = time_epoch(
                ds, total, batch_size=args.batch_size, output=out,
                delta_timestamps=deltas, chunk_size=args.length // 4 or 1,
            )
            base = base or fps
            print(f"{out:>12} | {fps:>12,.0f} | {fps / base:>7.2f}x")
        print()
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
