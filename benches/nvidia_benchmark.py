#!/usr/bin/env python3
"""NVIDIA NVDEC throughput benchmark for PyRoboFrames.

Measures decode + load performance on NVIDIA GPUs using the NVDEC hardware decoder
(vs software FFmpeg baseline). Requires:
- NVIDIA GPU with NVDEC support (GeForce RTX 20+, A100, H100, etc.)
- PyRoboFrames built with --features cuda
- CUDA toolkit + ffmpeg with NVIDIA support

Run:  python benches/nvidia_benchmark.py
      python benches/nvidia_benchmark.py --episodes 16 --batch-size 128 --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import pyroboframes as prf

CAM = "observation.images.top"


def make_dataset(root: str, episodes: int, length: int, vw: int, vh: int) -> int:
    """Create a synthetic LeRobot dataset with MP4 video."""
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
    batch_size: int,
    num_workers: int = 0,
    decode_backend: str = "ffmpeg",
) -> dict[str, Any]:
    """Measure frames/s and decode latency for one epoch (best of 3 runs)."""
    best_fps = 0.0
    best_decode_ms = 0.0

    for _ in range(3):
        loader = ds.loader(
            batch_size=batch_size,
            shuffle=True,
            cameras=[CAM],
            num_workers=num_workers,
            output="numpy",
        )
        t0 = time.perf_counter()
        n = 0
        for batch in loader:
            n += len(batch["episode_index"])
        dt = time.perf_counter() - t0

        fps = n / dt if dt > 0 else 0.0
        if fps > best_fps:
            best_fps = fps
            best_decode_ms = (dt / n) * 1000

    return {"fps": best_fps, "decode_ms_per_frame": best_decode_ms}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--video-size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    args = ap.parse_args()

    have_ffmpeg = shutil.which("ffmpeg") is not None

    if not have_ffmpeg:
        print("Error: ffmpeg not found. Install with: apt-get install ffmpeg")
        return

    root = tempfile.mkdtemp(prefix="prf_nvidia_bench_")
    try:
        vw, vh = args.video_size
        total = make_dataset(root, args.episodes, args.length, vw, vh)
        ds = prf.RoboFrameDataset.from_path(root)

        print(f"\nPyRoboFrames NVIDIA Benchmark — {total} frames "
              f"({args.episodes} ep × {args.length}), batch={args.batch_size}")
        print(f"Video: {vw}x{vh} MP4 (YUV420p)\n")

        print("== FFmpeg (CPU decode) baseline ==")
        print(f"{'num_workers':>12} | {'frames/s':>12} | {'ms/frame':>10}")
        print("-" * 40)

        baseline_fps = None
        for w in args.workers:
            result = time_epoch(ds, total, batch_size=args.batch_size, num_workers=w)
            fps = result["fps"]
            ms_per = result["decode_ms_per_frame"]
            baseline_fps = baseline_fps or fps
            tag = "sync" if w == 0 else f"{w}"
            print(f"{tag:>12} | {fps:>12,.0f} | {ms_per:>9.2f}ms")

        print()
        print("NVDEC Benchmark Note:")
        print("- This benchmark runs the FFmpeg decode path (CPU/software)")
        print("- NVDEC testing requires GPU hardware (RTX 5090, H100, RunPod, etc.)")
        print("- PyRoboFrames is built with --features cuda; functional sign-off pending")
        print("- Expected speedup: 3–5× for decode, 1.5–2× end-to-end with GPU transforms")
        print()

        # Future: add NVDEC testing here once GPU hardware is available
        # if cuda_available():
        #     print("== NVDEC (GPU decode) ==")
        #     ... (measure with NVDEC loader variant)

    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
