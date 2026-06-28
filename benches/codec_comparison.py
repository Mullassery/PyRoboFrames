#!/usr/bin/env python3
"""Codec storage and speed comparison: H.264 vs HEVC vs AV1.

Writes a synthetic dataset with each codec, measures disk usage and encode time,
then prints a comparison table.

Usage:
    python benches/codec_comparison.py --frames 500 --resolution 480x270
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def create_synthetic_features(n_frames: int) -> dict:
    rng = np.random.default_rng(42)
    return {
        "observation.state": rng.random((n_frames, 14), dtype=np.float32),
        "action": rng.random((n_frames, 14), dtype=np.float32),
    }


def measure_dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            total += os.path.getsize(os.path.join(dirpath, fname))
    return total / (1024 * 1024)


def benchmark_codec(
    codec: str,
    n_frames: int,
    profile: str | None,
    description: str,
    crf: int = 23,
) -> dict:
    import pyroboframes as prf

    features = create_synthetic_features(n_frames)
    episode_lengths = [n_frames // 2, n_frames - n_frames // 2]

    with tempfile.TemporaryDirectory() as out_dir:
        t0 = time.perf_counter()
        prf.write_lerobot_dataset(
            out_dir,
            features,
            episode_lengths,
            video_codec=codec,
            video_profile=profile,
            video_crf=crf,
        )
        elapsed = time.perf_counter() - t0
        size_mb = measure_dir_size_mb(out_dir)

    return {
        "codec": description,
        "size_mb": size_mb,
        "encode_s": elapsed,
        "mb_per_frame": size_mb / n_frames,
    }


def print_table(rows: list[dict], n_frames: int, extrapolate_to: int = 10_000) -> None:
    print(f"\n{'─'*72}")
    print(f"  Codec Comparison  │  {n_frames} frames  │  extrapolated to {extrapolate_to:,} frames")
    print(f"{'─'*72}")
    header = f"  {'Codec':<20} {'Size (MB)':>10} {'Encode (s)':>12} {'MB/frame':>10} {'Est. 10k (GB)':>14}"
    print(header)
    print(f"  {'─'*68}")
    for row in rows:
        est_gb = (row["mb_per_frame"] * extrapolate_to) / 1024
        print(
            f"  {row['codec']:<20} {row['size_mb']:>10.2f} {row['encode_s']:>12.2f} "
            f"{row['mb_per_frame']:>10.4f} {est_gb:>14.2f}"
        )
    print(f"{'─'*72}\n")

    if len(rows) > 1:
        baseline = rows[0]
        print("  Savings vs H.264:")
        for row in rows[1:]:
            savings = (1 - row["size_mb"] / max(baseline["size_mb"], 1e-9)) * 100
            print(f"    {row['codec']:<20} {savings:+.1f}%")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="PyRoboFrames codec benchmark")
    parser.add_argument("--frames", type=int, default=200, help="Total frames to encode")
    parser.add_argument("--crf", type=int, default=23, help="CRF quality setting")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found on PATH. Install ffmpeg to run codec benchmarks.")
        sys.exit(1)

    n_frames = args.frames
    crf = args.crf

    codecs = [
        ("h264", None, "H.264 (libx264)"),
        ("hevc", "main", "HEVC/H.265 (main)"),
        ("av1", None, "AV1 (libsvtav1)"),
    ]

    print(f"\nRunning codec benchmark ({n_frames} frames, CRF={crf}) …")
    print("This may take a minute — AV1 is slow to encode.\n")

    results = []
    for codec, profile, description in codecs:
        print(f"  Encoding {description} …", end="", flush=True)
        try:
            result = benchmark_codec(codec, n_frames, profile, description, crf)
            results.append(result)
            print(f" {result['size_mb']:.2f} MB in {result['encode_s']:.1f}s")
        except Exception as exc:
            print(f" FAILED: {exc}")

    if results:
        print_table(results, n_frames)


if __name__ == "__main__":
    main()
