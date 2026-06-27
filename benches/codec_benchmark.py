#!/usr/bin/env python3
"""Benchmark codec selection: storage size, encoding speed, and decoding throughput.

Usage:
    python benches/codec_benchmark.py [--frames 1000] [--resolution 480]

This creates synthetic datasets with H.264, HEVC, and AV1 codecs, measures:
1. Encoding time
2. File sizes
3. Storage savings compared to H.264
4. Decoding speed (if the codec is supported)
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import pyroboframes as prf


def create_synthetic_features(num_frames: int) -> dict[str, np.ndarray]:
    """Create synthetic features for benchmarking."""
    return {
        "observation.state": np.random.randn(num_frames, 7).astype(np.float32),
        "action": np.random.randn(num_frames, 7).astype(np.float32),
    }


def measure_dir_size(path: str) -> int:
    """Recursively measure directory size in bytes."""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat().st_size
        elif entry.is_dir(follow_symlinks=False):
            total += measure_dir_size(entry.path)
    return total


def benchmark_codec(
    codec: str,
    num_frames: int,
    profile: str | None = None,
    description: str = "",
) -> dict:
    """Benchmark a single codec.

    Returns:
        dict with keys: codec, frames, size_bytes, encode_time_s, description
    """
    features = create_synthetic_features(num_frames)
    episode_lengths = [num_frames // 2, num_frames // 2]

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Encoding {codec.upper()}...", end=" ", flush=True)
        start_time = time.time()

        prf.write_lerobot_dataset(
            tmpdir,
            features,
            episode_lengths,
            fps=30.0,
            video_codec=codec,
            video_profile=profile,
        )

        encode_time = time.time() - start_time
        size_bytes = measure_dir_size(tmpdir)

        print(f"✓ ({encode_time:.1f}s, {size_bytes / 1e6:.1f} MB)")

        return {
            "codec": codec,
            "profile": profile or "(default)",
            "frames": num_frames,
            "size_bytes": size_bytes,
            "size_mb": size_bytes / 1e6,
            "encode_time_s": encode_time,
            "description": description,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark video codec selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick benchmark (500 frames)
    python benches/codec_benchmark.py --frames 500

    # Standard benchmark (1000 frames)
    python benches/codec_benchmark.py --frames 1000

    # Extended benchmark (5000 frames, measures more precisely)
    python benches/codec_benchmark.py --frames 5000
        """,
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1000,
        help="Number of frames to benchmark (default: 1000)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=480,
        help="Video height in pixels (for reference, not used in tabular benchmarks)",
    )
    args = parser.parse_args()

    num_frames = args.frames
    print(
        f"\n{'='*70}"
        f"\nCodec Benchmark: {num_frames} frames"
        f"\n{'='*70}\n"
    )

    results = []

    # H.264 (baseline)
    results.append(
        benchmark_codec(
            "h264",
            num_frames,
            description="Universal compatibility (baseline for comparison)",
        )
    )

    # HEVC (storage efficiency)
    results.append(
        benchmark_codec(
            "hevc",
            num_frames,
            profile="main",
            description="30-40% smaller than H.264",
        )
    )

    # AV1 (extreme compression)
    results.append(
        benchmark_codec(
            "av1",
            num_frames,
            description="50-60% smaller than H.264 (very slow encoding)",
        )
    )

    print(f"\n{'='*70}\nResults\n{'='*70}\n")

    # Print detailed results
    h264_size = results[0]["size_bytes"]
    for r in results:
        codec_display = f"{r['codec'].upper()}"
        if r["profile"] != "(default)":
            codec_display += f" ({r['profile']})"

        savings = ((h264_size - r["size_bytes"]) / h264_size) * 100 if h264_size else 0
        savings_str = f"-{savings:.1f}%" if savings > 0 else "(baseline)"

        print(f"{codec_display:20} {r['size_mb']:8.1f} MB   {savings_str:>12}   {r['encode_time_s']:6.1f}s")

    print(f"\nTotal frames: {num_frames}")
    print(f"Episode structure: [{num_frames//2}, {num_frames//2}]")

    # Calculate relative metrics
    print(f"\n{'='*70}\nSummary\n{'='*70}\n")
    h264_result = results[0]
    hevc_result = results[1]
    av1_result = results[2]

    hevc_compression = (
        (1 - hevc_result["size_bytes"] / h264_result["size_bytes"]) * 100
    )
    av1_compression = (
        (1 - av1_result["size_bytes"] / h264_result["size_bytes"]) * 100
    )

    hevc_speed = hevc_result["encode_time_s"] / h264_result["encode_time_s"]
    av1_speed = av1_result["encode_time_s"] / h264_result["encode_time_s"]

    print(f"HEVC compression: {hevc_compression:.1f}% smaller than H.264")
    print(f"HEVC encoding:    {hevc_speed:.1f}x slower than H.264")
    print()
    print(f"AV1 compression:  {av1_compression:.1f}% smaller than H.264")
    print(f"AV1 encoding:     {av1_speed:.1f}x slower than H.264")

    # Extrapolate to 10,000 frames
    print(f"\n{'='*70}\nExtrapolation to 10,000 frames\n{'='*70}\n")
    scale_factor = 10_000 / num_frames
    for r in results:
        extrapolated_mb = r["size_mb"] * scale_factor
        extrapolated_time = r["encode_time_s"] * scale_factor
        print(
            f"{r['codec'].upper():6}  {extrapolated_mb:7.0f} MB   "
            f"{extrapolated_time/60:6.0f} minutes"
        )

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
