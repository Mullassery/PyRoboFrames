#!/usr/bin/env python3
"""GPU Support Verification Script for PyRoboFrames.

Checks for NVIDIA NVDEC and CV-CUDA availability, runs diagnostics,
and reports what GPU features are available.

Usage:
    python scripts/verify_gpu_support.py
    python scripts/verify_gpu_support.py --detailed
    python scripts/verify_gpu_support.py --run-benchmark
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Any

import numpy as np


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def check(condition: bool, label: str, details: str = "") -> None:
    """Print a check result."""
    status = "✓" if condition else "✗"
    print(f"{status} {label}")
    if details:
        print(f"  {details}")


def run_command(cmd: list[str]) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"
    except Exception as e:
        return False, str(e)


def check_ffmpeg() -> dict[str, Any]:
    """Check FFmpeg and NVDEC support."""
    results = {
        "ffmpeg_installed": False,
        "ffmpeg_version": None,
        "nvdec_h264": False,
        "nvdec_hevc": False,
        "nvdec_av1": False,
    }

    success, output = run_command(["ffmpeg", "-version"])
    if success:
        results["ffmpeg_installed"] = True
        version_line = output.split("\n")[0]
        results["ffmpeg_version"] = version_line.split("version")[1].strip().split()[0]

    # Check for NVDEC codecs
    success, output = run_command(["ffmpeg", "-codecs"])
    if success:
        results["nvdec_h264"] = "h264_nvdec" in output
        results["nvdec_hevc"] = "hevc_nvdec" in output
        results["nvdec_av1"] = "av1_nvdec" in output

    return results


def check_nvidia_smi() -> dict[str, Any]:
    """Check nvidia-smi and GPU availability."""
    results = {
        "nvidia_smi_available": False,
        "gpu_count": 0,
        "gpus": [],
    }

    success, output = run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])
    if success:
        results["nvidia_smi_available"] = True
        lines = output.strip().split("\n")[1:]  # Skip header
        results["gpu_count"] = len(lines)
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                results["gpus"].append({
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_mb": int(parts[2].split()[0]) if parts[2] else None,
                })

    return results


def check_cuda() -> dict[str, Any]:
    """Check CUDA availability via PyTorch or cupy."""
    results = {
        "cuda_available": False,
        "cuda_version": None,
        "torch_available": False,
        "torch_cuda": False,
    }

    # Check torch
    try:
        import torch
        results["torch_available"] = True
        results["torch_cuda"] = torch.cuda.is_available()
        if results["torch_cuda"]:
            results["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return results


def check_cvcuda() -> dict[str, Any]:
    """Check CV-CUDA availability."""
    results = {
        "cvcuda_installed": False,
        "cvcuda_version": None,
    }

    try:
        import cvcuda
        results["cvcuda_installed"] = True
        results["cvcuda_version"] = cvcuda.__version__
    except ImportError:
        pass

    return results


def check_pyroboframes() -> dict[str, Any]:
    """Check PyRoboFrames GPU features."""
    results = {
        "pyroboframes_installed": False,
        "version": None,
        "cuda_feature": False,
        "videotoolbox_feature": False,
        "preferred_backend": None,
    }

    try:
        import pyroboframes as prf
        results["pyroboframes_installed"] = True
        results["version"] = prf.__version__

        # Check which features are compiled
        try:
            from pyroboframes._core import Backend
            results["preferred_backend"] = str(Backend.preferred())
            # Check if CUDA feature is available (indirectly by checking if CudaDecoder exists)
            try:
                from pyroboframes._core import CudaDecoder
                results["cuda_feature"] = True
            except (ImportError, AttributeError):
                results["cuda_feature"] = False

            # Check if VideoToolbox feature is available
            try:
                from pyroboframes._core import VideoToolboxDecoder
                results["videotoolbox_feature"] = True
            except (ImportError, AttributeError):
                results["videotoolbox_feature"] = False
        except Exception as e:
            results["error"] = str(e)

    except ImportError:
        pass

    return results


def check_transform_backends() -> dict[str, Any]:
    """Check available transform backends."""
    results = {
        "mlx": False,
        "torch": False,
        "cvcuda": False,
        "numpy": True,  # Always available
        "resolved_backend": None,
    }

    try:
        from pyroboframes import transforms as T

        # Try each backend
        try:
            import mlx.core
            results["mlx"] = True
        except ImportError:
            pass

        try:
            import torch
            results["torch"] = True
        except ImportError:
            pass

        try:
            import cvcuda
            results["cvcuda"] = True
        except ImportError:
            pass

        # Check what backend is actually selected
        results["resolved_backend"] = T.resolve_transform_backend("auto")

    except ImportError:
        pass

    return results


def benchmark_transforms(detailed: bool = False) -> dict[str, Any]:
    """Quick benchmark of transform backends."""
    try:
        from pyroboframes import transforms as T
        import time as time_module

        results = {"success": False, "measurements": {}}

        # Create synthetic frames [N, H, W, C]
        frames = np.random.randint(0, 256, (8, 480, 640, 3), dtype=np.uint8)

        for backend in ["numpy", "torch", "mlx", "cvcuda"]:
            try:
                resize = T.Resize(224, 224)
                resize._backend = backend
                normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                normalize._backend = backend

                # Warm up
                try:
                    _ = resize(frames)
                    _ = normalize(frames)
                except ImportError:
                    continue

                # Benchmark
                t0 = time_module.perf_counter()
                for _ in range(10):
                    _ = resize(frames)
                dt_resize = (time_module.perf_counter() - t0) / 10

                t0 = time_module.perf_counter()
                for _ in range(10):
                    _ = normalize(frames)
                dt_norm = (time_module.perf_counter() - t0) / 10

                results["measurements"][backend] = {
                    "resize_ms": dt_resize * 1000,
                    "normalize_ms": dt_norm * 1000,
                }
                results["success"] = True

            except (ImportError, ValueError, Exception) as e:
                if detailed:
                    results["measurements"][backend] = {"error": str(e)}

        return results

    except ImportError:
        return {"success": False, "error": "PyRoboFrames not installed"}


def main() -> None:
    """Run all GPU verification checks."""
    parser = argparse.ArgumentParser(description="Verify GPU support in PyRoboFrames")
    parser.add_argument("--detailed", action="store_true", help="Show detailed diagnostics")
    parser.add_argument("--run-benchmark", action="store_true", help="Run transform benchmarks")
    args = parser.parse_args()

    all_results = {}

    # === Hardware ===
    section("HARDWARE & DRIVERS")
    nvidia_results = check_nvidia_smi()
    check(nvidia_results["nvidia_smi_available"], "nvidia-smi available")
    if nvidia_results["gpus"]:
        for i, gpu in enumerate(nvidia_results["gpus"]):
            print(f"  GPU {i}: {gpu['name']} ({gpu['memory_mb']} MB)")
    else:
        print("  No NVIDIA GPUs detected")
    all_results["nvidia_smi"] = nvidia_results

    # === FFmpeg ===
    section("FFMPEG & NVDEC")
    ffmpeg_results = check_ffmpeg()
    check(ffmpeg_results["ffmpeg_installed"], "FFmpeg installed", ffmpeg_results.get("ffmpeg_version", ""))
    check(ffmpeg_results["nvdec_h264"], "H.264 NVDEC decoder", "h264_nvdec")
    check(ffmpeg_results["nvdec_hevc"], "HEVC NVDEC decoder", "hevc_nvdec")
    check(ffmpeg_results["nvdec_av1"], "AV1 NVDEC decoder", "av1_nvdec")
    if not (ffmpeg_results["nvdec_h264"] or ffmpeg_results["nvdec_hevc"]):
        print("\n  ⚠️  FFmpeg does not support NVDEC. Install with NVIDIA support:")
        print("     conda install -c conda-forge ffmpeg")
    all_results["ffmpeg"] = ffmpeg_results

    # === CUDA ===
    section("CUDA & PYTORCH")
    cuda_results = check_cuda()
    check(cuda_results["cuda_available"] or cuda_results["torch_available"], "CUDA-capable PyTorch")
    check(cuda_results["torch_cuda"], "CUDA available in PyTorch", cuda_results.get("cuda_version", ""))
    all_results["cuda"] = cuda_results

    # === CV-CUDA ===
    section("CV-CUDA (TRANSFORM ACCELERATION)")
    cvcuda_results = check_cvcuda()
    check(cvcuda_results["cvcuda_installed"], "CV-CUDA installed", cvcuda_results.get("cvcuda_version", ""))
    if not cvcuda_results["cvcuda_installed"]:
        print("\n  To install CV-CUDA:")
        print("    pip install cvcuda-cu12  # CUDA 12.x")
        print("    pip install cvcuda-cu11  # CUDA 11.x")
    all_results["cvcuda"] = cvcuda_results

    # === PyRoboFrames ===
    section("PYROBOFRAMES")
    prf_results = check_pyroboframes()
    check(prf_results["pyroboframes_installed"], "PyRoboFrames installed", prf_results.get("version", ""))
    check(prf_results["cuda_feature"], "CUDA decoder (--features cuda)", prf_results.get("preferred_backend", ""))
    check(prf_results["videotoolbox_feature"], "VideoToolbox decoder (macOS)")
    all_results["pyroboframes"] = prf_results

    # === Transform Backends ===
    section("TRANSFORM BACKENDS")
    tf_results = check_transform_backends()
    check(tf_results["cvcuda"], "CV-CUDA backend available")
    check(tf_results["mlx"], "MLX backend available")
    check(tf_results["torch"], "Torch backend available")
    print(f"\n  Resolved backend (auto): {tf_results['resolved_backend']}")
    print(f"  Fallback chain: CV-CUDA → MLX → Torch → NumPy")
    all_results["transforms"] = tf_results

    # === Benchmarks ===
    if args.run_benchmark:
        section("TRANSFORM BENCHMARKS")
        bench_results = benchmark_transforms(detailed=args.detailed)
        if bench_results.get("success"):
            for backend, metrics in bench_results["measurements"].items():
                if "error" not in metrics:
                    print(f"  {backend:10} | Resize: {metrics['resize_ms']:7.2f}ms | Normalize: {metrics['normalize_ms']:7.2f}ms")
        else:
            print(f"  Benchmark failed: {bench_results.get('error', 'unknown')}")
        all_results["benchmarks"] = bench_results

    # === Summary ===
    section("SUMMARY")
    summary = {
        "gpu_available": nvidia_results["gpu_count"] > 0,
        "nvdec_ready": ffmpeg_results["nvdec_h264"] or ffmpeg_results["nvdec_hevc"],
        "cuda_ready": cuda_results["torch_cuda"],
        "cvcuda_ready": cvcuda_results["cvcuda_installed"],
        "pyroboframes_ready": prf_results["pyroboframes_installed"],
    }

    print("GPU Support Summary:")
    print(f"  GPU Hardware:       {'✓' if summary['gpu_available'] else '✗'}")
    print(f"  NVDEC (decode):     {'✓' if summary['nvdec_ready'] else '✗'}")
    print(f"  CUDA (PyTorch):     {'✓' if summary['cuda_ready'] else '✗'}")
    print(f"  CV-CUDA (transforms): {'✓' if summary['cvcuda_ready'] else '✗'}")
    print(f"  PyRoboFrames:       {'✓' if summary['pyroboframes_ready'] else '✗'}")

    if args.detailed:
        print("\n\nDetailed Results (JSON):")
        print(json.dumps(all_results, indent=2, default=str))

    # Recommendations
    print("\n\nNext Steps:")
    if not summary["gpu_available"]:
        print("  • No NVIDIA GPU detected; GPU features not available on this machine")
    else:
        if not summary["nvdec_ready"]:
            print("  • Install FFmpeg with NVDEC: conda install -c conda-forge ffmpeg")
        if not summary["cvcuda_ready"]:
            print("  • Install CV-CUDA: pip install cvcuda-cu12 (or cvcuda-cu11)")
        if summary["nvdec_ready"] and summary["cuda_ready"]:
            print("  • Run GPU benchmark: python benches/nvidia_benchmark.py")
        if summary["cvcuda_ready"]:
            print("  • Test transforms: python -c \"from pyroboframes import transforms as T; ...\"")

    print()


if __name__ == "__main__":
    main()
