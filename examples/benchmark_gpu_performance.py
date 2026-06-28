"""GPU Performance Benchmarking for PyRoboFrames.

Benchmarks Phase 4a (GPU acceleration) on different hardware:
- NVIDIA CUDA (A100, H100, RTX 4090)
- Apple Silicon (M1/M2/M3 with MLX)
- CPU baseline (NumPy)

Run on target hardware to validate production performance.

Usage:
    # Benchmark on M1/M2 with MLX
    python benchmark_gpu_performance.py --device mlx --backend mlx

    # Benchmark on NVIDIA with CuPy
    python benchmark_gpu_performance.py --device cuda --backend cupy

    # CPU baseline
    python benchmark_gpu_performance.py --device cpu --backend numpy
"""

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np


class GPUBenchmark:
    """Benchmark GPU operations for automotive processing."""

    def __init__(self, device: str, backend_name: str):
        """Initialize benchmark.

        Args:
            device: "cuda", "mlx", or "cpu"
            backend_name: Name of backend for reporting
        """
        self.device = device
        self.backend_name = backend_name
        self.results = {}

    def benchmark_gaussian_blur(
        self,
        image_sizes: List[Tuple[int, int]] = None,
        iterations: int = 100,
    ) -> Dict[str, float]:
        """Benchmark Gaussian blur operation.

        Args:
            image_sizes: List of (H, W) sizes to test
            iterations: Number of iterations per size

        Returns:
            {image_size -> avg_ms}
        """
        if image_sizes is None:
            image_sizes = [(480, 640), (720, 1280), (1080, 1920)]

        results = {}

        for H, W in image_sizes:
            image = np.random.rand(H, W, 3).astype(np.float32)

            # Warmup
            _ = self._blur_operation(image)

            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = self._blur_operation(image)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_time = np.mean(times)
            results[f"{H}x{W}"] = avg_time

        return results

    def _blur_operation(self, image: np.ndarray) -> np.ndarray:
        """Execute blur operation on target backend.

        Args:
            image: [H, W, 3] image

        Returns:
            Blurred image
        """
        if self.backend_name == "cupy":
            try:
                import cupy as cp

                gpu_image = cp.asarray(image)
                from cupyx.scipy.ndimage import gaussian_filter

                blurred = gaussian_filter(gpu_image, sigma=1.0)

                return cp.asnumpy(blurred)
            except ImportError:
                raise ImportError("CuPy backend requires: pip install cupy-cuda11x")

        elif self.backend_name == "mlx":
            try:
                import mlx.core as mx
                from mlx.nn import gaussian_blur

                gpu_image = mx.array(image)
                blurred = gaussian_blur(gpu_image, sigma=1.0)

                return np.array(blurred)
            except ImportError:
                raise ImportError("MLX backend requires: pip install mlx")

        else:  # numpy
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(image, sigma=1.0).astype(np.float32)

    def benchmark_panorama_stitching(
        self,
        num_iterations: int = 10,
    ) -> Dict[str, float]:
        """Benchmark panorama stitching operation.

        Args:
            num_iterations: Number of stitching iterations

        Returns:
            {metric -> value}
        """
        # Simulate 5 camera frames (Waymo layout)
        frames = {
            cam: np.random.rand(720, 1280, 3).astype(np.uint8)
            for cam in ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
        }

        from pyroboframes.automotive import CylindricalStitcher, get_waymo_layout

        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout, device=self.device)

        # Warmup
        _ = stitcher.stitch(frames)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = stitcher.stitch(frames)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        fps = 1000 / avg_time if avg_time > 0 else 0

        return {
            "avg_ms": avg_time,
            "fps": fps,
            "device": self.device,
        }

    def benchmark_sam3_segmentation(
        self,
        num_iterations: int = 5,
    ) -> Dict[str, float]:
        """Benchmark SAM3 segmentation.

        Args:
            num_iterations: Number of segmentation iterations

        Returns:
            {metric -> value}
        """
        frame = np.random.rand(480, 640, 3).astype(np.uint8)

        try:
            from pyroboframes.automotive import SAM3Segmenter

            segmenter = SAM3Segmenter(device=self.device)

            # Warmup
            try:
                _ = segmenter.segment(frame)
            except ImportError:
                return {"status": "Model not available"}

            # Benchmark
            times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = segmenter.segment(frame)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_time = np.mean(times)
            fps = 1000 / avg_time if avg_time > 0 else 0

            return {
                "avg_ms": avg_time,
                "fps": fps,
                "device": self.device,
            }

        except Exception as e:
            return {"error": str(e)}

    def print_results(self, results: Dict) -> None:
        """Print benchmark results in tabular format.

        Args:
            results: Results dict
        """
        print(f"\n{'=' * 70}")
        print(f"Benchmark Results: {self.backend_name.upper()} on {self.device.upper()}")
        print(f"{'=' * 70}\n")

        for key, value in results.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.2f}")
                    else:
                        print(f"  {k}: {v}")
            else:
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")

        print()


def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark GPU performance")
    parser.add_argument(
        "--device",
        choices=["cuda", "mlx", "cpu"],
        default="cpu",
        help="Target device",
    )
    parser.add_argument(
        "--backend",
        choices=["cupy", "mlx", "numpy"],
        default="numpy",
        help="Backend to benchmark",
    )
    parser.add_argument(
        "--blur-only",
        action="store_true",
        help="Only benchmark Gaussian blur",
    )
    parser.add_argument(
        "--stitch-only",
        action="store_true",
        help="Only benchmark panorama stitching",
    )

    args = parser.parse_args()

    benchmark = GPUBenchmark(args.device, args.backend)

    all_results = {}

    # Gaussian blur
    if not args.stitch_only:
        print(f"Benchmarking Gaussian blur on {args.backend}...")
        blur_results = benchmark.benchmark_gaussian_blur()
        all_results["Gaussian Blur (ms)"] = blur_results

    # Panorama stitching
    if not args.blur_only:
        print(f"Benchmarking panorama stitching on {args.backend}...")
        try:
            stitch_results = benchmark.benchmark_panorama_stitching()
            all_results["Panorama Stitching"] = stitch_results
        except Exception as e:
            all_results["Panorama Stitching"] = {"error": str(e)}

    benchmark.print_results(all_results)

    # Print recommendations
    print("Recommendations:")
    print("- Run on target hardware (GPU box or M1/M2)")
    print("- Compare results across backends")
    print("- Target: GPU should be 2-10x faster than CPU")
    print("- For production: aim for >30 FPS on Waymo layout (5 cameras)")
    print()


if __name__ == "__main__":
    main()
