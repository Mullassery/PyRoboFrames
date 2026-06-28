# PyRoboFrames Benchmarks

The headline metric for PyRoboFrames is **decode + load throughput** — measuring frames/second 
(FPS) across different hardware backends and configurations.

## Running Benchmarks

### Prerequisites

```bash
# Install benchmark dependencies
pip install pyroboframes pytest pytest-benchmark

# For NVIDIA GPU benchmarks (optional)
pip install cupy-cuda11x torch[cuda]
```

### Available Benchmarks

#### 1. Codec Benchmark (`codec_benchmark.py`)
Tests video decode performance across backends:
- **VideoToolbox** (macOS Apple Silicon) — hardware-accelerated H.264/H.265
- **NVDEC** (NVIDIA CUDA) — GPU-accelerated decode
- **FFmpeg** (CPU) — software decode baseline

```bash
python codec_benchmark.py --codec h264 --frames 100
```

Measures: **frames decoded/second, memory usage, decode latency**

#### 2. Throughput Benchmark (`throughput.py`)
End-to-end dataloader throughput: decode + batching + output conversion

```bash
python throughput.py --batch-size 64 --num-workers 4 --device mlx
```

Measures: **batches/sec, total images/sec, end-to-end latency**

#### 3. NVIDIA Benchmark (`nvidia_benchmark.py`)
CUDA-specific decode performance (NVIDIA GPUs only)

```bash
python nvidia_benchmark.py --codec h264 --batch-size 128
```

Measures: **GPU memory, PCIe throughput, hardware decode FPS**

## Expected Results (v1.0.0)

### Apple Silicon (M3 Max)
- **VideoToolbox hardware decode:** 100+ FPS (H.264)
- **Dataloader throughput:** 50–100 images/sec with 4 workers
- **Memory overhead:** ~2GB for 64-image batch

### NVIDIA (RTX 4090)
- **NVDEC hardware decode:** 200+ FPS (H.264/H.265)
- **Dataloader throughput:** 150–200 images/sec with 8 workers
- **VRAM usage:** ~8GB for 64-image batch

### CPU (Baseline)
- **FFmpeg software decode:** 5–10 FPS
- **Expected speedup vs. CPU:** 10–20x with hardware decode

## Continuous Benchmarking

To track performance over time:

```bash
# Save baseline
python codec_benchmark.py --save baseline.json

# Run tests, then compare
python codec_benchmark.py --compare baseline.json
```

## Reporting Issues

If you see unexpected performance:
1. Check your system load (`top`, `Activity Monitor`)
2. Verify hardware acceleration is enabled
3. Include benchmark output and system info in your issue
