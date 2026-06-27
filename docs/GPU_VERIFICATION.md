# GPU Verification Guide

This document describes how to verify PyRoboFrames GPU acceleration features (NVIDIA NVDEC, CV-CUDA) on hardware.

## Prerequisites

### NVIDIA NVDEC (Decode Acceleration)

**Hardware Requirements:**
- NVIDIA GPU with NVDEC support: GeForce RTX 20+, A100, H100, RTX 5090, etc.
- Check [NVIDIA NVDEC support matrix](https://developer.nvidia.com/nvidia-video-codec-sdk)

**Software Requirements:**
```bash
# CUDA Toolkit 11.x or 12.x (matches your ffmpeg build)
# NVIDIA driver >= 450

# FFmpeg built with NVDEC support
ffmpeg -codecs | grep hevc_nvdec  # Should output the decoder

# PyRoboFrames built with --features cuda
pip install --no-binary :all: pyroboframes --config-settings="--build-option=--features=cuda"
# Or from source:
pip install -e . --no-build-isolation -C="setup-args=--features=cuda"
```

**Verification:**
```python
import pyroboframes as prf
ds = prf.RoboFrameDataset.from_path("/path/to/dataset")
# Loader will auto-select CudaDecoder on Linux with --features cuda
loader = ds.loader(batch_size=32, cameras=["observation.images.top"])
batch = next(iter(loader))
print(f"Decoded batch shape: {batch['observation.images.top'].shape}")
```

### CV-CUDA (Transform Acceleration)

**Hardware Requirements:**
- Any NVIDIA GPU with CUDA Compute Capability 5.0+ (Maxwell era+)

**Software Requirements:**
```bash
# Install CV-CUDA (separate from CUDA Toolkit)
pip install cvcuda-cu12  # For CUDA 12.x
# OR
pip install cvcuda-cu11  # For CUDA 11.x

# Verify installation
python -c "import cvcuda; print(cvcuda.__version__)"
```

**Verification:**
```python
from pyroboframes import transforms as T
import numpy as np

# Create test frames [N, H, W, C]
frames = np.random.randint(0, 256, (32, 480, 640, 3), dtype=np.uint8)

# Transform uses CV-CUDA automatically if available
transform = T.Resize(224, 224)
result = transform(frames)
print(f"Resized shape: {result.shape}")  # Should be [32, 224, 224, 3]
```

---

## Running the Verification Suite

### 1. NVDEC Decode Throughput Benchmark

```bash
cd ~/PyRoboFrames
python benches/nvidia_benchmark.py \
  --episodes 8 \
  --length 200 \
  --batch-size 64 \
  --workers 2 \
  --video-size 640 480
```

**Expected Output:**
```
PyRoboFrames NVIDIA Benchmark — 1600 frames (8 ep × 200), batch=64
Video: 640x480 MP4 (YUV420p)

== FFmpeg (CPU decode) baseline ==
num_workers |     frames/s |     ms/frame
------------ | ----------- | ----------
        sync |        1200 |       0.83ms
           1 |        1800 |       0.56ms
           2 |        2400 |       0.42ms

NVDEC Benchmark Note:
- This benchmark runs the FFmpeg decode path (CPU/software)
- NVDEC testing requires GPU hardware (RTX 5090, H100, RunPod, etc.)
- PyRoboFrames is built with --features cuda; functional sign-off pending
- Expected speedup: 3–5× for decode, 1.5–2× end-to-end with GPU transforms
```

**Note:** The benchmark above shows FFmpeg baseline. To test NVDEC:
- Run with `-hwaccel cuda` enabled in `ffmpeg` (automatic if supported)
- Compare throughput vs FFmpeg CPU path

### 2. Transform Backend Resolution

```python
from pyroboframes import transforms as T

# Check which backend is active
backend = T.resolve_transform_backend("auto")
print(f"Active transform backend: {backend}")

# Expected fallback chain on GPU machine with CV-CUDA:
# CV-CUDA > MLX > Torch > NumPy
```

### 3. Full Pipeline Integration Test

```python
import pyroboframes as prf
from pyroboframes import transforms as T
import numpy as np

# Create synthetic LeRobot dataset
ds = prf.RoboFrameDataset.from_path("/path/to/lerobot/dataset")

# Loader with GPU transforms
transform_pipeline = T.Compose([
    T.Resize(224, 224),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

loader = ds.loader(
    batch_size=64,
    cameras=["observation.images.top"],
    num_workers=2,  # Async decode
    transforms=transform_pipeline,
)

# Iterate and time
import time
for i, batch in enumerate(loader):
    t0 = time.perf_counter()
    # Process batch (e.g., model inference)
    dt = time.perf_counter() - t0
    print(f"Batch {i}: shape={batch['observation.images.top'].shape}, dt={dt:.3f}s")
    if i >= 10:
        break
```

---

## Performance Expectations

| Backend | Decode (fps) | Latency (ms/frame) | Notes |
|---------|--------------|-------------------|-------|
| FFmpeg (CPU) | ~1,200–2,000 | 0.5–0.8ms | Portable baseline |
| NVDEC (GPU) | ~4,000–6,000 | 0.15–0.25ms | 3–5× speedup expected |
| CV-CUDA Resize | ~10,000+ fps | <0.1ms | Very fast on GPU |
| MLX Resize | ~5,000–8,000 fps | 0.1–0.2ms | macOS GPU alternative |

---

## Troubleshooting

### "ffmpeg: nvidia hwaccel not available"

**Cause:** FFmpeg built without NVDEC support.

**Fix:**
```bash
# Check ffmpeg capabilities
ffmpeg -codecs | grep -E "h264_nvdec|hevc_nvdec|av1_nvdec"

# If nothing appears, rebuild ffmpeg with NVIDIA support
# Option 1: Use conda-forge (includes NVDEC)
conda install -c conda-forge ffmpeg

# Option 2: Build from source with NVIDIA SDK
# https://developer.nvidia.com/nvidia-video-codec-sdk
```

### "ImportError: cvcuda not installed"

**Fix:**
```bash
pip install cvcuda-cu12  # CUDA 12.x
# OR
pip install cvcuda-cu11  # CUDA 11.x

# Verify
python -c "import cvcuda; print(cvcuda.__version__)"
```

### NVDEC decoder "hangs" or is slow

**Cause:** GPU memory pressure or missing optimizations.

**Debug:**
```bash
# Check GPU memory
nvidia-smi

# Run with profiling
python -c "
import pyroboframes as prf
from pyroboframes._core import Backend
print(f'Preferred backend: {Backend.preferred()}')

ds = prf.RoboFrameDataset.from_path('/path/to/dataset')
loader = ds.loader(batch_size=1, num_workers=0)  # Single-threaded for profiling
batch = next(iter(loader))
" 2>&1 | grep -E "decode|backend|error"
```

---

## Submitting Verification Results

When you've tested on GPU hardware, please share:

1. **Hardware specs:**
   ```
   GPU: [model, memory, arch]
   Driver: [version]
   CUDA: [version]
   FFmpeg: [version, with NVDEC Y/N]
   CV-CUDA: [version, if installed]
   ```

2. **Benchmark results:**
   ```
   ffmpeg baseline fps: X
   NVDEC fps: Y (speedup: Y/X)
   CV-CUDA Resize fps: Z
   ```

3. **Any issues encountered:**
   - Command to reproduce
   - Error messages
   - Environment details

---

## References

- [NVIDIA NVDEC Documentation](https://developer.nvidia.com/nvidia-video-codec-sdk)
- [CV-CUDA Documentation](https://nvidia.github.io/cvcuda/)
- [FFmpeg NVDEC Support](https://trac.ffmpeg.org/wiki/HWAccelIntro#NVIDIA)
- [LeRobot Dataset Format](https://github.com/huggingface/lerobot)
