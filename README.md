# PyRoboFrames

[![PyPI](https://img.shields.io/pypi/v/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![Python](https://img.shields.io/pypi/pyversions/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-115%20passing-brightgreen)]()

**Fast ML dataloader for robot learning on Apple Silicon, NVIDIA, and CPU.**

PyRoboFrames is a **foundation library**—load LeRobot datasets, accelerate video decode with hardware (VideoToolbox/NVDEC), and output to NumPy/MLX/PyTorch/JAX.

**For autonomous driving perception and foundation models, see [PyRoboVision](https://github.com/Mullassery/PyRoboVision).**

---

## Installation

```bash
pip install pyroboframes
```

Or with uv:
```bash
uv add pyroboframes
```

**Requires:** Python ≥ 3.10  
**Prebuilt wheels:** macOS (Apple Silicon), Linux (x86_64)  
**From source:** Rust 1.78+ required

---

## Quick Start

### Load LeRobot Datasets

```python
import pyroboframes as prf

# Open a LeRobot dataset
ds = prf.RoboFrameDataset.from_path("/path/to/lerobot_dataset")
print(ds)  # RoboFrameDataset(episodes=…, frames=…, cameras=[…])

# Create a dataloader with hardware-accelerated video decode
loader = ds.loader(
    batch_size=64,
    cameras=["observation.images.top"],
    output="torch",  # or "mlx", "numpy", "jax"
    num_workers=4,  # parallel video decode
)

# Train
for batch in loader:
    state = batch["observation.state"]    # [64, state_dim]
    frames = batch["observation.images.top"]  # [64, H, W, 3]
    action = batch["action"]               # [64, action_dim]
    # your training step...
```

### Proprioceptive-Only (No Video) for 10x Speedup

```python
# For policies that only use state/action (no camera)
loader = prf.ProprioceptiveLoader(
    dataset_path="/path/to/lerobot_dataset",
    batch_size=256,
    device="mlx",  # or "cuda", "cpu"
)

for batch in loader:
    state = batch["state"]    # [256, state_dim]
    action = batch["action"]  # [256, action_dim]
    # ~10x faster than loading video frames
```

### Temporal Windows for Sequence Models

```python
# Multi-timestep windows for RNN/Transformer policies
loader = ds.loader(
    batch_size=32,
    chunk_size=16,  # 16-frame sequences
    delta_timestamps={"observation.state": [-0.2, -0.1, 0.0]},
    output="mlx",
)
for batch in loader:
    seq = batch["observation.state"]  # [32, 3, state_dim]
    # Perfect for temporal policies
```

---

## Features

| Feature | Status | Notes |
|---------|--------|-------|
| **LeRobot v3.0 loading** | ✅ | Full schema support |
| **Video frame decoding** | ✅ | FFmpeg + hardware acceleration |
| **Proprioceptive-only loader** | ✅ | 10x speedup (no video) |
| **Temporal windows** | ✅ | Multi-timestep sequences |
| **Multi-camera batching** | ✅ | Arbitrary camera combinations |
| **Output formats** | ✅ | NumPy, MLX, PyTorch, JAX |
| **Parallel prefetch** | ✅ | num_workers for async loading |
| **Data augmentation** | ✅ | Rotate, flip, crop, color jitter |
| **Dataset validation** | ✅ | Frame integrity checks |
| **MCAP ingestion** | ✅ | JSON, protobuf, CDR support |
| **ROS 2 bag ingestion** | ✅ | .db3 native format |
| **GPU decode** | ✅ | VideoToolbox (macOS), NVDEC (CUDA) |
| **Distributed loading** | ✅ | Multi-GPU synchronized sampling |
| **Episode quality scoring** | ✅ | Diversity, sharpness, state variance |
| **Streaming** | ✅ | Kafka, MQTT real-time data |

---

## Test Coverage: 115 Tests Passing ✅

```
Dataloader:     30 tests
Video decode:   25 tests
Proprioceptive: 16 tests
Augmentation:   15 tests
Temporal ops:   12 tests
Quality/scoring: 10 tests
Streaming:       7 tests
```

Run all tests:
```bash
pytest tests/ -v
```

---

## GPU Support

- **Apple Silicon**: VideoToolbox hardware decode, MLX zero-copy arrays
- **NVIDIA**: NVDEC hardware decode, PyTorch CUDA acceleration
- **CPU**: NumPy fallback (works everywhere, ~10× slower)

```python
# Auto-detect device
loader = ds.loader(device="auto", ...)

# Or explicit
loader = ds.loader(device="mlx", ...)  # Apple Silicon
loader = ds.loader(device="cuda", ...)  # NVIDIA
loader = ds.loader(device="cpu", ...)   # CPU
```

---

## Use Cases

- **LeRobot policy training** — Fast loading for imitation learning
- **Robot manipulation** — Multi-camera, temporal windows, state/action
- **Robotdog navigation** — Proprioceptive-only loaders for low-latency inference
- **Mobile base control** — Egocentric perception + proprioceptive fusion

---

## Performance

- **Video decode:** 100+ FPS (hardware-accelerated on macOS/CUDA)
- **Dataloader throughput:** 50-100 images/sec (PyTorch, Mac M3)
- **Proprioceptive loader:** 1,000+ batch/sec (no video decode)

---

## Architecture

```
PyRoboFrames (Rust core + Python surface)

Input: LeRobot dataset (Parquet + MP4 video)
   ↓
Rust Decoder (VideoToolbox / NVDEC / FFmpeg)
   ↓
RoboFrameDataset (episode index, frame manifest)
   ↓
Loader (temporal windows, augmentation, batching)
   ↓
Output: NumPy / MLX / PyTorch / JAX
   ↓
Your training loop
```

### Module Organization

```
pyroboframes/
├── RoboFrameDataset      # Load LeRobot datasets
├── ProprioceptiveLoader  # State/action only (no video)
├── DataLoader            # Flexible batching + augmentation
├── backend/              # Device abstractions (MLX, PyTorch, JAX)
├── transforms/           # Augmentation pipelines
├── depth_io/             # Depth camera support
├── sensor_fusion/        # Multi-sensor time-alignment
└── [streaming, quality, distributed, ...]
```

---

## Related Projects

- **[LeRobot](https://github.com/huggingface/lerobot)** — Robot learning datasets
- **[PyRoboVision](https://github.com/Mullassery/PyRoboVision)** — Autonomous driving perception + foundation models
- **[MLX](https://github.com/ml-explore/mlx)** — Apple Silicon ML framework
- **[PyTorch](https://pytorch.org/)** — Deep learning framework

---

## Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Design and implementation
- [CONTRIBUTING.md](./CONTRIBUTING.md) — How to contribute
- [CHANGELOG.md](./CHANGELOG.md) — Version history

---

## License

[MIT](./LICENSE) © Georgi Mammen Mullassery

---

## Citation

```bibtex
@software{mullassery2025pyroboframes,
  title={PyRoboFrames: Fast ML dataloader for robot learning},
  author={Mullassery, Georgi},
  url={https://github.com/Mullassery/PyRoboFrames},
  year={2025}
}
```
