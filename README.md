# PyRoboFrames

[![PyPI](https://img.shields.io/pypi/v/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![Python](https://img.shields.io/pypi/pyversions/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-222%20passing-brightgreen)]()

**Fast ML dataloader for robot learning — LeRobot, RLDS, HDF5, NetCDF, hardware video decode, distributed S3/GCS streaming.**

PyRoboFrames is a **foundation library** — load any robot learning dataset, accelerate video decode with hardware (VideoToolbox/NVDEC), validate data quality, and stream to NumPy/MLX/PyTorch/JAX. The heavy lifting runs in a Rust engine; Python is the ergonomic surface.

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

Optional extras (install as needed):

```bash
pip install pyroboframes h5py               # HDF5 datasets
pip install pyroboframes xarray netCDF4     # NetCDF datasets
pip install pyroboframes tensorflow-datasets  # RLDS / Open X-Embodiment
pip install pyroboframes fsspec s3fs        # S3 remote streaming
pip install pyroboframes fsspec gcsfs       # GCS remote streaming
pip install pyroboframes ray                # Ray distributed loading
```

---

## Quick Start

### Load LeRobot Datasets

```python
import pyroboframes as prf

ds = prf.RoboFrameDataset.from_path("/path/to/lerobot_dataset")

loader = ds.loader(
    batch_size=64,
    cameras=["observation.images.top"],
    output="torch",       # or "mlx", "numpy", "jax"
    num_workers=4,
    cache_size=4096,      # LRU frame cache (frames)
    episode_prefetch=True,
)

for batch in loader:
    state  = batch["observation.state"]          # [64, state_dim]
    frames = batch["observation.images.top"]     # [64, H, W, 3]
    action = batch["action"]                     # [64, action_dim]
```

### Proprioceptive-Only (No Video) for 10× Speedup

```python
loader = prf.ProprioceptiveLoader(
    dataset_path="/path/to/lerobot_dataset",
    batch_size=256,
    device="mlx",
)

for batch in loader:
    state  = batch["state"]    # [256, state_dim]
    action = batch["action"]   # [256, action_dim]
```

### Temporal Windows for Sequence Models

```python
loader = ds.loader(
    batch_size=32,
    chunk_size=16,
    delta_timestamps={"observation.state": [-0.2, -0.1, 0.0]},
    output="mlx",
)
for batch in loader:
    seq = batch["observation.state"]  # [32, 3, state_dim]
```

---

## What's New in v1.1

### Video Codec Selection — 40–50% Storage Savings

```python
# Write with HEVC (H.265) instead of the H.264 default
prf.write_lerobot_dataset(
    path="/out/dataset",
    features={"observation.state": state_arr, "action": action_arr},
    episode_lengths=[500, 500],
    video_codec="hevc",   # "h264" | "hevc" | "av1"
    video_crf=23,         # lower = better quality, larger file
)

# Standalone video encoding
prf.encode_video_frames(frames, "output.mp4", codec="av1", crf=30)
```

### Data Validation Toolkit

```python
from pyroboframes import DatasetValidator

validator = DatasetValidator(
    ds,
    check_frames=True,    # frame count vs. metadata
    check_temporal=True,  # timestamp gap detection
    check_codec=True,     # sample-decode health check
    sample_rate=0.1,      # probe 10% of episodes
)
report = validator.validate()
print(report.summary())
report.raise_if_errors()
```

### Episode-Level Caching for Repeated Epochs

```python
from pyroboframes import EpisodeCache

cache = EpisodeCache(ds, max_episodes=8)

for epoch in range(10):
    for ep_idx in range(ds.num_episodes()):
        ep = cache.get_episode(ep_idx)   # decoded once, cached after
        states = ep["observation.state"]  # [T, D]

cache.prefetch([0, 1, 2, 3])  # background pre-decode
```

### Cross-Dataset Quality Comparison

```python
from pyroboframes import EpisodeScorer, DatasetQualityProfile, CrossDatasetComparator

scorer = EpisodeScorer()
profile_a = DatasetQualityProfile.from_scores("dataset_a", scorer.score_episodes(df_a))
profile_b = DatasetQualityProfile.from_scores("dataset_b", scorer.score_episodes(df_b))

comparator = CrossDatasetComparator(reference=profile_a)
print(comparator.compare(profile_b))            # Cohen's d, percentile overlap
print(comparator.recommend_mixing_ratio(profile_b))  # curriculum mixing weight
```

### HDF5 / NetCDF / RLDS Format Support

```python
# HDF5 (ROBOMIMIC, ACT, custom) — pip install h5py
from pyroboframes import HDF5Dataset, convert_hdf5
convert_hdf5("robomimic.hdf5", "/out/lerobot")

# NetCDF (scientific/simulation datasets) — pip install xarray netCDF4
from pyroboframes import NetCDFDataset, convert_netcdf
convert_netcdf("sim_data.nc", "/out/lerobot", episode_breaks=[0, 500, 1200])

# RLDS / Open X-Embodiment — pip install tensorflow-datasets
from pyroboframes import RLDSDataset, convert_rlds
convert_rlds("fractal20220817_data", "/out/lerobot", split="train")
```

### Remote S3/GCS Streaming + Ray Distributed Loading

```python
# Stream from S3
from pyroboframes import RemoteDataset
ds = RemoteDataset.from_s3("s3://my-bucket/lerobot_dataset").open()
ds.prefetch_episodes([0, 1, 2, 3])   # background download
loader = ds.loader(batch_size=32)

# Ray distributed — pip install ray
from pyroboframes import RayDistributedLoader, shard_episodes
loader = RayDistributedLoader(
    "/path/to/dataset", num_workers=4, rank=0, world_size=4, batch_size=32
)

# Or just shard episodes yourself
my_episodes = shard_episodes(total_episodes=200, world_size=4, rank=0)
# → [0, 4, 8, …, 196]
```

---

## What's New in v1.2

### GPU-Accelerated Image Transforms

Transform frames on NVIDIA (CuPy), Apple Silicon (MLX), or CPU with automatic fallback:

```python
from pyroboframes.gpu_acceleration import GPUTransforms

transforms = GPUTransforms(device="auto")  # Picks best available: cuda → mlx → cpu

# Resize + normalize on GPU
resized = transforms.resize(frame, size=(224, 224), interpolation="bilinear")
normalized = transforms.normalize(
    resized,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Temporal Consistency for Video Stitching

Smooth stitched panoramas and reduce flickering with optical flow and temporal filtering:

```python
from pyroboframes.gpu_acceleration import OpticalFlowEstimator, TemporalFilter

# Optical flow for seam tracking
flow = OpticalFlowEstimator.estimate_lucas_kanade(frame1, frame2)

# Temporal smoothing (exponential moving average)
frames = [frame1, frame2, frame3, ...]
smoothed = TemporalFilter.apply_temporal_smoothing(frames, alpha=0.7)

# Median filtering
denoised = TemporalFilter.apply_median_filter(frames, kernel_size=3)
```

### Real-World Autonomous Driving Datasets

Load Waymo, nuScenes, and KITTI with unified interface:

```python
from pyroboframes.dataset_loaders import (
    WaymoDatasetLoader,
    nuScenesDatasetLoader,
    KITTIDatasetLoader,
)

# Waymo Open Dataset
waymo = WaymoDatasetLoader("/path/to/waymo")
image, metadata = waymo.get_frame(scene_idx=0, frame_idx=10, camera="FRONT")
print(f"Camera calibration: fx={metadata.calibration.fx}")

# nuScenes
nuscenes = nuScenesDatasetLoader("/path/to/nuscenes")
image, metadata = nuscenes.get_frame(scene_idx=0, frame_idx=10, camera="CAM_FRONT")

# KITTI
kitti = KITTIDatasetLoader("/path/to/kitti", split="training")
image, metadata = kitti.get_frame(seq_idx=0, frame_idx=10, camera=0)
```

### Occupancy Grid Mapping for 3D Perception

Convert point clouds and 3D bounding boxes to occupancy grids for path planning:

```python
from pyroboframes.occupancy_3d import OccupancyGrid, OccupancyGridConfig, LiDARProcessor

# Create occupancy grid
config = OccupancyGridConfig(size_x=100.0, size_y=100.0, resolution=0.1)
grid = OccupancyGrid(config)

# Add LiDAR point cloud
points = lidar_points[:, :2]  # [N, 2] XY coordinates
grid.add_point_cloud(points)

# Add 3D bounding boxes
bbox = {"x": 0, "y": 0, "width": 2.0, "length": 4.0, "height": 2.0}
grid.add_bounding_box(bbox)

# Morphological operations for smoothing
grid.dilate(kernel_size=3)
grid.erode(kernel_size=5)

# Get results
free_space = grid.get_free_space_mask()  # [H, W] binary mask
occupied_cells = grid.get_occupied_cells()  # List of (x, y) cells
```

### LiDAR Processing & Radar Fusion

Process 3D point clouds and fuse with radar for velocity estimates:

```python
from pyroboframes.occupancy_3d import LiDARProcessor, RadarFusionProcessor

# Filter points
points = LiDARProcessor.filter_by_distance(lidar_points, max_distance=100.0)
points = LiDARProcessor.filter_by_height(points, min_height=-1.0, max_height=3.0)

# Ground segmentation
ground, non_ground = LiDARProcessor.ground_segmentation(points, threshold=0.1)

# Clustering
clusters = LiDARProcessor.cluster_points(points, distance_threshold=0.2, min_points=5)

# Compute normals for surface analysis
normals = LiDARProcessor.compute_normals(points, k=10)

# Radar-LiDAR fusion
radar_detections = [{"x": 0, "y": 0, "z": 0, "vx": 1.0, "vy": 0, "vz": 0}]
fused = RadarFusionProcessor.fuse_radar_lidar(
    lidar_points,
    radar_detections,
    distance_threshold=1.0
)
```

---

## Full Feature Table

| Feature | Status | Notes |
|---|---|---|
| **LeRobot v3.0 loading** | ✅ | Full schema support |
| **Video decode** | ✅ | FFmpeg + VideoToolbox + NVDEC |
| **Proprioceptive loader** | ✅ | 10× speedup (no video) |
| **Temporal windows** | ✅ | Multi-timestep sequences |
| **Multi-camera batching** | ✅ | Arbitrary camera combinations |
| **Output formats** | ✅ | NumPy, MLX, PyTorch, JAX |
| **Parallel prefetch** | ✅ | num_workers for async loading |
| **Data augmentation** | ✅ | Rotate, flip, crop, color jitter |
| **Video codec selection** | ✅ | H.264 / HEVC / AV1 + CRF control |
| **Dataset validation** | ✅ | Temporal gaps, missing frames, codec health |
| **Episode caching** | ✅ | RAM-based LRU cache, background prefetch |
| **MCAP ingestion** | ✅ | JSON, protobuf, CDR support |
| **ROS 2 bag ingestion** | ✅ | .db3 native format |
| **HDF5 ingestion** | ✅ | ROBOMIMIC, ACT, custom layouts |
| **NetCDF ingestion** | ✅ | Scientific/simulation datasets |
| **RLDS / Open X-Embodiment** | ✅ | tensorflow-datasets integration |
| **Episode quality scoring** | ✅ | Diversity, sharpness, state variance |
| **Cross-dataset comparison** | ✅ | Cohen's d, percentile ranking, mixing ratio |
| **S3/GCS streaming** | ✅ | fsspec-backed remote datasets |
| **Ray distributed loading** | ✅ | Episode sharding across Ray workers |
| **Streaming ingestion** | ✅ | Kafka, MQTT real-time data |
| **Distributed loading** | ✅ | Multi-GPU synchronized sampling |

---

## Test Coverage: 222 Tests Passing ✅

```
Dataloader:       30 tests
Video decode:     25 tests
Proprioceptive:   16 tests
Augmentation:     15 tests
Temporal ops:     12 tests
Quality/scoring:  17 tests    (+7 cross-dataset)
Validation:       13 tests
Caching:           5 tests
HDF5:              7 tests
NetCDF:            7 tests
Distributed:       8 tests
Streaming:         7 tests
Codecs:            7 tests    (+3 round-trip)
GPU Acceleration: 10 tests   (NEW v1.2)
Dataset Loaders:  16 tests   (NEW v1.2)
Occupancy/3D:     32 tests   (NEW v1.2)
Other:             6 tests
```

```bash
pytest tests/ -v
```

---

## How PyRoboFrames Compares

PyRoboFrames occupies a unique position in the robot learning dataloader ecosystem:

| Dimension | PyRoboFrames | torchcodec | Robo-DM | LeRobot |
|-----------|--------------|-----------|---------|----------|
| Multi-format support (LeRobot + RLDS + HDF5 + MCAP) | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐ |
| Apple Silicon native GPU | ⭐⭐⭐⭐⭐ | ❌ | ❌ | ⭐ |
| Waymo + nuScenes + KITTI loaders | ⭐⭐⭐⭐ | ❌ | ❌ | ❌ |
| 3D occupancy grids + sensor fusion | ⭐⭐⭐⭐ | ❌ | ? | ❌ |
| Unified GPU fallback chain | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ? | ⭐ |
| Production maturity | ⭐⭐⭐ | ⭐⭐⭐⭐ | ? | ⭐⭐⭐⭐ |

**Best for:** Multi-lab robotics collaboration + autonomous driving integration + cross-platform training.

**See [COMPETITIVE_ANALYSIS.md](./docs/COMPETITIVE_ANALYSIS.md) for detailed comparison.**

---

## GPU Support

- **Apple Silicon**: VideoToolbox hardware decode, MLX zero-copy arrays
- **NVIDIA**: NVDEC hardware decode, PyTorch CUDA acceleration
- **CPU**: NumPy fallback (~10× slower than hardware)

```python
loader = ds.loader(device="auto", ...)   # auto-detect
loader = ds.loader(device="mlx", ...)   # Apple Silicon
loader = ds.loader(device="cuda", ...)  # NVIDIA
loader = ds.loader(device="cpu", ...)   # CPU
```

---

## Use Cases

- **LeRobot policy training** — Fast loading for imitation learning
- **Open X-Embodiment fine-tuning** — RLDS ingestion + LeRobot conversion
- **Large-scale cloud training** — S3/GCS streaming + Ray distribution
- **Multi-dataset curriculum** — Cross-dataset quality comparison + mixing ratios
- **Data quality auditing** — Validate integrity before long training runs
- **Legacy dataset migration** — HDF5/NetCDF → LeRobot conversion

---

## Performance

- **Video decode:** 100+ FPS (hardware-accelerated on macOS/CUDA)
- **Dataloader throughput:** 50–100 images/sec (PyTorch, Mac M3)
- **Proprioceptive loader:** 1,000+ batch/sec (no video decode)
- **Storage savings:** 40–50% with HEVC vs H.264 at equivalent quality

---

## Architecture

```
PyRoboFrames (Rust core + Python surface)

Input: LeRobot / HDF5 / NetCDF / RLDS / MCAP / ROS2 / S3 / GCS
   ↓
Format Converters  →  LeRobot v3.0 (Parquet + MP4)
   ↓
Rust Decoder (VideoToolbox / NVDEC / FFmpeg)
   ↓
RoboFrameDataset (episode index, frame manifest)
   ↓
Loader (temporal windows, augmentation, caching, batching)
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
├── EpisodeCache          # RAM-based episode LRU cache
├── DatasetValidator      # Deep data quality checks
├── hdf5                  # HDF5 reader + converter
├── netcdf                # NetCDF reader + converter
├── rlds                  # RLDS / Open X-Embodiment reader
├── distributed           # RemoteDataset, RayDistributedLoader, shard_episodes
├── quality               # EpisodeScorer, CrossDatasetComparator
├── backend/              # Device abstractions (MLX, PyTorch, JAX)
├── transforms/           # Augmentation pipelines
└── [streaming, sensor_fusion, depth_io, ...]
```

---

## Related Projects

- **[LeRobot](https://github.com/huggingface/lerobot)** — Robot learning datasets
- **[PyRoboVision](https://github.com/Mullassery/PyRoboVision)** — Autonomous driving perception + foundation models
- **[MLX](https://github.com/ml-explore/mlx)** — Apple Silicon ML framework
- **[Open X-Embodiment](https://robotics-transformer-x.github.io/)** — Cross-embodiment robotics datasets

---

## Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Design and implementation
- [CONTRIBUTING.md](./CONTRIBUTING.md) — How to contribute
- [CHANGELOG.md](./CHANGELOG.md) — Version history
- [IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md) — Planned features

---

## Community

- **GitHub Issues** — [Ask questions, report bugs](https://github.com/Mullassery/PyRoboFrames/issues)
- **GitHub Discussions** — [Share ideas and best practices](https://github.com/Mullassery/PyRoboFrames/discussions)
- **Code of Conduct** — [Be respectful and constructive](./CODE_OF_CONDUCT.md)

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

## 🔒 Security & Error Handling

PyRoboFrames includes:

- **Secure Credential Handling**: IAM roles recommended over long-term credentials (see `DEPLOYMENT_SECURITY.md`)
- **Path Validation**: Prevents path traversal for S3/GCS access
- **Hardware Warnings**: Graceful degradation with fallback from GPU video decode
- **Detailed Error Messages**: See `python/pyroboframes/error_messages.py` for dataset recovery steps

### Security Roadmap

- ✅ v1.1.0: Path traversal protection, hardware warnings
- ✅ v1.0.2: Dependencies pinned
- 🔄 v1.2.0: Zero-copy MLX arrays, temporal window fixes
- 🔄 v1.3.0: HDF5 and distributed loading support
- 📋 v2.0.0: Data augmentation and offline RL integration

Full roadmap: [ROADMAP_HONEST.md](ROADMAP_HONEST.md)

## 🆕 What's New in v1.3.0 (Q4 2026)

### Multi-Format Dataset Support 📦
Load datasets in multiple formats seamlessly:

```python
from pyroboframes import load_dataset, DatasetFormat

# Auto-detect format
loader = load_dataset('/path/to/dataset')

# Or hint the format explicitly
loader_rlds = load_dataset('/path/to/rlds_data', format_hint='RLDS')
loader_hdf5 = load_dataset('/path/to/hdf5_data', format_hint='HDF5')

# Load episodes and frames
episode = loader.load_episode(0)
frame = loader.load_frame(0, 42)
```

**Supported Formats:**

| Format | Source | Best For | Stream | Random |
|--------|--------|----------|--------|--------|
| LeRobot | HuggingFace | Modern datasets | ✅ | ❌ |
| RLDS | OpenX Embodiment | Multi-lab datasets | ✅ | ❌ |
| HDF5 | Traditional ML | Large hierarchical | ❌ | ✅ |
| Custom | Plugin system | Your format | 🔧 | 🔧 |

**Why This Matters:**
- Robot learning has 5+ competing dataset formats
- Teams locked into single format couldn't collaborate
- Multi-format support opens ecosystem collaboration
- Plugin system enables custom formats without forking

See `pyroboframes/_format_registry.py` for implementation.
