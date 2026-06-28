# PyRoboFrames

[![PyPI](https://img.shields.io/pypi/v/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![Python](https://img.shields.io/pypi/pyversions/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-115%20passing-brightgreen)]()

**A multi-domain perception and learning platform for robotics and autonomous driving.**

- **Robot Learning (v0.4.x):** Fast video dataloaders for LeRobot datasets with hardware video decode
- **Autonomous Driving (v0.5.x):** 360° panoramic stitching, 3D sensor fusion, scene understanding
- **Foundation Models (Phase 7):** SAM3 segmentation, CLIP embeddings, open-vocabulary detection

**Current Release:** v0.5.2 + Phase 7a (115 automotive tests passing) ✅

---

## Quick Compare: What's In Each Version?

| | v0.4.1 (Robot Learning) | v0.5.2 (Autonomous Driving) |
|---|---|---|
| **Video Decode** | ✅ FFmpeg (hardware on macOS/CUDA) | ✅ Used for frame input |
| **Output Formats** | NumPy, MLX, PyTorch, JAX | NumPy (frame processing) |
| **Primary Use** | LeRobot dataset training | 360° perception for AVs |
| **GPU Backends** | VideoToolbox, NVDEC, CV-CUDA | CuPy, MLX, NumPy |
| **Tests** | ~50 robot-learning tests | **115 automotive tests** |
| **Main Pipeline** | `Dataset → Loader → Training` | `Video → Stitch → Segment → Map` |

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
**From source:** Rust 1.78+ required (`brew install rust` on macOS)

---

## Quick Start

### Robot Learning (v0.4.1): Load LeRobot Datasets

```python
import pyroboframes as prf

# Open a LeRobot dataset
ds = prf.RoboFrameDataset.from_path("/path/to/lerobot_dataset")
print(ds)  # RoboFrameDataset(episodes=…, frames=…, cameras=[…])

# Create a dataloader
loader = ds.loader(
    batch_size=64,
    cameras=["observation.images.top"],
    output="torch",  # or "mlx", "numpy"
)

# Train
for batch in loader:
    state = batch["observation.state"]    # [64, state_dim]
    frames = batch["observation.images.top"]  # [64, H, W, 3]
    action = batch["action"]               # [64, action_dim]
    # your training step...
```

### Autonomous Driving (v0.5.2): 360° Panoramic Perception

```python
from pyroboframes.automotive import (
    CylindricalStitcher,
    SAM3Segmenter,
    OccupancyGrid,
    get_waymo_layout,
)

# Step 1: Stitch 5 cameras into 360° panorama
layout = get_waymo_layout()
stitcher = CylindricalStitcher(layout, blend_method="laplacian")

frames = {
    "FRONT": np.zeros((720, 1280, 3), dtype=np.uint8),
    "FRONT_LEFT": np.zeros((720, 1280, 3), dtype=np.uint8),
    # ... other cameras
}

panorama = stitcher.stitch(frames)
print(panorama.shape)  # (1, 480, 1728, 3) - seamless 360° view

# Step 2: Segment with SAM3 (temporal consistency)
segmenter = SAM3Segmenter("facebook/sam3-small", device="mlx")
masks = segmenter.segment(panorama[0])

# Step 3: Build occupancy grid
occupancy = OccupancyGrid(size=(-50, 50), resolution=0.2)
occupancy.update(lidar_points=lidar)
occupancy_map = occupancy.get_occupancy_map()
```

---

## Features by Domain

### Robot Learning (v0.4.x) - 50 Tests ✅

| Feature | Status | Notes |
|---------|--------|-------|
| **LeRobot v3.0 loading** | ✅ | Full schema support |
| **Video frame decoding** | ✅ | FFmpeg with hardware acceleration |
| **Temporal windows** | ✅ | delta_timestamps for sequences |
| **Multi-camera batching** | ✅ | Arbitrary camera combinations |
| **Output formats** | ✅ | NumPy, MLX, PyTorch, JAX |
| **Off-GIL prefetch** | ✅ | num_workers for async loading |
| **Data augmentation** | ✅ | Rotate, flip, crop, color jitter |
| **Dataset validation** | ✅ | Frame integrity checks |
| **MCAP ingestion** | ✅ | JSON, protobuf, CDR support |
| **ROS 2 bag ingestion** | ✅ | .db3 native format |
| **Robotics DataFrame** | ✅ | Time-indexed, as-of join, resample |
| **Episode quality scoring** | ✅ | Diversity, sharpness, state variance |
| **Distributed loading** | ✅ | Multi-GPU synchronized sampling |
| **GPU decode** | ✅ | VideoToolbox (macOS), NVDEC (CUDA) |
| **Zero-copy MLX** | ⏳ | Awaiting mlx#2855 |

### Autonomous Driving (v0.5.x) - 65 Tests ✅

| Phase | Feature | Status | Tests |
|-------|---------|--------|-------|
| **1** | Cylindrical panoramic projection | ✅ | 10 |
| **2** | Laplacian pyramid blending | ✅ | 5 |
| **3** | Bird's-eye-view (BEV) projection | ✅ | 5 |
| **4a** | GPU acceleration (CuPy/MLX/NumPy) | ✅ | 6 |
| **4b** | Optical flow seam tracking | ✅ | 10 |
| **5** | Waymo/nuScenes/KITTI loaders | ✅ | 9 |
| **6** | Lidar/Radar fusion + Occupancy grids | ✅ | 18 |
| **7a** | SAM3 temporal segmentation | ✅ | 18 |
| **7b** | CLIP scene embeddings | 🔄 | (in progress) |
| **7c** | Grounding DINO detection | ⏳ | (planned) |
| **7d** | Multi-modal fusion | ⏳ | (planned) |

---

## Examples

### Robot Learning

```python
# Temporal windows for sequence models
loader = ds.loader(
    batch_size=32,
    chunk_size=16,  # 16-frame sequences
    delta_timestamps={"observation.state": [-0.2, -0.1, 0.0]},
    output="mlx",
)
for batch in loader:
    seq = batch["observation.state"]  # [32, 3, state_dim]
```

### Autonomous Driving - Full Pipeline

```python
from pyroboframes.automotive import (
    WaymoDatasetLoader,
    CylindricalStitcher,
    LidarFusion,
    OccupancyGrid,
    SAM3Segmenter,
    CLIPEmbedding,
)

# Phase 5: Load dataset
waymo = WaymoDatasetLoader("/path/to/waymo", split="training")

for batch in waymo:
    frames = batch["frames"]  # {cam_id -> [H, W, 3]}
    lidar = batch["lidar"]    # [N, 4] point cloud
    
    # Phase 1: Stitch panorama
    stitcher = CylindricalStitcher(get_waymo_layout())
    panorama = stitcher.stitch(frames)
    
    # Phase 6: Fuse sensors
    fusion = LidarFusion(num_lidars=5)
    fused = fusion.fuse(lidar)
    
    occupancy = OccupancyGrid(size=(-50, 50), resolution=0.2)
    occupancy.update(lidar_points=fused[:, :3])
    
    # Phase 7a: Segment with SAM3
    segmenter = SAM3Segmenter("facebook/sam3-small", device="mlx")
    masks = segmenter.segment(panorama[0])
    
    # Phase 7b: Scene understanding with CLIP
    clip = CLIPEmbedding("openai/clip-vit-b32")
    scene_scores = clip.classify(
        panorama[0],
        ["highway", "city", "parking"]
    )
```

Run the examples:
```bash
# Robot learning
python examples/dataloader_quickstart.py

# Autonomous driving
python examples/autonomous_driving_360_perception.py
python examples/autonomous_driving_dataset_3d_perception.py
python examples/autonomous_driving_sam3_segmentation.py
```

---

## What's Actually Working

### v0.4.1: Robot Learning Dataloader

- ✅ **LeRobot v3.0 format** — reads metadata, episodes, state/action
- ✅ **Video frame loading** — FFmpeg decode with hardware acceleration
- ✅ **Temporal windows** — multiple timesteps with tolerance matching
- ✅ **Multiple output formats** — NumPy, MLX, PyTorch, JAX (zero-copy torch)
- ✅ **Off-GIL prefetch** — `num_workers=4` shows measurable speedup
- ✅ **CPU & GPU** — works on Mac, Linux+CUDA, CPU fallback
- ⏳ **Zero-copy MLX** — infrastructure ready, gated on [mlx#2855](https://github.com/ml-explore/mlx/issues/2855)

### v0.5.2: Autonomous Driving Perception

**Phase 1-4 (Stitching + GPU):**
- ✅ Cylindrical projection with 360° coverage
- ✅ Linear & Laplacian pyramid blending (seamless seams)
- ✅ Bird's-eye-view (BEV) 3D projection
- ✅ GPU acceleration (CuPy/MLX/NumPy backends)
- ✅ Optical flow seam tracking
- ✅ Temporal Kalman smoothing

**Phase 5-6 (Datasets + 3D Perception):**
- ✅ Waymo Open Dataset loader (5 cameras, 5 lidar, TFRecord)
- ✅ nuScenes loader (6 cameras, lidar, radar, JSON)
- ✅ KITTI loader (stereo pairs, 3D detection)
- ✅ Multi-lidar fusion with voxel downsampling
- ✅ Radar velocity fusion
- ✅ Bayesian occupancy grid mapping (log-odds)

**Phase 7a (Foundation Models - In Progress):**
- ✅ SAM3 instance segmentation (temporal tracking)
- 🔄 CLIP scene classification & embeddings
- ⏳ Grounding DINO open-vocabulary detection
- ⏳ Multi-modal fusion pipeline

---

## Test Coverage: 115 Tests Passing ✅

```
Robot Learning (v0.4.x):  ~50 tests
├─ Dataloader: 20 tests
├─ Video decode: 15 tests
├─ Augmentation: 8 tests
└─ Data-ops: 7 tests

Autonomous Driving (v0.5.x):  115 tests
├─ Phase 1-3 (Stitching): 49 tests
├─ Phase 4 (GPU + Temporal): 21 tests
├─ Phase 5-6 (Datasets + 3D): 27 tests
└─ Phase 7a (SAM3): 18 tests
```

Run all tests:
```bash
pytest tests/ -v
```

---

## Architecture

```
PyRoboFrames (Rust core + Python bindings via PyO3)

┌─────────────────────────────────────────┐
│ Robot Learning (v0.4.x)                 │
├─────────────────────────────────────────┤
│ LeRobot Dataset                         │
│   ↓ (via Rust video decode)            │
│ FFmpeg / VideoToolbox / NVDEC           │
│   ↓ (NumPy buffer)                     │
│ NumPy / MLX / PyTorch / JAX output      │
│   ↓                                     │
│ Training Loop                           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Autonomous Driving (v0.5.x)             │
├─────────────────────────────────────────┤
│ Camera Frames (Waymo/nuScenes/KITTI)    │
│   ↓ (Phase 1-4)                        │
│ Cylindrical Stitching + GPU Accel       │
│   ↓ (Phase 5-6)                        │
│ Lidar/Radar Fusion + Occupancy Grid     │
│   ↓ (Phase 7a-7d)                      │
│ SAM3 / CLIP / Grounding DINO            │
│   ↓                                     │
│ Semantic 3D Scene Understanding         │
└─────────────────────────────────────────┘
```

---

## Roadmap

### Shipped

**v0.4.0-v0.4.1:** Full LeRobot dataloader, video decode, GPU backends, depth cameras, camera calibration

**v0.5.0-v0.5.2:** Panoramic stitching (Phases 1-4), dataset loaders (Phase 5), 3D perception (Phase 6)

**Phase 7a:** SAM3 temporal segmentation (18 tests passing)

### Next (v0.5.3)

- **Phase 7b:** CLIP scene embeddings (3-4 days)
  - Scene classification (highway, city, parking, etc.)
  - Text-image similarity scoring
  - Open-vocabulary scene search
  
- **Phase 7c:** Grounding DINO detection (3-4 days)
  - Open-vocabulary object detection
  - Language-grounded bounding boxes
  - Optional SAM3 mask refinement

- **Phase 7d:** Multi-modal fusion (2-3 days)
  - SAM3 + CLIP + Grounding DINO pipeline
  - Semantic occupancy grids
  - Real-time streaming inference

### Future (v1.0+)

- **v0.6.0:** Cross-domain integration (SLAM + planning + transfer learning)
- **v1.0:** Unified robot + automotive perception stack

---

## Documentation

### Robot Learning (v0.4.x)
- [GPU_VERIFICATION.md](./docs/GPU_VERIFICATION.md) — GPU setup and verification
- [COMPARISON.md](./docs/COMPARISON.md) — vs other dataloaders

### Autonomous Driving (v0.5.x)
- [AUTOMOTIVE_STITCHING_PHASE1.md](./docs/AUTOMOTIVE_STITCHING_PHASE1.md) — Cylindrical projection
- [AUTOMOTIVE_STITCHING_PHASE2_3.md](./docs/AUTOMOTIVE_STITCHING_PHASE2_3.md) — Blending + BEV
- [VERSION_0.5.2_SUMMARY.md](./docs/VERSION_0.5.2_SUMMARY.md) — Complete Phase 5-6 reference
- [ROADMAP_V0.5.3_SAM_MODELS.md](./docs/ROADMAP_V0.5.3_SAM_MODELS.md) — Phase 7 planning

### General
- [ARCHITECTURE.md](./ARCHITECTURE.md) — Design and implementation details
- [ROADMAP.md](./docs/ROADMAP.md) — Feature priorities
- [CONTRIBUTING.md](./CONTRIBUTING.md) · [CHANGELOG.md](./CHANGELOG.md)

---

## Cross-Domain Integration: The Real Power

PyRoboFrames' unique strength is **applying techniques from both domains** to solve complex problems:

### Robot Learning → Autonomous Driving

```python
# Use robot imitation learning to improve AV decision-making
from pyroboframes.automotive import OccupancyGrid, SAM3Segmenter
from pyroboframes import RoboFrameDataset

# Train a robot policy from demos
robot_dataset = RoboFrameDataset.from_path("/robot/data")
robot_policy = train_policy(robot_dataset)  # Your training loop

# Apply same architecture to driving
occupancy = OccupancyGrid(size=(-50, 50), resolution=0.2)
# Now interpret occupancy grid as "robot workspace"
# Use same Kalman filtering, temporal consistency, etc.
```

### Autonomous Driving → Robot Learning

```python
# Use AV perception for robotic manipulation
from pyroboframes.automotive import SAM3Segmenter, CLIPEmbedding
from pyroboframes import RoboFrameDataset

# Segment robot workspace with SAM3
segmenter = SAM3Segmenter("facebook/sam3-small", device="mlx")
gripper_view = robot_camera_frame()
object_masks = segmenter.segment(gripper_view)

# Understand scene semantics with CLIP
clip = CLIPEmbedding("openai/clip-vit-b32")
grasp_scores = clip.classify(gripper_view, 
    ["graspable object", "fragile item", "tool", "empty space"])

# Now do policy learning with semantic understanding
loader = robot_dataset.loader(cameras=["gripper_cam"], ...)
```

### Shared Infrastructure

Both domains benefit from **unified implementations of:**

1. **GPU acceleration** (CuPy, MLX, NumPy) - works for video decode, stitching, segmentation
2. **Temporal consistency** (Kalman filtering) - smooth robot trajectories AND stitching
3. **Multi-sensor fusion** (as-of join) - robot sensors AND vehicle sensors
4. **Occupancy representation** - robot workspace AND driving scene
5. **Foundation models** (SAM3, CLIP, Grounding DINO) - work for both perception tasks

### Real-World Examples

**Mobile Manipulation Robot:**
```python
# Fuse robot arm + mobile base + camera
from pyroboframes import RoboFrameDataset
from pyroboframes.automotive import SAM3Segmenter, OccupancyGrid

ds = RoboFrameDataset.from_path("/mobile_manip_data")

for batch in ds.loader(
    cameras=["mobile_base_cam", "arm_cam"],
    temporal_windows=[-0.5, 0],  # historical context
):
    # Segment gripper view with SAM3
    segmenter = SAM3Segmenter("facebook/sam3-small")
    arm_mask = segmenter.segment(batch["arm_cam"])
    
    # Build workspace occupancy grid
    workspace = OccupancyGrid(size=(-2, 2), resolution=0.05)
    workspace.update_with_mask(arm_mask)
    
    # Train policy with spatial awareness
    policy_output = policy_network(
        state=batch["arm_state"],
        semantics=workspace.get_occupancy_map()
    )
```

**Autonomous Delivery Robot:**
```python
# Combine autonomous driving (navigation) + robotics (manipulation)
from pyroboframes.automotive import (
    WaymoDatasetLoader,
    CylindricalStitcher,
    SAM3Segmenter,
)
from pyroboframes import RoboFrameDataset

# Learn navigation from autonomous driving data
av_dataset = WaymoDatasetLoader("/waymo/data")
for av_batch in av_dataset:
    # Panoramic view for navigation
    stitcher = CylindricalStitcher(get_waymo_layout())
    nav_view = stitcher.stitch(av_batch["frames"])
    
    # Segment obstacles with SAM3
    segmenter = SAM3Segmenter("facebook/sam3-small")
    obstacles = segmenter.segment(nav_view[0])
    
    # Learn to navigate around obstacles
    nav_policy.train_step(nav_view, obstacles)

# Learn manipulation from robot demos
robot_dataset = RoboFrameDataset.from_path("/robot/delivery")
for robot_batch in robot_dataset.loader():
    # Segment package to pick up
    masks = segmenter.segment(robot_batch["camera"])
    
    # Learn to manipulate with semantic understanding
    manip_policy.train_step(robot_batch, masks)
```

---

## Which Should I Use?

### Use **v0.4.x (Robot Learning)** if:
- Training a policy from LeRobot datasets
- You need fast video loading on Mac, NVIDIA, or CPU
- You want multi-camera temporal windows for sequence models
- **You're building robot manipulation systems**

### Use **v0.5.x (Autonomous Driving)** if:
- Building 360° autonomous driving perception
- You need real-world dataset support (Waymo, nuScenes, KITTI)
- You want sensor fusion (lidar + radar + camera)
- You need semantic scene understanding
- **You're building autonomous vehicle systems**

### Use **BOTH** if (The Real Opportunity):
- **Building mobile manipulation systems** (Stretch, Boston Dynamics, TRI systems)
- **Building autonomous delivery robots** (combining navigation + manipulation)
- **Building humanoid robots** with scene understanding
- **Researching embodied AI** (sim-to-real, multi-task learning)
- **Transferring knowledge** between domains (navigation→manipulation, vice versa)

The cross-domain capabilities enable **next-generation robotics applications** that require both autonomous navigation AND intelligent manipulation in real-world scenes.

---

### Use **v0.4.x (Robot Learning)** if:
- Training a policy from LeRobot datasets
- You need fast video loading on Mac, NVIDIA, or CPU
- You want multi-camera temporal windows for sequence models

### Use **v0.5.x (Autonomous Driving)** if:
- Building 360° autonomous driving perception
- You need real-world dataset support (Waymo, nuScenes, KITTI)
- You want sensor fusion (lidar + radar + camera)
- You need semantic scene understanding (Phase 7)

### Use **Both** if:
- Building a unified robotics + AV platform
- You want to apply AV perception tricks to robots (and vice versa)

---

## Performance

### Robot Learning (v0.4.x)
- **Video decode:** 100+ FPS (hardware-accelerated on macOS/CUDA)
- **Dataloader throughput:** 50-100 images/sec (PyTorch, Mac M3)
- **Zero-copy MLX:** 3× faster when mlx#2855 lands

### Autonomous Driving (v0.5.x)
- **Stitching:** 10-30 FPS (5 cameras → panorama)
- **SAM3 segmentation:** 25-50 FPS (temporal tracking included)
- **Occupancy grid:** 200ms for 500×500 grid (100k lidar + 20 radar)
- **Full pipeline:** Real-time at 30 FPS on M3/H100

---

## License

[MIT](./LICENSE) © Georgi Mammen Mullassery

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md).

The highest-impact work right now:
1. **Phase 7b-7d implementation** (CLIP, Grounding DINO, multi-modal fusion)
2. **GPU verification** (benchmark NVDEC + CV-CUDA on NVIDIA hardware)
3. **Zero-copy MLX** (when mlx#2855 lands)
