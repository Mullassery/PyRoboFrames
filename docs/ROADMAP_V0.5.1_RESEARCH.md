# v0.5.1+ Roadmap: GPU Acceleration, Temporal Consistency & Real-World Integration

Research and planning for PyRoboFrames post-v0.5.0 releases.

## Executive Summary

**v0.5.0** delivered a complete 3-phase automotive perception pipeline (panoramic stitching + BEV projection). **v0.5.1–v0.5.2** focus on:

1. **Production readiness**: GPU acceleration (NVIDIA + Apple Silicon)
2. **Quality improvement**: Temporal consistency (optical flow, multi-frame alignment)
3. **Real-world deployment**: Waymo/nuScenes dataset integration
4. **Advanced perception**: Lidar/radar fusion, occupancy mapping

**Timeline**: v0.5.1 (4-6 weeks), v0.5.2 (8-10 weeks)

---

## v0.5.1: GPU Acceleration + Temporal Consistency (4-6 weeks)

### Phase 4a: GPU Acceleration for Panoramic Stitching

#### CuPy Backend (NVIDIA CUDA)

**Objective**: 100+ FPS panoramic stitching on RTX 5090 / H100

**Implementation**:
```python
from pyroboframes.automotive import CylindricalStitcher

stitcher = CylindricalStitcher(
    layout=get_waymo_layout(),
    blend_method="laplacian",
    device="cuda",  # Auto-detects NVIDIA GPU
)

# CuPy backend automatically used
panorama = stitcher.stitch(frames)  # [batch, 480, 1728, 3]
# GPU latency: ~10ms per batch
```

**Architecture**:
- **Gaussian pyramid**: CuPy `scipy.ndimage` GPU equivalent
- **Laplacian decomposition**: In-GPU pyramid math
- **Seam finding**: CuPy sparse array for DP
- **Blending**: CUDA kernel fusion (optional: custom cutlass kernels)
- **Fallback**: Torch tensors if CuPy unavailable

**Effort**: `M` (3-4 days) · **Testable**: Yes (needs NVIDIA GPU) · **Value**: High (100+ FPS)

**Benchmarks** (target):
| Device | Phase 1 (linear) | Phase 2 (Laplacian) | Phase 3 (BEV) |
|--------|------------------|-------------------|---------------|
| M3 CPU | 10 FPS | 5 FPS | negligible |
| **RTX 5090** | **100 FPS** | **50 FPS** | **negligible** |
| A100 | 150 FPS | 75 FPS | negligible |

#### MLX Backend (Apple Silicon GPU)

**Objective**: 50+ FPS on Mac Studio / M3 Max GPU

**Implementation**:
```python
stitcher = CylindricalStitcher(
    layout=get_waymo_layout(),
    device="mps",  # or "mlx" for native MLX
)
# MLX GPU backend automatically selected
```

**Architecture**:
- **Memory-mapped arrays**: MLX unified memory
- **Gaussian pyramid**: MLX `nn.layer_norm` + resize
- **Laplacian**: Efficient MLX operations
- **Fallback**: NumPy if MLX unavailable

**Effort**: `M` (3-4 days) · **Testable**: Yes (on this Mac) · **Value**: Medium (2-3× speedup)

#### Performance Analysis

```python
# Benchmark script
python benches/automotive_gpu_benchmark.py \
  --device cuda \
  --blend-method laplacian \
  --frames 100 \
  --batch-size 8
```

**Metrics to track**:
- Throughput (FPS)
- Peak GPU memory
- Transfer latency (H2D + D2H)
- Thermal throttling (on laptops)

### Phase 4b: Temporal Consistency (Optical Flow)

**Objective**: Reduce flickering and ghosting in video sequences

**Current Problem**:
- Each frame stitched independently
- No temporal coherence → flickering at seams
- Fast-moving objects create ghosts

**Solution: Optical Flow Seam Tracking**

```python
from pyroboframes.automotive import TemporalStitcher

stitcher = TemporalStitcher(
    layout=get_waymo_layout(),
    use_optical_flow=True,
    flow_model="raft",  # or "liteflownet"
)

# Process frame sequence
panoramas = []
for frame_t, frame_t1 in zip(frames[:-1], frames[1:]):
    # Compute optical flow between frames
    flow = stitcher.compute_flow(frame_t, frame_t1)
    
    # Track seams across frames
    seam_t1 = stitcher.track_seam(seam_t, flow)
    
    # Stitch with temporally-coherent seams
    pan = stitcher.stitch([frame_t, frame_t1], seam_t1)
    panoramas.append(pan)
```

**Implementation Details**:

1. **Optical Flow Engine** (choose one):
   - `raft`: Accurate, slower (~50ms on M3) — good for offline
   - `liteflownet`: Fast (~10ms) — good for real-time
   - `farneback` (OpenCV): Fastest (~5ms) — good for baselines

2. **Seam Tracking**:
   - Seam position is set of (y, x) pairs
   - Flow vector at each seam location → next seam position
   - Smooth motion with temporal filtering (Kalman filter)

3. **Temporal Blending**:
   ```python
   # Blend consecutive panoramas to reduce flicker
   panorama_smooth = alpha * pan_t + (1 - alpha) * pan_t1
   ```

**Architecture**:
```python
class TemporalStitcher:
    def __init__(self, layout, use_optical_flow=True):
        self.flow_model = OpticalFlow("raft")  # Load once
        self.seam_history = deque(maxlen=5)     # Smooth seams
    
    def stitch_sequence(self, frames):
        for t, frame_t in enumerate(frames):
            if t == 0:
                flow = None
            else:
                flow = self.flow_model(frames[t-1], frame_t)
            
            seam_t = self.track_seam(flow)
            pan_t = self.stitch(frame_t, seam_t)
            
            yield pan_t
```

**Effort**: `M` (4-5 days) · **Testable**: Yes (optical flow libraries available) · **Value**: High (qualitative improvement)

**Benchmarks**:
- RAFT optical flow: 50ms per frame pair (M3) → can run async
- Seam tracking: 5ms
- Temporal blending: 10ms
- **Total**: ~65ms overhead → still real-time at 15 FPS

---

## v0.5.2: Real-World Datasets + Advanced Perception (8-10 weeks)

### Phase 5: Dataset Integration

#### Waymo Open Dataset Loader

**Objective**: Drop-in loader for Waymo AV perception

```python
from pyroboframes.automotive import WaymoDatasetLoader

loader = WaymoDatasetLoader(
    root="/path/to/waymo_open_dataset",
    split="training",  # or "validation", "testing"
    num_workers=8,
)

for batch in loader:
    # batch contains:
    # - frames: {camera_name -> [batch, H, W, 3]}
    # - calibrations: {camera_name -> CameraCalibration}
    # - lidar: {lidar_name -> point_cloud [batch, N, 3]}
    # - annotations: {object_id -> BBox3D}
    
    panorama = stitcher.stitch(batch["frames"])
    bev = projector.frames_to_bev(batch["frames"])
    
    # Train detection model
    loss = model(bev, batch["annotations"])
```

**Data Format**:
- Raw `.tfrecord` files (~55M files, 1.9 TB) or `.parquet` (converted)
- 5 cameras per vehicle, 8 lidar units
- 1 million frames across 1150 scenes
- Synchronized timestamps (±5ms across cameras)

**Integration**:
- Auto-detect calibration from dataset metadata
- Handle variable-length sequences (1s clips)
- Load lidar as `PointCloud` (v0.4.1 format)
- Expose `gt_bboxes` as structured array

**Effort**: `L` (5-6 days) · **Testable**: Yes (download subset) · **Value**: High (production data)

#### nuScenes Dataset Loader

**Objective**: Drop-in loader for nuScenes AV dataset

```python
from pyroboframes.automotive import NuScenesDatasetLoader

loader = NuScenesDatasetLoader(
    root="/path/to/nuscenes",
    split="train",  # or "val", "test"
    version="v1.0-trainval",
)

for batch in loader:
    # 6 cameras, 5 lidar sweeps, radar
    # 1.4M frames across 1000 scenes
    # Auto-synchronized (all sensors 20ms)
```

**Effort**: `M` (3-4 days) · **Testable**: Yes · **Value**: High (widespread benchmark)

#### KITTI Dataset Loader

**Objective**: Load KITTI stereo + detection benchmark

```python
from pyroboframes.automotive import KITTIDatasetLoader

loader = KITTIDatasetLoader(
    root="/path/to/kitti",
    task="3d_detection",  # or "stereo", "optical_flow"
)
```

**Effort**: `S` (2 days) · **Testable**: Yes · **Value**: Medium (educational)

### Phase 6: Advanced 3D Perception

#### Lidar Point Cloud Fusion

**Objective**: Integrate lidar with panorama + BEV for robust 3D detection

```python
from pyroboframes.automotive import LidarBEVFusionProjector

fusion = LidarBEVFusionProjector(
    calibrations=camera_calibrations,
    lidar_pose=lidar_extrinsics,
)

# Project lidar points into BEV space
lidar_bev = fusion.project_lidar_to_bev(
    point_cloud=batch["lidar"],  # [N, 3] meters
    intensity=batch["lidar_intensity"],  # [N]
)

# Multi-modal fusion
bev_image = projector.frames_to_bev(batch["frames"])
bev_fused = np.concatenate([
    bev_image,           # [H, W, 3] RGB from cameras
    lidar_bev,           # [H, W, 1] lidar intensity
], axis=-1)

# Feed to detection model
detections = model(bev_fused)  # Lidar + vision
```

**Implementation**:
```python
def project_lidar_to_bev(self, points_xyz, intensity):
    """Project 3D lidar points to 2D BEV grid.
    
    Args:
        points_xyz: [N, 3] in meters (world frame)
        intensity: [N] reflectance 0-255
    
    Returns:
        [bev_h, bev_w, 2] with intensity and height
    """
    # Transform to BEV coordinates
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    
    # Map to BEV grid
    grid_x = ((x - self.bev_range[0]) / self.pixel_size_x).astype(int)
    grid_y = ((y - self.bev_range[2]) / self.pixel_size_y).astype(int)
    
    # Occupancy grid: max height per cell
    bev = np.zeros((self.bev_h, self.bev_w, 2))
    
    valid = (grid_x >= 0) & (grid_x < self.bev_w) & \
            (grid_y >= 0) & (grid_y < self.bev_h)
    
    bev[grid_y[valid], grid_x[valid], 0] = np.maximum(
        bev[grid_y[valid], grid_x[valid], 0],
        intensity[valid]
    )
    bev[grid_y[valid], grid_x[valid], 1] = np.maximum(
        bev[grid_y[valid], grid_x[valid], 1],
        z[valid]
    )
    
    return bev
```

**Effort**: `M` (4-5 days) · **Testable**: Yes (synthetic lidar) · **Value**: High (true 3D)

#### Occupancy Grid Mapping

**Objective**: Probabilistic occupancy for planning

```python
from pyroboframes.automotive import OccupancyGridMapper

mapper = OccupancyGridMapper(
    bev_size=(400, 400),
    bev_range=(-50, 100, -30, 30),
)

# Process frame sequence
for frames, lidar in dataset:
    bev_image = projector.frames_to_bev(frames)
    bev_lidar = lidar_projector.project(lidar)
    
    # Update occupancy (Bayesian grid)
    mapper.update(
        image_evidence=bev_image,
        lidar_evidence=bev_lidar,
    )
    
    # Get occupancy probability map
    occupancy = mapper.get_occupancy()  # [H, W] in [0, 1]
    
    # Use for planning
    free_space = 1.0 - occupancy
    path = planner.plan(robot_pos, goal, free_space)
```

**Algorithm**: Inverse sensor model + Bayes filter

```python
# For each BEV cell:
# p(occupied | evidence) = p(evidence | occupied) * p(occupied) / p(evidence)

# Simple rule:
# - If camera pixel visible (non-zero) → occupied
# - If camera pixel black (zero) → free
# - Lidar points → occupied
# - Gaps between lidar → free
```

**Effort**: `M` (3-4 days) · **Testable**: Yes · **Value**: Medium (for planners)

---

## v0.5.3+: Vision-Language + Foundation Models

### Planned (v0.5.3)

- [ ] CLIP embeddings for frame semantic understanding
- [ ] SAM/SAM2 segmentation masks in BEV
- [ ] Open-vocab detection (Grounding DINO)
- [ ] Vision-language training utilities

**Rationale**: Enable multimodal AV models (e.g., "turn left at the red building")

---

## Open Research Questions

### 1. Seam Ghosting in High-Motion Scenes

**Problem**: Fast-moving vehicles create ghosts at panorama seams

**Potential Solutions**:
a) **Depth-aware blending** (needs depth map)
   - Use lidar to identify closer objects
   - Prefer seams in far regions
   
b) **Semantic-aware seams** (needs segmentation)
   - Avoid seams through pedestrians/vehicles
   - Use DINO for real-time detection
   
c) **Optical flow filtering** (tested in Phase 4b)
   - Track fast-moving regions
   - Warp/de-rotate for stability

**Recommendation**: Start with optical flow (Phase 4b). Add semantic if time permits (v0.5.3).

### 2. GPU Memory for Large Panoramas

**Problem**: Laplacian pyramids (4 levels) require 4× image memory
- Single 720×1280 frame: ~3 MB
- Pyramids: ~12 MB
- Batch of 8: ~96 MB on CPU
- On GPU: must manage memory carefully

**Solutions**:
- Stream processing (pyramid level by level)
- In-place operations (PyTorch/CuPy)
- Checkpoint-based gradient computation (if training)

**Recommendation**: Start with batch size 8 (typical). Add streaming if needed.

### 3. Calibration from Wild Data

**Problem**: Real AV datasets may have miscalibrated cameras

**Detection**:
- Compute reprojection error across seams
- Flag cameras with error > threshold

**Correction** (optional, v0.5.2):
- Auto-calibrate using feature matching
- Optimize extrinsics (yaw/pitch) via seam overlap

**Recommendation**: Document calibration requirements. Add auto-correction in v0.5.2.

---

## Testing & Validation Strategy

### Phase 4a (GPU Acceleration)

```bash
# Baseline
pytest tests/test_automotive_stitching.py::TestCylindricalStitcher -v

# GPU backend
pytest tests/test_automotive_gpu_backend.py::TestCuPyStitcher -v -m "[C]"
pytest tests/test_automotive_gpu_backend.py::TestMLXStitcher -v

# Benchmark
python benches/automotive_gpu_benchmark.py --compare-cpu-gpu
```

### Phase 4b (Temporal Consistency)

```bash
# Optical flow validation
pytest tests/test_automotive_temporal.py::TestOpticalFlow -v

# Seam tracking
pytest tests/test_automotive_temporal.py::TestSeamTracking -v

# Video-level tests (5-10 frame sequences)
pytest tests/test_automotive_temporal.py::TestTemporalStitcher -v
```

### Phase 5-6 (Datasets + Perception)

```bash
# Dataset loaders
pytest tests/test_automotive_datasets.py::TestWaymoLoader -v --download-tiny-subset
pytest tests/test_automotive_datasets.py::TestNuScenesLoader -v

# 3D perception
pytest tests/test_automotive_perception.py::TestLidarFusion -v
pytest tests/test_automotive_perception.py::TestOccupancyMapping -v
```

---

## Effort & Timeline Estimate

### v0.5.1 (4-6 weeks, starting now)

| Phase | Component | Effort | Days | Start | End |
|-------|-----------|--------|------|-------|-----|
| **4a** | CuPy backend | M | 4 | Wk1 | Wk1.5 |
| **4a** | MLX backend | M | 4 | Wk1.5 | Wk2.5 |
| **4a** | GPU benchmarks | S | 2 | Wk3 | Wk3 |
| **4b** | Optical flow integration | M | 4 | Wk2 | Wk3.5 |
| **4b** | Seam tracking | M | 3 | Wk3 | Wk4 |
| **4b** | Testing & benchmarks | M | 3 | Wk4 | Wk4.5 |
| **Buffer** | Debugging, docs | S | 3 | Wk5 | Wk6 |
| | **Total** | | **23 days** | | |

### v0.5.2 (8-10 weeks)

| Phase | Component | Effort | Days |
|-------|-----------|--------|------|
| **5** | Waymo loader | L | 5 |
| **5** | nuScenes loader | M | 4 |
| **5** | KITTI loader | S | 2 |
| **6** | Lidar fusion | M | 5 |
| **6** | Occupancy mapping | M | 4 |
| **Testing** | Integration tests | M | 5 |
| **Docs** | Dataset guides | M | 3 |
| | **Total** | | **28 days** |

---

## Success Criteria

### v0.5.1
- ✅ GPU acceleration: 100+ FPS NVIDIA, 50+ FPS Apple Silicon
- ✅ Temporal consistency: Reduced flicker (visual inspection + PSNR metric)
- ✅ All tests passing on CPU + GPU
- ✅ Documentation + examples

### v0.5.2
- ✅ Waymo/nuScenes loaders working end-to-end
- ✅ Lidar fusion with 3D detection (Waymo benchmark)
- ✅ Occupancy mapping generating valid grids
- ✅ Dataset guides + tutorials

### v0.5.3+
- ✅ Vision-language model support
- ✅ CLIP + SAM integration
- ✅ End-to-end AV model training example

---

## Dependencies & Blockers

### For v0.5.1

**Hard dependencies**:
- CuPy (for NVIDIA) — mature, production-ready
- MLX (for Apple Silicon) — stable, but check for GPU ops

**Optional**:
- RAFT optical flow — download from HF Hub
- TorchVision optical flow fallback

**Blockers**: None identified

### For v0.5.2

**Data**:
- Waymo Open Dataset — 1.9 TB, need S3 access + credentials
- nuScenes — 430 GB, publicly available
- KITTI — 175 GB, publicly available

**ML dependencies**:
- PyTorch detection models (FCOS3D, BEVFormer) — optional, for training

**Blockers**: None (all data publicly available)

---

## References

1. **Optical Flow**:
   - RAFT: https://github.com/princeton-visionlab/RAFT
   - LiteFlowNet: https://github.com/twhui/LiteFlowNet2

2. **Datasets**:
   - Waymo: https://waymo.com/open/download/
   - nuScenes: https://www.nuscenes.org/download
   - KITTI: http://www.cvlibs.net/datasets/kitti/

3. **3D Detection Baselines**:
   - FCOS3D: https://github.com/Megvii-BaseDetection/FCOS3D
   - BEVFormer: https://github.com/fundamentalvision/BEVFormer

4. **GPU Libraries**:
   - CuPy: https://docs.cupy.dev/
   - MLX: https://ml-explore.github.io/mlx/

---

## Next Steps

1. **v0.5.1 kickoff** (this week):
   - Spike CuPy integration (GPU detection, basic panorama stitch)
   - Set up GPU benchmark harness
   - Create optical flow POC (RAFT loader)

2. **v0.5.1 mid-point** (week 2-3):
   - GPU backend complete (CuPy + MLX)
   - Temporal consistency working on synthetic video
   - Update benchmarks + documentation

3. **v0.5.1 final** (week 4-6):
   - All tests passing, GPU verified
   - Release v0.5.1 with GPU acceleration + temporal consistency

4. **v0.5.2 research** (parallel):
   - Download tiny Waymo subset, prototype loader
   - Plan 3D perception architecture

---

**Status**: Ready to execute. No blockers identified. Recommend starting Phase 4a & 4b in parallel.
