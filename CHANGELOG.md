# Changelog

All notable changes to PyRoboFrames are documented in this file.

## [1.2.0] — 2026-07-17

### 🎉 Major Features (Complete Phases 4-7)

#### Phase 4: GPU Acceleration
- **GPU Transform Operators** — `gpu_acceleration.py` module with automatic fallback chain
  - CuPy backend for NVIDIA (CUDA)
  - MLX backend for Apple Silicon GPU
  - NumPy fallback for CPU
  - `GPUTransforms` class with `resize()` and `normalize()` operations
- **Automatic Device Detection** — `device="auto"` picks best available: CUDA → MLX → NumPy
- **Cross-Platform Parity** — Same transform code runs on M3, RTX 5090, H100 without changes

#### Phase 5: Temporal Consistency
- **Optical Flow Estimation** — `OpticalFlowEstimator` for motion detection
  - Lucas-Kanade method (real OpenCV when available)
  - Gradient-based fallback (pure NumPy)
- **Temporal Filtering** — `TemporalFilter` for video smoothing
  - Exponential moving average (EMA) for motion smoothing
  - Median filtering for temporal denoising
- **Use Case** — Reduces flickering in panoramic stitching, improves temporal coherence

#### Phase 6: Real-World Autonomous Driving Datasets
- **Waymo Open Dataset Loader** — `WaymoDatasetLoader`
  - Scene indexing and metadata parsing
  - Per-camera frame loading with automatic calibration
  - Standard Waymo intrinsics (fx=2015, fy=2015, cx=960, cy=600)
- **nuScenes Integration** — `nuScenesDatasetLoader`
  - Multi-camera support (CAM_FRONT, etc.)
  - Standard nuScenes calibration (1266px focal length)
  - 20 Hz frame rate support
- **KITTI Dataset Support** — `KITTIDatasetLoader`
  - Training/testing split support
  - Stereo camera handling (camera 0, 1)
  - Calibration file parsing from KITTI format
- **Unified API** — All loaders expose:
  - `CameraCalibration` (fx, fy, cx, cy, width, height, distortion)
  - `FrameMetadata` (timestamp, camera_name, frame_index, image_path, calibration)

#### Phase 7: Occupancy & 3D Perception
- **Occupancy Grid Mapping** — `OccupancyGrid` class
  - World-to-grid coordinate transformation (configurable resolution)
  - Point cloud integration (ray tracing for free space)
  - 3D bounding box insertion
  - Morphological operations (dilate, erode)
  - Free space mask extraction
- **LiDAR Processing** — `LiDARProcessor` utility class
  - Distance filtering (by max range)
  - Height filtering (by min/max Z)
  - Ground segmentation (elevation threshold)
  - Point clustering (DBSCAN-like, with scikit-learn fallback)
  - Surface normal estimation (PCA on k-NN)
- **Radar Fusion** — `RadarFusionProcessor` for multi-sensor perception
  - Velocity estimation from Doppler shift
  - Spatial association (radar ↔ LiDAR)
  - Combined position + velocity output for tracking

### 📊 Testing
- **47 New Tests** (222 total, up from 175)
  - 10 GPU acceleration tests (resize, normalize, device selection)
  - 16 dataset loader tests (Waymo, nuScenes, KITTI, calibration)
  - 21 occupancy/3D tests (grid operations, LiDAR, radar fusion)
- **All Tests Passing** — 100% pass rate
- **Coverage** — 82%+ coverage across new modules

### 📚 Documentation
- **Updated README.md**
  - v1.2 feature showcase with code examples
  - Updated test count (175 → 222)
  - Competitive analysis matrix
  - GPU support section
- **ROADMAP.md Updates**
  - Marked P0-P11 complete
  - Updated version status (v0.5.0 → v1.2.0)
- **New COMPETITIVE_ANALYSIS.md**
  - Detailed comparison vs torchcodec, Robo-DM, LeRobot, PyAV
  - Honest strengths and weaknesses
  - Use case guidance ("Train Anywhere" positioning)

### 🔧 Engineering
- **Python 3.13 Support** — Updated pyproject.toml
  - Changed `requires-python = "==3.10"` → `">=3.10"`
  - Added Python 3.13 classifier
- **Version Bump** — Cargo.toml: 1.0.0 → 1.2.0
- **PyPI Release** — Published as `pyroboframes-1.2.0.tar.gz`

### ✨ Key Differentiators
1. **Only** dataloader with multi-platform GPU parity (NVIDIA + Apple Silicon)
2. **Only** tool bridging robot learning + autonomous driving datasets
3. **Only** loader with native occupancy grid + 3D perception
4. **Largest test suite** for robot dataloaders (222 tests, 82% coverage)

---

## [1.1.0] — Previous Release
See GitHub tags for v1.1.0 and earlier.

---

## Future Roadmap

### v1.3.0 (Planned)
- [ ] Streaming ingestion (MQTT / Kafka)
- [ ] Vision-language dataset generation (auto-annotation with CLIP/SAM2)
- [ ] Distributed loading (Ray / Slurm / RunPod)
- [ ] Additional AV datasets (KITTI-360, Argoverse 2)

### v2.0.0+ (Long-term)
- [ ] Zero-copy MLX arrays (awaiting upstream mlx#2855)
- [ ] Real-time imitation learning pipelines
- [ ] Advanced curriculum learning strategies
- [ ] Enterprise deployment templates

---

## Security

### 🔒 v1.2.0 Security Status
- ✅ Path traversal protection (S3/GCS access)
- ✅ Hardware fallback graceful degradation
- ✅ Dependency pinning (numpy==1.24, pyarrow==14)
- ⚠️ PyPI token rotated (was exposed in development)

See [DEPLOYMENT_SECURITY.md](./DEPLOYMENT_SECURITY.md) for details.

---

## Contributors
- **Georgi Mammen Mullassery** — Core implementation, GPU acceleration, dataset loaders, 3D perception

---

## Installation

```bash
pip install pyroboframes==1.2.0
# or
uv add pyroboframes==1.2.0
```

For development:
```bash
git clone https://github.com/Mullassery/PyRoboFrames.git
cd PyRoboFrames
pip install -e ".[dev]"
pytest tests/ -v
```
