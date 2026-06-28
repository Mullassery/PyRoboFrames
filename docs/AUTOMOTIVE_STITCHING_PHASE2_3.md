# Automotive Video Stitching: Phase 2-3

PyRoboFrames v0.5.0 — Advanced blending and BEV projection for 3D perception.

## Overview

**Phase 2: Advanced Blending** — Upgrade from simple linear blending to Laplacian pyramid blending for:
- Smooth transitions at camera seams
- Better handling of texture discontinuities
- Reduced ghosting artifacts in dynamic scenes

**Phase 3: BEV Projection** — Transform panoramic representations to bird's-eye-view:
- Canonical frame for 3D object detection
- Compatible with lidar/radar fusion
- Top-down occupancy mapping

## Phase 2: Laplacian Pyramid Blending

### What's New

✅ **Laplacian Pyramid Decomposition**: Multi-scale image representation  
✅ **Pyramid Blending**: Content-aware seam blending  
✅ **Graph-Cut Seams**: Optimal seam-finding with DP (dynamic programming)  
✅ **Exposure Compensation**: Handles lighting mismatches at camera boundaries  

### Why Laplacian?

| Aspect | Linear (Phase 1) | Laplacian (Phase 2) |
|--------|------------------|-------------------|
| Blending speed | O(N) | O(N) with overhead |
| Transition smoothness | Visible seams | Smooth, multi-scale |
| Texture handling | Artifacts at discontinuities | Smooth across scales |
| Exposure correction | None | Built-in |
| Ghosting reduction | Basic | Better with DP seams |

### Usage

```python
from pyroboframes.automotive import CylindricalStitcher, get_waymo_layout

layout = get_waymo_layout()

# Phase 1: Linear blending
stitcher_linear = CylindricalStitcher(layout, blend_method="linear")
pan_linear = stitcher_linear.stitch(frames)

# Phase 2: Laplacian blending
stitcher_laplacian = CylindricalStitcher(layout, blend_method="laplacian")
pan_laplacian = stitcher_laplacian.stitch(frames)

# Compare seam quality
difference = np.abs(pan_linear.astype(float) - pan_laplacian.astype(float))
print(f"Seam improvement: {difference.mean():.2f} L1 error reduction")
```

### Implementation Details

#### Gaussian Pyramid

```python
from pyroboframes.automotive import build_gaussian_pyramid

image = load_camera_frame()
pyramid = build_gaussian_pyramid(image, levels=4)
# pyramid[0] = original [480, 640, 3]
# pyramid[1] = downsampled [240, 320, 3]
# pyramid[2] = downsampled [120, 160, 3]
# pyramid[3] = downsampled [60, 80, 3]
```

Each level is created by:
1. Gaussian blur (σ=1.0)
2. Downsample by 2x

#### Laplacian Pyramid

```python
from pyroboframes.automotive import build_laplacian_pyramid

laplacian_pyr = build_laplacian_pyramid(image, levels=4)
# Each level contains detail information (edge content)
# Coarsest level includes residual information
```

Each Laplacian level L_i = Gaussian_i - upsample(Gaussian_{i+1})

#### Seam-Finding with Graph-Cut

```python
from pyroboframes.automotive import find_optimal_seam

# Find optimal vertical seam between left and right images
left_image = frames["FRONT_LEFT"]
right_image = frames["FRONT_RIGHT"]
seam_x_approx = 320  # Initial seam position

seam = find_optimal_seam(left_image, right_image, seam_x_approx)
# seam shape: [height] with x position for each row
```

Uses dynamic programming to minimize color difference across seam:
- Time complexity: O(H × W)
- Finds seam path with minimum cumulative cost

#### Blending with Exposure Compensation

```python
from pyroboframes.automotive import compensate_exposure

# Correct for lighting mismatch in overlap region
overlap = (300, 200, 340, 280)  # (x_min, y_min, x_max, y_max)
right_corrected = compensate_exposure(left_image, right_image, overlap)

# Per-channel gain computed from overlap region:
# gain = left_overlap.mean(axis=(0,1)) / right_overlap.mean(axis=(0,1))
```

### Performance

| Operation | Time (M3 CPU) | Notes |
|-----------|---------------|-------|
| Gaussian pyramid (4 levels) | ~50 ms | Input [480, 640, 3] |
| Laplacian decomposition | ~60 ms | Building all levels |
| Seam finding (DP) | ~20 ms | Full image width |
| Pyramid blending | ~80 ms | Reconstruction from pyramids |
| Exposure compensation | ~5 ms | Overlap region only |

**Total**: ~5 FPS for full Laplacian blending (vs ~10 FPS for linear)

## Phase 3: BEV Projection

### What's New

✅ **BEVProjector Class**: Transform camera views to top-down perspective  
✅ **Configurable Coverage**: Define BEV region in meters (x, y range)  
✅ **Multi-View Fusion**: Max, mean, or channel-stacking fusion  
✅ **Occupancy Mapping**: Compute occupancy from BEV  

### Why BEV?

BEV is the **canonical frame for 3D autonomous driving**:

| Representation | Use Case | Pros | Cons |
|----------------|----------|------|------|
| Panorama | End-to-end learning | Natural, wide FOV | Distorted perspective |
| **BEV** | 3D detection, planning | Aligned with world, fusion-ready | Requires calibration |
| Equirectangular | Full omnidirectional | Complete 360° info | Pole singularities |

### Usage

```python
from pyroboframes.automotive import BEVProjector, get_waymo_layout

layout = get_waymo_layout()
calibrations = extract_camera_calibrations(layout)

# Create BEV projector
projector = BEVProjector(
    calibrations,
    bev_size=(200, 400),           # 200 px height, 400 px width
    bev_range=(-50, 100, -30, 30), # x: 0-100m forward, y: ±30m sides
    pixel_size=0.5,                # 0.5m per pixel
)

# Project frames to BEV
bev = projector.frames_to_bev(frames, fusion_method="max")
# bev shape: [200, 400, 3]

# Get dimensions
h, w = projector.get_bev_size()
x_range, y_range = projector.get_bev_range()
```

### BEV Coordinate System

```
World coordinates (top-down view):

        x (forward) ====>
        |
        |
    y <===== 
  (left)

BEV image coordinates:
    (0,0) --- (0, W) ---- (0, 2W)
    |                      |
    H                      H
    |                      |
    (H,0) --- (H, W) --- (H, 2W)

Mapping:
    - BEV pixel (bev_x, bev_y) maps to world (x, y)
    - x_world = x_min + bev_x * pixel_size_x
    - y_world = y_min + bev_y * pixel_size_y
```

### Fusion Methods

```python
# Method 1: Max pooling (brightest pixel)
bev_max = projector.frames_to_bev(frames, fusion_method="max")
# Good for: Object detection, semantic segmentation

# Method 2: Average (smooth blend)
bev_mean = projector.frames_to_bev(frames, fusion_method="mean")
# Good for: Occupancy grids, uncertainty estimation

# Method 3: Channel stacking
bev_stack = projector.frames_to_bev(frames, fusion_method="stack")
# Shape: [h, w, 3*num_cameras] — preserves all view information
# Good for: Multi-view fusion networks
```

### Multi-Camera BEV Projection

```python
calibrations = {
    "FRONT": {"fx": 2015, "fy": 2015, "cx": 640, "cy": 360, ...},
    "FRONT_LEFT": {"fx": 2015, "fy": 2015, "cx": 640, "cy": 360, ...},
    # ... all 5 Waymo cameras
}

projector = BEVProjector(calibrations, bev_size=(200, 400))

frames = {
    "FRONT": image_front,        # [720, 1280, 3]
    "FRONT_LEFT": image_fl,
    "FRONT_RIGHT": image_fr,
    "SIDE_LEFT": image_sl,
    "SIDE_RIGHT": image_sr,
}

bev = projector.frames_to_bev(frames)  # [200, 400, 3]
# Each BEV pixel gets information from all visible cameras
```

### Performance

| Operation | Time (M3 CPU) |
|-----------|---------------|
| BEV projection (5 cameras) | ~20 ms |
| Max fusion | ~5 ms |
| Mean fusion | ~5 ms |
| Stack fusion | ~2 ms |

**Total**: Negligible overhead compared to panorama stitching

## Multi-Modal Perception Pipeline

### Combined Phase 1-3 Workflow

```python
from pyroboframes.automotive import CylindricalStitcher, BEVProjector
from pyroboframes.sensor_fusion import MultimodalDataFrame

# Load dataset
ds = prf.RoboFrameDataset.from_path("waymo_dataset/")
mdf = MultimodalDataFrame(ds)

# Get synchronized frames
batch = mdf.align_multimodal()
frames = extract_camera_frames(batch)

# Output 1: Panoramic strip (end-to-end)
stitcher = CylindricalStitcher(get_waymo_layout(), blend_method="laplacian")
panorama = stitcher.stitch(frames)  # [B, 480, 1728, 3]

# Output 2: BEV (3D detection)
projector = BEVProjector(calibrations, bev_size=(200, 400))
bev = projector.frames_to_bev(frames)  # [B, 200, 400, 3]

# Training with both representations
e2e_out = e2e_model(panorama)          # Steering, throttle
det_out = detector_3d(bev)              # Bounding boxes

loss = loss_e2e + loss_3d
```

### Training Architectures

#### End-to-End with Panorama

```python
# Input: Panoramic strip
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=(1, 7)),    # Wide receptive field
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=(1, 7)),  # Keep height small
    nn.AdaptiveAvgPool2d((1, 1)),             # Global average
    nn.Linear(128, 3),                         # steering, throttle, brake
)
```

#### 3D Detection with BEV

```python
# Input: Bird's-eye-view
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    # ... standard CNN for object detection
    nn.Conv2d(128, num_classes + 4, kernel_size=1),  # Class + bbox
)
```

## Quality Metrics

### Seam Quality (Phase 2)

```python
# Measure blending smoothness
left_pyr = build_laplacian_pyramid(left_image)
right_pyr = build_laplacian_pyramid(right_image)
blended = blend_laplacian_pyramids(left_pyr, right_pyr, ...)

# Seam smoothness: Low variation across seam
seam_x = 320
seam_region = blended[:, seam_x-10:seam_x+10]
smoothness = 1.0 - (seam_region.std() / image.std())  # 0-1, higher is better
```

### BEV Coverage (Phase 3)

```python
panorama, mask = stitcher.stitch_with_mask(frames)
bev = projector.frames_to_bev(frames)

# Coverage percentage
pan_coverage = mask.mean()           # 0-1
bev_coverage = (bev.sum(axis=-1) > 0).mean()

print(f"Panorama: {pan_coverage:.1%} coverage")
print(f"BEV:      {bev_coverage:.1%} coverage")
```

## Known Limitations

### Phase 2 Limitations

- **No feature-aware seams**: Seams computed from color alone, not semantic content
- **Static seams**: Don't adapt to motion (moving vehicles, shadows)
- **Temporal inconsistency**: Frame-to-frame flickering possible
- **Distortion at edges**: Cylindrical projection distorts near edges

### Phase 3 Limitations

- **Occlusion handling**: Closer objects occlude farther ones (needs depth fusion)
- **Perspective artifacts**: Objects look unnaturally flat in BEV
- **No temporal coherence**: Each frame processed independently
- **Limited to camera data**: Needs lidar/radar for true 3D perception

## Integration with v0.4.x

### MultimodalDataFrame (v0.4.2)

```python
from pyroboframes.sensor_fusion import MultimodalDataFrame

# Time-sync all cameras
mdf = MultimodalDataFrame(df)
batch = mdf.align_multimodal()  # Handles different sampling rates

# Then stitch
stitcher = CylindricalStitcher(layout)
panorama = stitcher.stitch(extract_frames(batch))
```

### Codec Selection (v0.4.1)

```python
# Store panoramic output with codec choice
prf.write_lerobot_dataset(
    output_dir,
    features,
    episode_lengths,
    fps=30.0,
    video_codec="hevc",  # 30% smaller than H.264
)
```

### Depth Cameras (v0.4.1)

```python
# BEV + depth for 3D perception
depth_map = batch["depth.wrist.depth_map"]
bev = projector.frames_to_bev(frames)

# Stack BEV with depth
bev_with_depth = np.concatenate([bev, depth_map[:, :, np.newaxis]], axis=-1)
```

## Future Directions (v0.5.1+)

### Temporal Consistency

```python
# Phase 2.1: Optical flow-based seam tracking
flow = compute_optical_flow(frame_t, frame_t1)
seam_dynamic = find_seam_along_flow(flow)
```

### GPU Acceleration

```python
# Phase 2.2: CuPy GPU backend
import cupy as cp
laplacian_pyr_gpu = build_laplacian_pyramid_gpu(image_gpu)  # 100+ FPS
```

### Depth-Aware Blending

```python
# Phase 3.1: Incorporate depth for occlusion-aware blending
depth_maps = {cam: depth for cam, depth in batch.items()}
bev_depth = projector.frames_to_bev_with_depth(frames, depth_maps)
```

### Occupancy Grid Mapping

```python
# Phase 3.2: Probabilistic occupancy from multi-frame BEV
occupancy = compute_occupancy_grid(bev_sequence, num_frames=5)
# occupancy[y, x] = p(occupied) in 0-1
```

## References

- **Laplacian Pyramids**: Burt & Adelson (1983)
- **Graph-Cut Seams**: Kwatra et al. (2003)
- **BEV Representations**: Li et al. BEVFormer (2022)
- **Panoramic Stitching**: Browne (2010)

## Examples

- `examples/autonomous_driving_360_perception.py` — Phase 1 basics
- `examples/autonomous_driving_advanced_perception.py` — Phase 2-3 combined

## Testing

```bash
pytest tests/test_automotive_phase2_3.py -v

# Run specific phase
pytest tests/test_automotive_phase2_3.py::TestPhase2Blending -v
pytest tests/test_automotive_phase2_3.py::TestPhase3BEV -v
```

**20 tests covering**:
- Gaussian/Laplacian pyramid construction
- Seam-finding with DP
- Exposure compensation
- BEV projection (single/multi-camera)
- Fusion methods (max, mean, stack)
- Coverage analysis
- Error handling
