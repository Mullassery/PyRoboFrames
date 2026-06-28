# Automotive Video Stitching Research — v0.5.0 Feature Exploration

**Objective:** Enable 360° perception for autonomous driving datasets by stitching multiple camera feeds into panoramic/spherical views.

**Scope:** Research document exploring design options, architecture, and implementation strategy.

---

## 1. Problem Statement

### Current Limitation
Autonomous vehicles (AVs) have 5-8 cameras covering 360° FOV:
- **Front**: 1-3 cameras (narrow, wide, ultra-wide)
- **Rear**: 1 camera
- **Sides**: 2-4 cameras (left/right, front/rear)

**Gap:** PyRoboFrames v0.4.2 handles time-sync but not **geometric stitching** into unified 360° representation.

### AV Learning Use Cases
1. **End-to-end driving models**: Single omnidirectional input → steering + throttle
2. **360° perception fusion**: Stitch camera inputs before 3D object detection
3. **Panoramic video creation**: Replay drive logs with unified view
4. **Coverage analysis**: Identify blind spots from stitched coverage map

### Datasets Affected
- **Waymo Open Dataset**: 5 cameras (front/back/left/right/side)
- **nuScenes**: 6 cameras + 5 radars + lidar
- **KITTI**: 2 stereo cameras + perspective views
- **Argoverse 2.0**: 7 camera streams

---

## 2. Design Options

### **Option A: Cylindrical Stitching (Recommended)**

**Concept:** Warp each camera view onto a cylinder, stitch seams, create panoramic strip.

**Advantages:**
- ✅ Minimal distortion in horizontal plane (where motion happens)
- ✅ Works well for driving scenarios (vehicles move forward/backward, not up/down)
- ✅ No pole singularities (unlike spherical stitching)
- ✅ Standard in panoramic photography
- ✅ Compatible with end-to-end AV models (single input channel)

**Disadvantages:**
- ❌ Vertical FOV limited (loses sky/ground details)
- ❌ Seam artifacts at camera boundaries
- ❌ Requires accurate camera calibration

**Implementation:**
```
1. Undistort each camera frame (using calibration)
2. Warp to cylindrical projection (homography or mesh warp)
3. Blend seams (Laplacian pyramid, graph-cut, or simple linear)
4. Stack into [batch, height, width] panoramic strip
```

**Output Shape:**
- Input: 5 cameras × [480×640] = 2.4M pixels
- Output: [480, 3200] = 1.5M pixels (stitched strip)
- Benefit: Single input channel for model (no channel explosion)

**Complexity:** Medium (camera calibration critical)

---

### **Option B: Multi-View Projection (Fish-eye → Planar)**

**Concept:** Keep each camera as separate 2D view, project onto common plane (bird's-eye-view).

**Advantages:**
- ✅ No stitching artifacts (each camera independent)
- ✅ Works with fish-eye cameras (ultra-wide)
- ✅ Natural for 3D perception (projects to world coordinates)
- ✅ Compatible with lidar/radar fusion (same BEV space)

**Disadvantages:**
- ❌ Creates 5 separate channels (bandwidth/compute increase)
- ❌ Loses perspective view (harder for end-to-end models trained on RGB)
- ❌ Distortion at far regions

**Implementation:**
```
1. Undistort each camera
2. Project via intrinsics K → world coordinates (BEV)
3. Render each camera onto BEV grid
4. Stack into [batch, channels=5, height, width]
```

**Output Shape:**
- Input: 5 cameras × [480×640×3] = 4.6M pixels
- Output: [5, 256, 512] BEV = 655K pixels (compressed, efficient)
- Benefit: Aligns with 3D perception (lidar, radar already in BEV)

**Complexity:** Medium (3D projection math)

---

### **Option C: Spherical Stitching (Complete 360°)**

**Concept:** Project all cameras onto a sphere, enable full omnidirectional representation.

**Advantages:**
- ✅ Complete 360° coverage (no blind spots)
- ✅ Can render any perspective (virtual camera)
- ✅ Compatible with spherical CNN architectures
- ✅ Research-ready (matches academic datasets)

**Disadvantages:**
- ❌ Pole singularities at top/bottom (distortion)
- ❌ Complex blending (6-point blend zones)
- ❌ Large output (equirectangular projection ~2× size)
- ❌ Slower inference (CNN must handle distortion)

**Implementation:**
```
1. Project each camera onto unit sphere
2. Blend at overlapping regions
3. Sample equirectangular grid
4. Output: [batch, 2048, 4096, 3] (full sphere)
```

**Output Shape:**
- Input: 5 cameras
- Output: [2048, 4096, 3] ≈ 25M pixels (large!)
- Benefit: Complete omnidirectional view

**Complexity:** High (sphere geometry, complex blending)

---

### **Option D: Hybrid: Cylindrical + BEV (Most Practical)**

**Concept:** Cylindrical stitching for RGB → end-to-end; simultaneously project to BEV for 3D.

**Advantages:**
- ✅ Keeps end-to-end models happy (cylindrical panorama input)
- ✅ Also supports 3D perception models (BEV for detection)
- ✅ Single time-sync → multiple representations
- ✅ No redundant compute (shared undistortion)

**Disadvantages:**
- ❌ More complex pipeline
- ❌ Dual representation overhead

**Implementation:**
```
Pipeline:
  cameras → undistort → {project cylindrical, project BEV}
  → output: (panorama, bev)
```

**Complexity:** Medium-High (dual projection)

---

## 3. Architecture Design

### 3.1 Core Components

```python
# Phase 1: Camera Calibration (already have in v0.4.1)
calibrations = {
    "camera.front_center": CameraCalibration(...),
    "camera.front_left": CameraCalibration(...),
    ...
}

# Phase 2: Undistortion (reuse v0.4.1 APIs)
undistorted = undistort_frame(frame, calibrations["camera.front_center"])

# Phase 3: NEW - Projection & Stitching
panorama = stitch_cylindrical(
    frames=[frame_f, frame_r, frame_l],
    calibrations={...},
    blend_method="laplacian",  # "linear", "laplacian", "graphcut"
)

# Output: [batch, 480, 3200] panoramic strip
```

### 3.2 Stitching Pipeline

```
Input cameras: 5 (front, left-front, left, right, rear)
                ↓
1. Time-sync (use v0.4.2 MultimodalDataFrame)
                ↓
2. Undistortion (use v0.4.1 calibration)
                ↓
3. Geometric alignment (homography estimation)
                ↓
4. Seam blending (smooth transitions)
                ↓
5. Panorama composition (stitch into strip)
                ↓
Output: [batch, H, W_panorama, 3]
```

### 3.3 Module Structure

```
pyroboframes/
├── automotive/
│   ├── __init__.py
│   ├── stitching.py          # Stitching core
│   ├── projection.py         # Cylindrical/spherical projection
│   ├── blending.py           # Seam blending (Laplacian pyramid)
│   ├── calibration_utils.py  # Camera alignment tools
│   └── bev_projection.py     # Bird's-eye-view projection
├── sensor_fusion.py          # (existing v0.4.2)
└── calibration.py            # (existing v0.4.1)

tests/
├── test_automotive_stitching.py
├── test_projection_methods.py
└── test_seam_blending.py

examples/
└── autonomous_driving_360_perception.py
```

---

## 4. Implementation Strategy

### 4.1 Phase 1: Cylindrical Stitching (Weeks 1-2)

**Deliverables:**
- Camera undistortion (use v0.4.1 calibration)
- Cylindrical projection math
- Linear seam blending (simple, fast)
- Tests: synthetic camera pairs

**Complexity:** Low-Medium  
**Effort:** 40 hours

```python
# Usage
stitcher = CylindricalStitcher(calibrations)
panorama = stitcher.stitch(frames)  # [batch, 480, 3200]
```

### 4.2 Phase 2: Advanced Blending (Week 3)

**Deliverables:**
- Laplacian pyramid blending
- Graph-cut seam optimization
- Feathering and exposure compensation
- Tests: real automotive footage

**Complexity:** Medium  
**Effort:** 30 hours

### 4.3 Phase 3: BEV Projection (Week 4)

**Deliverables:**
- Bird's-eye-view projection
- Multi-camera BEV fusion
- Lidar/radar alignment (optional)
- Tests: 3D perception pipeline

**Complexity:** Medium  
**Effort:** 30 hours

### 4.4 Phase 4: End-to-End Example (Week 5)

**Deliverables:**
- E2E autonomous driving example
- Synthetic AV dataset generation
- Training-ready batch stacking
- Full test coverage

**Complexity:** Low  
**Effort:** 20 hours

**Total Effort:** ~120 hours (3-4 weeks)

---

## 5. Technical Challenges & Solutions

### 5.1 Camera Calibration Accuracy

**Challenge:** Stitching quality depends entirely on accurate intrinsics + extrinsics.

**Solution:**
- Require camera calibration from dataset metadata
- Auto-detect from OpenCV calibration files or manufacturer specs
- Provide calibration validation tool
- Fall back to approximate extrinsics if unavailable

### 5.2 Temporal Consistency

**Challenge:** Frame-to-frame jitter in seams during motion.

**Solution:**
- Use v0.4.2 MultimodalDataFrame for perfect time-sync
- Apply temporal filtering on seam blending weights
- Track feature points across frames for stability

### 5.3 Dynamic Scenes

**Challenge:** Moving objects at seams cause ghosting/artifacts.

**Solution:**
- Compute per-pixel depth (from stereo or lidar)
- Warp via depth for correct occlusion handling
- Use depth-aware blending

### 5.4 Computational Cost

**Challenge:** Stitching 5 × [480×640] in real-time is expensive.

**Solution:**
- Vectorized OpenCV/NumPy implementation
- GPU acceleration via CuPy (optional)
- Lazy stitching (only stitch frames needed for training)
- Cache computed homographies between frames

---

## 6. Integration with v0.4.x Stack

### Current State
```
v0.4.0: Codecs + depth cameras
v0.4.1: Calibration + depth I/O
v0.4.2: Multimodal sensor fusion (RGB + depth + IMU)
```

### v0.5.0 Addition
```
v0.5.0: Automotive video stitching + 360° perception
├── Cylindrical stitching pipeline
├── BEV projection for 3D
├── Seam blending (Laplacian pyramid)
├── Autonomous driving examples
└── Compatible with all v0.4.x features
```

### API Integration
```python
# Use existing v0.4.2 MultimodalDataFrame for time-sync
mdf = MultimodalDataFrame(df)
batch = mdf.align_multimodal()  # Time-sync all cameras

# NEW v0.5.0: Add stitching
from pyroboframes.automotive import CylindricalStitcher

stitcher = CylindricalStitcher(calibrations)
panorama = stitcher.stitch(
    frames=[batch["camera.front"], batch["camera.left"], ...],
    blend_method="laplacian"
)

# Result: [batch, 480, 3200] panoramic strip for E2E model
```

---

## 7. Datasets & Benchmarks

### Target Datasets
1. **Waymo Open Dataset**
   - 5 cameras, ~1M frames
   - High-quality calibration provided
   - 200GB+ data

2. **nuScenes**
   - 6 cameras, ~1.4M frames
   - Lidar + radar ground truth
   - Ideal for 3D perception validation

3. **KITTI**
   - 2 stereo + perspective views
   - Smaller, easier for prototyping
   - 39 drives, ~390K frames

### Benchmark Plan
```
Metric 1: Seam quality (visual inspection + image difference)
Metric 2: 3D consistency (project through stitch boundary, check jitter)
Metric 3: Inference speed (frames/sec, GPU vs CPU)
Metric 4: Training stability (does stitched input converge faster/slower?)
```

---

## 8. Open Design Questions

### Q1: Single Channel vs Multi-Channel Output?
- **Option A:** Panoramic strip [480, 3200, 3] — easier for end-to-end, but tight FOV
- **Option B:** Multi-channel [5, 480, 640, 3] — flexible, but network complexity
- **Recommendation:** Start with panoramic (simpler), support multi-channel later

### Q2: Hardcoded Camera Layout vs Generic?
- **Option A:** Hardcoded for Waymo (front, sides, rear) — faster, simpler
- **Option B:** Generic (accept camera list) — more flexible, harder to get right
- **Recommendation:** Support both; provide common presets (Waymo, nuScenes, KITTI)

### Q3: Separate Module or Integrate with MultimodalDataFrame?
- **Option A:** Separate `automotive.stitching` module — modular, clean separation
- **Option B:** Extend MultimodalDataFrame — tight integration, but bloats it
- **Recommendation:** Separate module, with `MultimodalDataFrame` integration point

### Q4: Depth for Stitching?
- **Option A:** Use depth for occlusion handling — better quality, complex
- **Option B:** Ignore depth, use image-based stitching only — simpler, good enough
- **Recommendation:** Phase 1 without depth, add in Phase 3 if needed

---

## 9. Success Criteria for v0.5.0

- ✅ Cylindrical stitching with 5+ camera support
- ✅ Laplacian pyramid blending for smooth seams
- ✅ Waymo/nuScenes compatibility with auto-calibration detection
- ✅ 15 comprehensive tests (geometric, visual, temporal)
- ✅ Complete autonomous driving example (write panorama, load, train)
- ✅ 50+ FPS stitching on CPU (M3), 200+ FPS on GPU
- ✅ Documentation with comparison to prior work
- ✅ Zero breaking changes to v0.4.x APIs

---

## 10. Competitive Landscape

### Existing Solutions
- **ROS image_geometry**: Single-camera undistortion only
- **OpenCV stitching module**: Stitches 2-3 images, high overhead
- **CustomVision (Azure)**: Proprietary, not suitable for robotics
- **Autonomous driving papers**: CVPR/ICCV but no released code

### PyRoboFrames Advantage
- Integration with time-sync + calibration (v0.4.x)
- Zero-copy multimodal loading (depth + IMU already support)
- Optimized for training loops (batch processing)
- Open-source, MIT license

---

## 11. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Calibration inaccuracy | Medium | High | Validate with known good calibrations first |
| Seam ghosting in motion | Medium | Medium | Use depth + temporal filtering |
| Performance (too slow) | Low | Medium | Profile, optimize inner loops, GPU fallback |
| API design (wrong choice) | Low | Low | Design review before Phase 1 |

---

## 12. Recommended Next Steps

### If Implementing Now:
1. Get Waymo calibration docs + 10 sample frames
2. Prototype cylindrical stitching in Jupyter notebook
3. Compare visual quality against OpenCV's stitching
4. Decide on blending method (linear vs Laplacian)
5. Start Phase 1 implementation

### If Deferring:
1. Keep this research doc as v0.5.0 spec
2. Monitor community stitching solutions
3. Check if Waymo/nuScenes publish updated calibrations
4. Plan integration with LiDAR odometry if available

---

## Appendix A: Mathematical Formulas

### Cylindrical Projection
```
Point in camera frame: (X, Y, Z)
Normalize: x = X/Z, y = Y/Z

Cylindrical coordinates:
  theta = atan2(x, 1)
  v = y / sqrt(1 + x²)
  
Image coordinates:
  u_panorama = theta * R  (R = focal length / cylinder radius)
  v_panorama = v * R
```

### Laplacian Pyramid Blending
```
For each scale s:
  pyramid_A[s] = Gaussian(A, sigma=2^s) - Gaussian(A, sigma=2^(s+1))
  pyramid_B[s] = Gaussian(B, sigma=2^s) - Gaussian(B, sigma=2^(s+1))
  
  blend[s] = mask[s] * pyramid_A[s] + (1 - mask[s]) * pyramid_B[s]
  
Reconstruct: result = sum(blend[s] for all s)
```

---

## Appendix B: Example Dataset Specs

### Waymo
- 5 cameras: front_center, front_left, front_right, side_left, side_right
- Resolution: 1920×1280
- Calibration: Provided in dataset metadata
- Baseline: 1.3m (front to side)

### nuScenes
- 6 cameras: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_BACK
- Resolution: 1600×900
- Calibration: Provided, validated against lidar
- Baseline: 1.6m (front to side)

### KITTI
- 4 cameras: left stereo (gray/color), right stereo (gray/color)
- Resolution: 1242×375 (stereo) or 1242×375 (perspective)
- Calibration: Provided
- Baseline: 0.54m (stereo baseline)

---

## Summary

**Recommendation: Implement Option D (Hybrid: Cylindrical + BEV) for v0.5.0**

- **Scope:** 4-week sprint, ~120 hours
- **Complexity:** Medium (well-understood problem domain)
- **Impact:** Unlocks autonomous driving research on PyRoboFrames
- **Integration:** Seamless with v0.4.2 multimodal API
- **Risk:** Low (camera calibration is well-established)

**Starting point:** Prototype cylindrical stitching, then decide on BEV depth for Phase 3.
