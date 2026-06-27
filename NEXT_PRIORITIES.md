# PyRoboFrames — Next Batch Priorities (v0.3.1 → v1.0)

**Current Release:** v0.3.0 (GPU decode & transform backends)  
**Next Milestones:** v0.3.1 (GPU verification) → v0.4.0 (Quality improvements) → v1.0 (Full humanoid)

---

## Priority Tiers

### 🔴 P0 — Critical Path to v1.0 (Blocking other work)

#### **P0.1: GPU Hardware Verification** (1-2 weeks)
**Why:** Blocks performance claims, customer deployment decisions, v1.0 readiness  
**What:**
- [ ] Rent NVIDIA GPU box (RunPod, Lambda Labs, or AWS): H100 / RTX 5090 / A100
- [ ] Run NVDEC benchmarks: `benches/nvidia_benchmark.py` with `--features cuda`
- [ ] Run CV-CUDA transform benchmarks: decode + transforms end-to-end
- [ ] Document actual vs. expected performance (3-5× decode, 5-10× transforms)
- [ ] Create performance report: GPU vs CPU baseline + comparison table
- [ ] Update README with verified performance numbers

**Effort:** 40-60 hours (setup, testing, documentation)  
**Owner:** Needs GPU hardware access (external contributor or cloud rental)  
**Deliverables:**
- GPU benchmark results (CSV + plots)
- Updated performance section in README
- Known issues/workarounds doc (if any)

**Unblocks:**
- P0.2 (MLX zero-copy), P1.1 (v0.4.0 release), marketing claims

---

#### **P0.2: MLX Zero-Copy Integration** (Pending mlx#2855)
**Why:** 3× speedup on Apple Silicon, differentiator vs competitors  
**Status:** Blocked on upstream mlx#2855 (MLX IOSurface support)  
**Action Items:**
- [ ] Monitor mlx#2855 for resolution
- [ ] When resolved: Implement IOSurface FrameBuffer variant
- [ ] Implement Python/MLX FFI for IOSurface hand-off
- [ ] Add bench_mlx_zero_copy.py: compare NumPy hop vs direct
- [ ] Write MLX zero-copy guide in docs/
- [ ] Benchmark on M3/M4 Max to show 3× speedup

**Effort:** 30-40 hours (implementation + benchmarking)  
**Owner:** Can be started immediately; implementation waits on mlx#2855  
**Deliverables:**
- IOSurface FrameBuffer implementation
- MLX zero-copy benchmark
- Documentation + quickstart example

**Unblocks:**
- P1.3 (Apple Silicon marketing), v1.0 launch

---

### 🟠 P1 — High-Impact, Near-term (v0.3.1–v0.4.0)

#### **P1.1: v0.3.1 Point Release — GPU Verification Ready** (1 week)
**Why:** Deliver GPU verification results to customers  
**What:**
- [ ] Incorporate P0.1 results into docs/GPU_VERIFICATION.md
- [ ] Add actual benchmark numbers (NVDEC, CV-CUDA, MLX)
- [ ] Create "Known Good" configuration matrix (GPU model × CUDA version × OS)
- [ ] Bump version to v0.3.1, release to PyPI
- [ ] Publish GitHub release notes with performance data

**Effort:** 20 hours  
**Owner:** whoever runs P0.1  
**Deliverables:**
- v0.3.1 release on PyPI
- Updated GPU_VERIFICATION.md with real data
- GitHub release with performance report

---

#### **P1.2: Video Codec Selection & Storage Savings** (2 weeks)
**Why:** 40–50% storage reduction (critical for large datasets)  
**What:**
- [ ] Add codec selection to `write_lerobot_dataset()`: H.264 (default) vs HEVC vs AV1
- [ ] Implement codec detection in `RoboFrameDataset` (auto-detect from metadata)
- [ ] Add `loader(codec="hevc")` option for selective transcoding
- [ ] Benchmark storage savings: H.264 vs HEVC vs AV1
- [ ] Document trade-offs (encode time, decode speed, compatibility)
- [ ] Add storage_benchmark.py: measure codec storage + decode speed

**Effort:** 60 hours  
**Owner:** Internal (codec integration straightforward)  
**Deliverables:**
- Codec selection API
- Storage savings benchmark
- Migration guide (recompressing datasets)
- docs/CODEC_SELECTION.md

**Unblocks:**
- P1.4 (streaming at scale), v1.0 "data-efficient" claims

---

#### **P1.3: Depth Camera Support — Point Clouds** (3 weeks)
**Why:** VLA models need depth (dexterity, grasping, 3D reasoning)  
**What:**
- [ ] Add point cloud reading: Oak-D, Kinect, RealSense formats
- [ ] Implement `.xyz`, `.ply`, `.pcd` parsers (numpy-based)
- [ ] Memory-mapped storage for large point cloud sequences
- [ ] Time-alignment with RGB frames (`delta_timestamps` for depth)
- [ ] Depth transforms: downsampling, filtering, coordinate transforms
- [ ] LeRobot format extension: `observation.point_cloud` feature type

**Effort:** 80 hours  
**Owner:** Internal (camera format complexity moderate)  
**Deliverables:**
- Point cloud reader (ROS + vendor formats)
- Depth feature support in loader
- Aligned RGB+depth example notebook
- docs/DEPTH_CAMERAS.md

**Unblocks:**
- P1.5 (multimodal VLA), v1.0 "depth support" claim

---

#### **P1.4: Camera Calibration Registry** (2 weeks)
**Why:** Multi-camera alignment, 3D reconstruction, intrinsics/distortion tracking  
**What:**
- [ ] Store camera intrinsics/distortion in dataset metadata (per camera, per episode)
- [ ] Add `CameraCalibration` struct: K matrix, distortion coeffs, extrinsics
- [ ] Implement `loader(undistort=True)` option (OpenCV-based)
- [ ] Implement `loader(reproject=True)` for 3D point projection
- [ ] Add calibration validation: `ds.validate_calibration()`
- [ ] LeRobot format extension for calibration data

**Effort:** 40 hours  
**Owner:** Internal  
**Deliverables:**
- Camera calibration struct + storage
- Distortion correction API
- Calibration validation tool
- docs/CAMERA_CALIBRATION.md

**Unblocks:**
- P1.3 (depth alignment), P1.5 (multimodal), v1.0

---

#### **P1.5: Multimodal Integration — Fuse RGB + Depth + IMU** (2 weeks)
**Why:** Complete sensory fusion for humanoid robots (requires P1.3, P1.4)  
**Status:** Depends on P1.3 (depth) + P1.4 (calibration)  
**What:**
- [ ] Extend `RoboticsDataFrame.align()` to handle point clouds
- [ ] Add time-synchronized batches: RGB + depth + IMU at same timestamp
- [ ] Implement multimodal sampler (balanced across sensors)
- [ ] Add transforms for sensor fusion: project depth to image, fuse IMU
- [ ] Example: arm+gripper RGB + wrist depth + shoulder IMU → batch

**Effort:** 50 hours  
**Owner:** Internal (depends on P1.3, P1.4)  
**Deliverables:**
- Multimodal alignment API
- Sensor fusion transforms
- Humanoid example notebook
- docs/MULTIMODAL_FUSION.md

**Unblocks:**
- v1.0 "Full Humanoid" readiness

---

### 🟡 P2 — Important but Not Critical (v0.4.0+)

#### **P2.1: Streaming & Scale — Multi-node Distributed Loading** (3 weeks)
**Why:** Training on 1M+ frame datasets (S3/GCS streaming + Ray)  
**What:**
- [ ] Implement S3/GCS streaming backend (no local download, stream on-demand)
- [ ] Add Ray integration: distributed loader across multiple machines
- [ ] Implement prefetch queue over network boundary
- [ ] Benchmark: local vs remote vs cloud storage latency
- [ ] Add fault tolerance: retry + checkpointing for long training runs

**Effort:** 100+ hours  
**Owner:** Internal (distributed systems complexity high)  
**Deliverables:**
- S3/GCS loader implementation
- Ray distributed sampler
- Streaming benchmark
- docs/DISTRIBUTED_LOADING.md

**Blocks:** v1.0 "scale to 1M frames" claim

---

#### **P2.2: Additional Data Formats — RLDS / Open X-Embodiment** (2 weeks)
**Why:** Interop with HuggingFace Open X-Embodiment ecosystem  
**What:**
- [ ] Read RLDS (TFRecord format) from HF Hub
- [ ] Convert RLDS → LeRobot format
- [ ] Support Open X-Embodiment standardized features (state, action, etc.)
- [ ] Add RLDS roundtrip tests
- [ ] Document RLDS workflow

**Effort:** 60 hours  
**Owner:** Internal or community  
**Deliverables:**
- RLDS reader
- RLDS ↔ LeRobot converter
- docs/RLDS_SUPPORT.md

**Unblocks:** Ecosystem interop goals

---

#### **P2.3: HDF5 & NetCDF Support** (1 week)
**Why:** Legacy dataset compatibility, scientific computing  
**What:**
- [ ] Add HDF5 reader (h5py-based)
- [ ] Add NetCDF reader (xarray-based)
- [ ] Implement H5 → LeRobot converter
- [ ] Add roundtrip tests

**Effort:** 40 hours  
**Owner:** Community-friendly  
**Deliverables:**
- HDF5/NetCDF loaders
- Format converters
- docs/LEGACY_FORMATS.md

---

#### **P2.4: Throughput Benchmarking Suite** (1 week)
**Why:** Performance tracking, regression detection, marketing  
**What:**
- [ ] Publish detailed throughput benchmark vs LeRobot/Robo-DM/torchcodec
- [ ] Create continuous benchmarking (CI on each release)
- [ ] Add environment profiling (CPU, RAM, disk I/O, GPU)
- [ ] Create performance regression test suite
- [ ] Publish benchmark results dashboard

**Effort:** 40 hours  
**Owner:** Internal (performance measurement)  
**Deliverables:**
- benches/comparison_benchmark.py
- Performance regression tests
- docs/BENCHMARKING.md
- GitHub Pages performance dashboard

---

### 🟢 P3 — Nice to Have / Quality (v1.0+)

#### **P3.1: Data Quality Scoring Enhancements**
- [ ] Add motion blur detection (sharp vs blurry frames)
- [ ] Add scene diversity metrics (entropy, novelty)
- [ ] Add trajectory smoothness scoring
- [ ] Add gripper/end-effector visibility scoring

**Effort:** 60 hours  
**Owner:** Research-friendly

---

#### **P3.2: Curriculum Learning Helpers**
- [ ] Skill-level tagging (easy, medium, hard)
- [ ] Failure case detection & isolation
- [ ] Automated difficulty progression
- [ ] Docs: curriculum learning guide for RL training

**Effort:** 40 hours

---

#### **P3.3: Visualization & Debugging Tools**
- [ ] Dataset explorer web UI (play videos, filter by stats)
- [ ] Frame-by-frame debug viewer (show decode source, timing)
- [ ] Loader profiler visualization (bottleneck heatmap)
- [ ] Dataset diff tool (compare two datasets)

**Effort:** 80+ hours

---

## 📅 Proposed Timeline

```
v0.3.1 (GPU Verification Ready)       — 1 week     [P0.1 results + release]
├─ Depends on: P0.1 (GPU hardware testing)
└─ Unblocks: v0.4.0

v0.4.0 (Quality & Codecs)              — 4 weeks    [P1.2, P1.3, P1.4]
├─ Video codec selection (40% storage savings)
├─ Depth camera support (point clouds)
├─ Camera calibration registry
└─ Unblocks: v0.5.0

v0.5.0 (Multimodal Humanoid)           — 2 weeks    [P1.5, P2.1 start]
├─ Multimodal sensor fusion
├─ Humanoid example (RGB + depth + IMU)
└─ Unblocks: v1.0

v1.0 (Full Humanoid + Ecosystem)       — 3 weeks    [P0.2 + P2.x complete]
├─ MLX zero-copy (once mlx#2855 lands)
├─ Distributed loading (S3/GCS + Ray)
├─ Additional formats (RLDS, HDF5)
└─ Production-ready for large-scale training
```

**Timeline Estimate:** 10-12 weeks to v1.0 (assuming P0.1 completes in week 1-2)

---

## 🎯 Decision Points

### **Immediate (Next 1-2 weeks):**
1. **Do we rent GPU hardware ourselves, or crowdsource verification?**
   - Option A: Rent RunPod H100 (cost ~$100-200)
   - Option B: Wait for community contributions
   - **Recommendation:** Option A (faster, reliable, unblocks v0.3.1)

2. **Do we implement P0.2 (MLX zero-copy) proactively, or wait for mlx#2855?**
   - Option A: Start architecture now, land once upstream ready
   - Option B: Wait for mlx#2855 to land, then implement
   - **Recommendation:** Option A (design work can proceed in parallel)

### **Sequencing (Order matters):**
- **Must do in order:** P1.3 → P1.4 → P1.5 (depth → calibration → fusion)
- **Can parallelize:** P1.1, P1.2 (independent)
- **Can defer:** P2.x, P3.x (nice-to-have)

---

## 📊 Effort Estimation

| Priority | Task | Effort | Impact | Timeline |
|----------|------|--------|--------|----------|
| **P0.1** | GPU Verification | 40-60h | **CRITICAL** | Week 1-2 |
| **P0.2** | MLX Zero-Copy | 30-40h | **CRITICAL** | Week 3-4 (if mlx#2855 lands) |
| **P1.1** | v0.3.1 Release | 20h | **HIGH** | Week 2 |
| **P1.2** | Codec Selection | 60h | **HIGH** | Week 3-4 |
| **P1.3** | Depth Cameras | 80h | **HIGH** | Week 5-7 |
| **P1.4** | Calibration | 40h | **HIGH** | Week 6-7 |
| **P1.5** | Multimodal Fusion | 50h | **HIGH** | Week 8-9 |
| **P2.1** | Distributed Loading | 100+h | MEDIUM | Week 9-11 |
| **P2.2** | RLDS Support | 60h | MEDIUM | Week 10-11 |
| **P2.3** | HDF5/NetCDF | 40h | MEDIUM | Week 11+ |
| **P2.4** | Benchmarking Suite | 40h | MEDIUM | Week 8+ |

**Total (all priorities):** ~580 hours (14-15 weeks at 40h/week)  
**Total (P0+P1 critical path):** ~290 hours (7-8 weeks)

---

## 🚦 Success Criteria per Batch

### **v0.3.1 Success:**
- ✅ GPU benchmarks published (actual performance numbers)
- ✅ Known-good config matrix documented
- ✅ README updated with verified speedups
- ✅ Release notes highlight GPU verification

### **v0.4.0 Success:**
- ✅ Codec selection saves 40%+ storage (measured)
- ✅ Depth + RGB loader works end-to-end
- ✅ Camera calibration API stable
- ✅ Example notebook (humanoid arm)

### **v0.5.0 Success:**
- ✅ Multimodal batches (RGB + depth + IMU) working
- ✅ Humanoid training example runs
- ✅ Sensor fusion transforms tested

### **v1.0 Success:**
- ✅ MLX zero-copy live (3× speedup verified)
- ✅ Distributed loading (S3 + Ray) working
- ✅ Multiple format support (LeRobot + RLDS + HDF5)
- ✅ Production-ready for large-scale training
- ✅ Benchmarks published vs incumbents
- ✅ Humanoid training guide

---

## 📝 Recommended Next Steps

**Week 1:**
1. [ ] Decide on GPU verification approach (P0.1)
2. [ ] Set up RunPod/cloud GPU if decided
3. [ ] Start MLX zero-copy design doc (P0.2)
4. [ ] Plan v0.3.1 release checklist (P1.1)

**Week 2:**
1. [ ] GPU benchmarks running
2. [ ] P0.1 results coming in
3. [ ] v0.3.1 release candidate

**Week 3:**
1. [ ] v0.3.1 shipped
2. [ ] Start codec selection (P1.2)
3. [ ] Start depth camera format research (P1.3)

---

**Questions to address before proceeding:**
- Who will execute P0.1 (GPU verification)?
- Should we rent GPU hardware or seek community contributions?
- Do we wait for mlx#2855 or start MLX zero-copy design now?
- What's the priority: breadth (formats, features) or depth (performance)?
