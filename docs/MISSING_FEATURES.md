# PyRoboFrames: Critical & Good-to-Have Missing Features

Comprehensive audit of gaps vs. robot-learning requirements and competing tools (LeRobot, DALI, torchcodec, Vaex).

---

## 🔴 CRITICAL (needed for production datasets >1GB)

### 1. **Incremental append / dataset versioning**
- **Gap:** No way to add new episodes to existing dataset without rewriting entire Parquet
- **Impact:** Production datasets grow; must support `dataset_v1 + new_episodes → dataset_v2`
- **Current:** Save creates new dataset; no merge/append
- **Effort:** M (Parquet row-group append, metadata rollup)
- **Why it matters:** LeRobot v3.0 expects versioning; robotics datasets evolve with retraining

### 2. **Delta encoding for state/action columns**
- **Gap:** Full state values stored; no compression for near-constant or slowly-varying signals
- **Impact:** 10M-frame datasets waste storage on 99% redundant joint positions
- **Current:** NumPy/Parquet defaults (no delta encoding)
- **Effort:** S (write delta at save time, decode on read)
- **Why it matters:** DALI + torchcodec both do this; 30–50% storage savings typical

### 3. **Batched augmentation / prefetch augmentation**
- **Gap:** Transforms only; no on-the-fly augmentation (rotation, crop, noise, etc.)
- **Impact:** Must augment offline or implement per-model → no unified pipeline
- **Current:** Only `transforms.Resize`, `Normalize`, `Crop`, `RandomCrop`, `RandomHorizontalFlip`
- **Effort:** M (augmentation suite + feature flags for enable/disable)
- **Why it matters:** VLA models (ACT, Diffusion Policy) train on heavily augmented data

### 4. **Episode filtering / dynamic subsampling**
- **Gap:** Fixed dataset at load time; no runtime filtering by metadata (e.g., episode_length > 100)
- **Impact:** Curriculum learning needs to filter by quality score, task ID, or success flag
- **Current:** Train/val split only; no conditional sampling
- **Effort:** S (Python predicate filtering in Loader)
- **Why it matters:** Curriculum learning, quality-gated training, task-specific splits

### 5. **Distributed data loading (multi-GPU, multi-machine)**
- **Gap:** Single-machine prefetch pipeline; no sharding across workers
- **Impact:** Training on RTX 5090 + A100 in same job; wasted GPU cycles
- **Current:** `num_workers` is threading-only (off-GIL); no Ray/multiprocessing
- **Effort:** L (Ray/PyTorch distributed sampler)
- **Why it matters:** 100M-frame datasets need N×throughput on N GPUs

### 6. **Quality scoring / trajectory filtering**
- **Gap:** `validate()` checks data integrity; no scoring (diversity, sharpness, state-variance)
- **Impact:** Can't identify/skip low-quality episodes before training
- **Current:** `ds.stats()` has summary stats; no per-episode quality metrics
- **Effort:** M (implement scorers: diversity, sharpness via Laplacian, action-rank)
- **Why it matters:** 30–50% of demonstration data is low-quality; filtering saves GPU hours

---

## 🟡 HIGH PRIORITY (ship before v1.0)

### 7. **Sparse/masked episode support**
- **Gap:** Every frame must have every sensor; no optional/missing-data handling
- **Impact:** Real robots have sensor failures, gaps; must handle gracefully
- **Current:** Assumes dense [N, features] Parquet; no masking or interpolation
- **Effort:** M (mask arrays, optional interpolation modes)
- **Why it matters:** LeRobot supports optional features; real-world robustness

### 8. **Episode-level metadata querying**
- **Gap:** No `ds.filter(task="pick", success=True)` or `ds.episodes_where(...)`
- **Impact:** Must load dataset into Pandas to filter; slow, memory-hungry
- **Current:** `ds.episodes()` lists all; no filtering API
- **Effort:** S (Parquet metadata push-down, SQL-like query)
- **Why it matters:** Task-conditional training, failure analysis

### 9. **Streaming ingestion (MQTT/Kafka)**
- **Gap:** Designed for static datasets; can't ingest live telemetry streams
- **Impact:** Can't use as online learner feeder or closed-loop data collector
- **Current:** P7 deferred (MQTT/Kafka connectors)
- **Effort:** L (stream batcher, windowing, state checkpointing)
- **Why it matters:** Online learning, continual learning pipelines

### 10. **Native video codec selection (H.264, HEVC, AV1)**
- **Gap:** Hard-coded YUV420p MP4; no codec choice
- **Impact:** HEVC 40% smaller; AV1 50% smaller (vs H.264)
- **Current:** FFmpeg → MP4 (H.264 default); no flag to choose codec
- **Effort:** S (add `--codec` flag to converter; encoder fallback)
- **Why it matters:** Multi-TB datasets; storage cost matters

### 11. **Deterministic reproducibility controls**
- **Gap:** Seeding only for sampling; no guarantee on frame extraction order
- **Impact:** Checkpointed runs may resume with different batches (data shuffling)
- **Current:** `seed` parameter on Loader; shuffle may vary on resume
- **Effort:** S (RNG state + loader position tracking)
- **Why it matters:** Hyperparameter search, reproducible research

### 12. **Keras/TensorFlow adapter**
- **Gap:** NumPy/MLX/Torch/JAX; no TensorFlow/Keras
- **Impact:** TensorFlow users must wrap manually
- **Current:** `output="numpy"` only; no `tf.data.Dataset` bridge
- **Effort:** S (tf.data wrapper, tensor conversion)
- **Why it matters:** TensorFlow still used in industry (especially robotics)

---

## 🟠 GOOD-TO-HAVE (backlog, ship in v0.2+)

### 13. **Dataset lineage tracking**
- **Gap:** No record of "frame X came from this raw log, via this converter, at this timestamp"
- **Impact:** Debugging data issues (e.g., "which episodes used old sensor?"); compliance
- **Current:** `metadata.json` has converter/version; no per-episode provenance
- **Effort:** M (annotate during convert; embed in Parquet metadata)
- **Why it matters:** Data provenance for safety-critical (e.g., autonomous manipulation)

### 14. **Real-time statistics / online profiling**
- **Gap:** `ds.stats()` computes from entire dataset; no running stats
- **Impact:** Can't monitor dataset quality during live data collection
- **Current:** Post-hoc `stats.json` only
- **Effort:** M (HyperLogLog for cardinality, Welford online variance)
- **Why it matters:** Detects sensor drift, data degradation live

### 15. **Multi-dataset joining / cross-dataset episodes**
- **Gap:** Cannot combine LeRobot dataset A + dataset B for training
- **Impact:** Transfer learning across robot embodiments needs data fusion
- **Current:** Load separately; manual concatenation in training loop
- **Effort:** L (unified schema, episode index mapping)
- **Why it matters:** Foundation models for robotics require multi-source training

### 16. **Interactive dataset explorer (web UI)**
- **Gap:** No visual exploration; users must write Jupyter scripts
- **Impact:** Slow iteration on data understanding
- **Current:** `ds.stats()` and `ds.validate()`; no visualizer
- **Effort:** L (streamlit/fastapi + frame viewer, timeseries plots)
- **Why it matters:** Better UX; faster data debugging

### 17. **Compression benchmarking toolkit**
- **Gap:** Can't compare storage vs. speed for H.264 vs HEVC vs AV1
- **Impact:** Users guess on codec; can't optimize for their bandwidth budget
- **Current:** Single converter; no benchmarking
- **Effort:** M (multi-codec benchmark harness, reporting)
- **Why it matters:** Storage-constrained deployments (edge robots)

### 18. **Field-of-view correction / camera calibration registry**
- **Gap:** No support for camera intrinsics; can't undistort, rectify, reproject
- **Impact:** Multi-camera datasets without calibration are hard to align
- **Current:** Assumes frames as-is; no calibration data storage
- **Effort:** M (calibration schema, distortion correction transform)
- **Why it matters:** Precise multi-camera setups (industrial manipulation)

### 19. **Depth camera (point cloud) support**
- **Gap:** Video (RGB) only; no depth, LiDAR, or structured light
- **Impact:** Humanoid hands, complex grasping need depth; RGB-only limits tasks
- **Current:** Designed for RGB video
- **Effort:** L (PointCloud type, memory-mapped .ply/.npz storage, transforms)
- **Why it matters:** Modern robot sensors (Oak-D, Kinect, etc.) have depth

### 20. **Action-space compatibility validation**
- **Gap:** No checking that action shapes/ranges match robot hardware
- **Impact:** Loading bad data into sim/robot → crashes, unsafe training
- **Current:** No validation beyond schema
- **Effort:** S (action bounds schema, validation at load time)
- **Why it matters:** Safety: prevent OOB actions from reaching hardware

### 21. **Efficient random access by timestamp**
- **Gap:** Can seek to frame index; can't seek to timestamp (e.g., "get frame at t=12.5s")
- **Impact:** Aligning multi-sensor data requires binary search on timestamps
- **Current:** Index on episode/frame only
- **Effort:** S (timestamp index in Parquet metadata)
- **Why it matters:** Cross-sensor fusion (vision + IMU + force)

### 22. **Memory profiling / peak-usage analysis**
- **Gap:** No tools to measure peak RAM during epoch
- **Impact:** Can't predict if large dataset fits in available memory
- **Current:** Stats show total rows; no per-operator memory tracking
- **Effort:** M (context manager + memory hooks via tracemalloc)
- **Why it matters:** Deployment on memory-constrained hardware (embedded robots)

### 23. **Trajectory-level metadata (task, success, duration)**
- **Gap:** Episode metadata only; no per-trajectory (goal) labels
- **Impact:** Goal-conditioned learning needs trajectory outcome + task tag
- **Current:** Episode-level stats/metadata; no trajectory buckets
- **Effort:** S (optional trajectory index in episode metadata)
- **Why it matters:** Goal-conditioned + multi-task learning

### 24. **Plugin system for custom loaders/converters**
- **Gap:** Adding new format requires forking PyRoboFrames
- **Impact:** Roboticists with proprietary formats can't extend easily
- **Current:** Converters are built-in; no plugin registry
- **Effort:** M (entry-point registration, loader interface)
- **Why it matters:** Ecosystem extensibility

---

## 📊 Summary: By Category

### **Correctness & Safety** (critical)
- ✅ Quality scoring (episode filtering)
- ✅ Action-space validation
- ✅ Sparse/masked data support
- ✅ Deterministic reproducibility

### **Performance & Scale** (critical)
- ✅ Incremental append / versioning
- ✅ Delta encoding for compression
- ✅ Distributed data loading
- ✅ Multi-GPU sharding

### **Usability** (high-priority)
- ✅ Episode metadata querying
- ✅ Augmentation pipeline
- ✅ Interactive explorer
- ✅ Keras/TensorFlow adapter

### **Robustness** (good-to-have)
- ✅ Streaming ingestion (live data)
- ✅ Lineage tracking (provenance)
- ✅ Camera calibration (intrinsics)
- ✅ Depth camera support

### **Analytics** (good-to-have)
- ✅ Real-time statistics
- ✅ Memory profiling
- ✅ Compression benchmarking
- ✅ Cross-dataset joining

---

## 🎯 Recommended Priority

### **v0.2 (next major)** — Unblock production use (3–4 items)
1. **Incremental append** — needed for evolving datasets
2. **Quality scoring** — needed for data curation
3. **Episode filtering** — unlocks curriculum learning
4. **Distributed loading** — unlocks multi-GPU training

### **v0.3** — Full robustness (4–5 items)
5. **Sparse/masked data** — real-world sensor handling
6. **Streaming ingestion** — online learning
7. **Batched augmentation** — training best practices
8. **Keras/TensorFlow** — ecosystem support

### **v1.0** — Production-ready (remaining items)
- Depth support, calibration, lineage, explorer, interop with other frameworks

---

## 🔗 Compatibility Notes

- **LeRobot v3.0:** Supports all features here; PyRoboFrames should reach feature parity
- **DALI (NVIDIA):** Has delta encoding, augmentation; PyRoboFrames lacks both
- **torchcodec:** Torch-native codecs, prefetch augmentation; PyRoboFrames uses FFmpeg
- **Vaex:** Out-of-core Pandas; PyRoboFrames could adopt similar lazy evaluation
