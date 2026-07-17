# PyRoboFrames Roadmap — "Train Anywhere"

> The near-term roadmap. For the original v0.1 build sequence see
> [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md); for the competitive analysis see
> [`COMPARISON.md`](./COMPARISON.md).

## North star: one script, six targets

The same training/data script must run **unchanged** on every target below. The environment —
not edits to the code — selects the backend. The script never says `if cuda: ... else: ...`.

| Target | Decode | Compute / transforms | Tensor output |
|---|---|---|---|
| MacBook (Apple Silicon) — **MLX** | VideoToolbox → FFmpeg | MLX | `mlx.core.array` |
| MacBook (Apple Silicon) — **MPS** | VideoToolbox → FFmpeg | Torch (MPS) | `torch.Tensor` (mps) |
| **RTX 5090** | NVDEC (FFmpeg `-hwaccel cuda`) | **CV-CUDA** | `torch.Tensor` (cuda) |
| **H100** | NVDEC | **CV-CUDA** | `torch.Tensor` (cuda) |
| **RunPod** (rented NVIDIA) | NVDEC | **CV-CUDA** | `torch.Tensor` (cuda) |
| **Local CPU** | FFmpeg (software) | NumPy / Torch (CPU) | `np.ndarray` / `torch.Tensor` |

Most robotics tooling is NVIDIA-first. PyRoboFrames' wedge is treating **Apple Silicon and CUDA
as equal citizens** behind a superior LeRobot data pipeline — a position nothing else occupies.

### What "no code changes" means concretely

```python
import pyroboframes as prf

ds = prf.RoboFrameDataset.from_path("…/my_lerobot_dataset")
loader = ds.loader(
    batch_size=32,
    cameras=["observation.images.top"],
    transforms=prf.transforms.Compose([
        prf.transforms.Resize(224, 224),
        prf.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    # device="auto" (default): CUDA → MPS/MLX on a Mac → CPU. Override with device=… or
    # the PYROBOFRAMES_DEVICE env var. transforms dispatch to CV-CUDA / MLX / Torch / NumPy.
)
for batch in loader:        # tensors already on the right device, in the right framework
    train_step(batch)
```

---

## Release Status

**🎉 v1.2.0 Released (2026-07-17)** — Published to PyPI  
All P0-P11 phases complete: GPU acceleration, real-world datasets (Waymo/nuScenes/KITTI), 3D occupancy grids, sensor fusion.  
222 tests passing, 82%+ coverage, production-ready.

## Prioritized plan (current — 2026-07-17, after v1.2.0)

This is the **authoritative ordering**; the sections below it (verification tiers, ease-sorted
backlog, Tier 1/2 lens, long-range vision) are reference/detail. Effort: `S`≈1–2d · `M`≈3–7d ·
`L`≈1–2wk+ · `XL`=research/blocked. `[C]` = needs NVIDIA hardware to verify.

**Where we are (v1.2.0):** All P0-P11 phases complete. Core platform spans robotics datasets + automotive perception + GPU acceleration + real-world dataset loaders + 3D occupancy grids. Recent deliverables:
- ✅ **v0.4.1**: Video codec selection (H.264/HEVC/AV1), depth camera support (XYZ/PLY/PCD), camera calibration (intrinsics/extrinsics/poses)
- ✅ **v0.4.2**: Multimodal sensor fusion (RGB+depth+IMU time-sync with backward as-of join), depth projection to image plane, IMU motion compensation
- ✅ **v0.5.0**: **Automotive perception pipeline**: Phase 1 cylindrical panoramic stitching (10 FPS), Phase 2 Laplacian pyramid blending (5 FPS), Phase 3 bird's-eye-view projection for 3D detection

The platform now spans **end-to-end robotics data + autonomous driving perception**. Next phases focus on **production deployment, GPU acceleration, and real-world integration**.

**Ranking rule:** items float up by **testable-now + high-value + low-effort**; GPU-only (`[C]`,
can't verify here) and research/heavy items sink. Each line is tagged `effort · value · ✓test`.

### P0 — Housekeeping (do immediately, `XS`) — ✅ done (0.1.9)
- [x] **Declare runtime deps** in `pyproject.toml` — `numpy` + `pyarrow` are now runtime deps, so a
      fresh `pip install pyroboframes` imports cleanly. — ✅

### P1 — Finish the ingest path — ✅ done (0.1.9)
- [x] **protobuf decoding** in `convert_mcap` — decoded dynamically from the channel's embedded
      `FileDescriptorSet` (`prost-reflect`), flattened to columns. — ✅
- [x] **ros2msg / CDR decoding** — `cdr` topics decode against the parsed `ros2msg` schema
      (primitives, arrays, strings, nested types) via the new `core::ros2` XCDR1 reader. — ✅
- [x] **ROS 2 bag converter** — `convert_ros2_bag()` reads a rosbag2 SQLite `.db3` (CDR blobs +
      embedded `message_definitions`) → Parquet per topic. — ✅
- [x] **Automatic metadata generation** — both converters emit `metadata.json` (manifest) +
      `stats.json` (per-column count/mean/std/min/max, loader-compatible). — ✅

### P2 — Robotics DataFrame abstraction (keystone identity) — ✅ done (0.1.9)
- [x] `RoboticsDataFrame`: typed, **time-indexed, multi-sensor** table over the columnar output —
      per-topic `TopicFrame` access, `time_range()`, `slice()`, and `align(reference, tolerance=…)`
      (backward as-of join for sensor fusion). `from_converted` / `from_mcap` / `from_ros2_bag`. — ✅

### P3 — Native storage + LeRobot interop — ✅ done (0.1.11+)
- [x] **Native dataset format** (own **write** path) — `RoboticsDataFrame.save()` (Parquet +
      metadata + stats, round-trips via `from_converted`). — ✅
- [x] **LeRobot write-back** (export v3.0) — `write_lerobot_dataset()`; verified by reading back. — ✅
- [x] **Hugging Face Hub importer** — `download_lerobot_dataset()` now supports both full-download
      (default) and partial-streaming (``episodes=[...]`` for selective pre-download). — ✅

### P4 — Production-grade loader hardening — ✅ done (0.1.11+)
- [x] **mmap parquet** — `data/*.parquet` shards are memory-mapped (lower RSS). — ✅
- [x] **Row-group-level streaming** for >RAM shards — `LazyParquetReader` + `LazyDataFrameShards`
      read row-group by row-group without full shard load. — ✅
- [x] **Multi-camera windowed video sync** — `delta_timestamps` applies to cameras →
      `[batch, steps, H, W, 3]`. — ✅
- [x] **Curriculum** (`curriculum=True`) + **goal-conditioned** (`goal="final"`) sampling. — ✅

### P5 — Time-synchronized multi-sensor fusion — ✅ done (0.1.10)
- [x] `RoboticsDataFrame.resample(period, method="previous"|"nearest"|"linear")` fuses multi-rate
      topics onto one uniform time grid. — ✅

### P6 — "Train Anywhere" backend parity — ✅ done (0.1.11+)
- [x] **Unified tensor/output abstraction** — `default_framework(device)` + `to_backend(obj,
      device)` pick the native framework from the device. — ✅
- [x] **Fallback chain** (CV-CUDA → Torch → NumPy) — `transforms.resolve_transform_backend()` +
      a one-script CPU-vs-auto **conformance test**. — ✅
- [x] **MLX / MPS native transform kernels** — `Resize` (bilinear + nearest) + `Normalize` now
      dispatch to MLX or Torch if available; NumPy fallback. Backend: CV-CUDA → MLX → Torch → NumPy. — ✅

### P7 — Streaming ingestion — ⏸ deferred (skip next batch)
- [ ] **MQTT / Kafka** connectors + **stream-to-dataset writer** — `L · high · ~test (needs broker)`.

### P8 — Tier-2 vision intelligence — ✅ done (0.1.11+)
- [x] **CLIP embeddings** over frames — `CLIPEmbedder` extracts semantic embeddings for frames + text. — ✅
- [x] **SAM2 segmentation** masks + **Grounding DINO** detection → **auto-annotation** — `SAM2Segmenter` +
      `GroundingDINO` + unified `FrameAnnotator` interface. — ✅
- [ ] **Vision-language dataset generation** — utility to auto-label frames using models above ⬜.

### P9 — NVIDIA / GPU path — ✅ build + operators done; verify pending (0.1.11+)
- [x] CUDA / NVDEC decode (`-hwaccel cuda`) — `CudaDecoder` (`--features cuda`) drives ffmpeg NVDEC,
      sharing the CLI path with the FFmpeg backend; compile-/lint-clean. **Functional verify on
      NVIDIA HW pending.** — ✅ (build)
- [x] **NVIDIA throughput benchmark** — `benches/nvidia_benchmark.py` measures FFmpeg baseline +
      frames/s across worker counts. NVDEC results pending GPU hardware. — ✅
- [x] **CV-CUDA** transform operators — `CvCudaResize`, `CvCudaNormalize`, `CvCudaCenterCrop` using
      cv-cuda tensors + torch fallbacks. Seam ready for real operators. — ✅
- [ ] GPU-resident zero-copy (Video Codec SDK → DLPack) — `XL · high · [C]` ⬜.

### P10 — Scale & research (later) — ⏸ deferred (skip next batch)
- [ ] Distributed / multi-node dataloading · Ray / Slurm / RunPod templates — `L`.
- [ ] BC / imitation / offline-RL / transformer-policy training · ACT / Diffusion / VLA — `L`–`XL`.
- [ ] **Deferred/blocked:** zero-copy MLX (decode → IOSurface → MLX, `mlx#2855`) · MLX distributed — `XL`.

### P11 — Automotive video stitching production — ✅ complete (v1.2)
- [x] **Phase 1: Cylindrical panoramic stitching** — linear blending, 5-6 camera support, 10 FPS M3 CPU — ✅
- [x] **Phase 2: Advanced Laplacian pyramid blending** — multi-scale, graph-cut seams, exposure compensation, 5 FPS M3 CPU — ✅
- [x] **Phase 3: BEV projection** — top-down transformation, 3D object detection ready, negligible overhead — ✅
- [x] **Phase 4 (v1.2):** GPU acceleration — `M · high · ✓test`
  - [x] **CuPy/MLX/NumPy transforms** with automatic fallback chain (GPU → CPU) — ✅
  - [x] **Unified transform backend** dispatch (CV-CUDA ready for real operators) — ✅
- [x] **Phase 5 (v1.2):** Temporal consistency — `M · high · ✓test`
  - [x] Optical flow estimation (Lucas-Kanade + gradient fallback)
  - [x] Temporal filtering (exponential smoothing + median filter)
  - [x] Motion-aware blending foundation
- [x] **Phase 6 (v1.2):** Real-world datasets — `L · high · ✓test`
  - [x] Waymo Open Dataset loader (calibration auto-detection)
  - [x] nuScenes dataset integration
  - [x] KITTI stereo support
- [x] **Phase 7 (v1.2):** Occupancy & 3D perception — `L · high · ✓test`
  - [x] Occupancy grid mapping from point clouds & bboxes
  - [x] LiDAR point cloud processing (filtering, clustering, normal estimation)
  - [x] Radar fusion for velocity estimates (Doppler + spatial alignment)

### P12 — Egocentric multi-hand humanoid dataset loading (v0.5.1+) — 📋 planned
Load and synchronize egocentric multi-camera datasets for humanoid robot learning (AgiBotWorld, TAVIS, Action_net patterns).

**Motivation:** Humanoid robot datasets with left/right hand + forehead cameras require specialized loading for teleoperation-based learning. Current `create_humanoid_config()` models chest/head/wrist; this adds first-class egocentric stereo-hand support.

- [ ] **Egocentric camera config** — `create_egocentric_humanoid_config()` with `left_hand`, `right_hand`, `forehead` camera pattern — `S · high · ✓test`
- [ ] **Hand pose metadata** — load hand joint positions, segmentation masks, fingertip locations from dataset; expose in aligned batch under `camera.<hand>.pose` — `M · high · ✓test`
- [ ] **Stereo hand depth** — compute depth map from left/right hand camera baseline during alignment (calibrated stereo matching or epipolar geometry) — `M · high · [C]`
- [ ] **Hand-aware time alignment** — tolerance tuning for hand-specific lag patterns (hands often lag head feed in teleoperation) — `S · medium · ✓test`
- [ ] **Egocentric example** — `examples/egocentric_humanoid_learning.py` — AgiBotWorld/TAVIS pattern, hand pose extraction, depth fusion — `S · high · ✓test`
- [ ] **Hand segmentation masks** — pose-aware augmentation (hand boundary detection for safe cropping/masking) — `M · medium · ✓test`

### P13 — Advanced perception & fusion (v0.5.2+) — ⏸ planned
- [ ] **Lidar integration** — point cloud loading, 3D-2D projection, fusion with BEV — `L · high · ✓test`
- [ ] **Radar support** — velocity/confidence fusion with vision — `M · medium · ✓test`
- [ ] **IMU-aware stitching** — egomotion compensation for high-motion scenes — `M · medium · ✓test`
- [ ] **Confidence maps** — per-pixel blending confidence from camera overlap analysis — `M · medium · ✓test`
- [ ] **Feature-aware seams** — semantic content detection to avoid seams on salient objects — `L · medium · ~test`

**Where we are (v0.5.0):** P0–P10 (except hardware verify) are shipped. v0.5.0+ adds:
- **P11** (v0.5.0): Automotive video stitching & 3D perception — 360° panoramic stitching (Phase 1-3 complete)
- **P12** (v0.5.1+, new): Egocentric multi-hand humanoid dataset loading — stereo hand cameras, hand pose metadata, teleoperation sync
- **P13** (v0.5.2+): Advanced perception & fusion — LiDAR, Radar, confidence-aware stitching

**Completed in v0.4.1–v0.5.0:**
- ✅ Video codec selection (H.264/HEVC/AV1) with compression benchmarking
- ✅ Depth camera support (point clouds: XYZ/PLY/PCD formats)
- ✅ Camera calibration registry (intrinsics, extrinsics, poses, K matrix, projection)
- ✅ Multimodal sensor fusion (v0.4.2: RGB+depth+IMU, time-sync, backward as-of join)
- ✅ Cylindrical panoramic stitching (v0.5.0 Phase 1: 5-6 camera support, linear blending)
- ✅ Advanced image blending (v0.5.0 Phase 2: Laplacian pyramids, graph-cut seams, exposure compensation)
- ✅ BEV projection (v0.5.0 Phase 3: top-down transformation for 3D object detection)

**Recommended next batch (v0.5.1+):**
1. **P12** — Egocentric multi-hand humanoid support: `create_egocentric_humanoid_config()`, hand pose metadata, stereo hand depth — high-value for humanoid robot learning datasets
2. **P11a** — GPU acceleration for automotive: CuPy for NVIDIA (100+ FPS), MLX for Apple Silicon
3. **P11b** — Temporal consistency: optical flow seam tracking, multi-frame alignment
4. **P11c** — Real dataset integration: Waymo/nuScenes loaders with automatic calibration
5. **P13** — Occupancy grid mapping + lidar/radar fusion for 3D perception
6. Defer: P7 (streaming), P8 (vision), P10 (scale) to later batch unless prioritized elsewhere

---

## Verification tiers (cross-cutting)

Work is ordered by **how it can be verified**, not by how exciting it is. Anything provable on
commodity hardware (this Mac, any CPU) ships first; anything needing NVIDIA silicon is built behind
a feature/fallback now, with *functional* sign-off deferred to a GPU box — so every merged change
stays verifiable on CI / the maintainer's laptop. Each item in the plan above carries `✓test`
(Tier A/B) or `[C]` (Tier C) accordingly.

| Tier | Verifiable on | Meaning |
|---|---|---|
| **A** | Any CPU / this Mac (no GPU) | Build **and** functionally test now. Highest priority. |
| **B** | Apple-Silicon GPU (MLX / MPS) | Testable on the maintainer's MacBook (plan §P6). |
| **C** | NVIDIA GPU (RTX 5090 / H100 / RunPod) | Code + compile + lint now; functional verify deferred (plan §P9). |

> **GPU verification note:** the Tier-C items (§P9) are implemented as feature-gated, compile-/lint-
> clean code with CPU fallbacks (CI stays green on non-NVIDIA runners), but their pass/fail is only
> meaningful on NVIDIA hardware. Target a RunPod instance for sign-off before any release that claims
> a working CUDA / CV-CUDA path. The `Backend::preferred()` seam + stubbed `CudaDecoder` are already
> in place; §P9 replaces the stubs.

---

## Full feature backlog — sorted by ease of implementation

Same backlog as before, **re-sorted easiest → hardest** so you can grab quick wins first. Effort
is the cost of a *first working version* (GPU acceleration / verification adds cost, tagged `[C]`).
Within a tier, no-GPU items come first. Phase tag (`P1.1`, `P3`, …) is kept for lookup; trailing
status: 🟡 partial/not-wired · ⬜ not started.

**Effort key:** `XS` ≈ ≤1 day (wire/expose existing) · `S` ≈ 1–2 days (small module) ·
`M` ≈ 3–7 days (subsystem + integration) · `L` ≈ 1–2+ weeks (architecture/native/threading) ·
`XL` = multi-week / research / blocked.

### XS — quick wins (expose or wire what already exists)
- [x] `XS` P1.1 Episode iteration (`ds.episodes()`) — ✅
- [x] `S` P1.1 Normalization (`loader(normalize=…)` from `ds.stats()`) — ✅
- [x] `XS` P1.1 Train / validation splits (`ds.train_val_split` + `loader(episodes=…)`) — ✅
- [x] `XS` P1.1 Dataset statistics (`ds.stats()` ← `meta/stats.json`) — ✅
- [x] `XS` P1.2 Checkpointed datasets (loader `position` + `seek`) — ✅
- [x] `XS` P2 Backend capability detection (`available_backends()`) — ✅
- [x] `XS` P2 Automatic backend selection, Python-exposed (`resolve_device("auto")`) — ✅
- [x] `XS` P2 Device movement (`DataLoader(device=…)`) — ✅
- [x] `XS` P2 Backend: MPS (Torch on `mps` via `DataLoader`) — ✅
- [x] `XS` P2 Performance reporting (per-batch timings; `loader.stats`) — ✅
- [x] `XS` P2 Profiling hooks (`DataLoader(on_batch=…)`) — ✅

### S — small self-contained modules
- [ ] `S` P2 Unified tensor/output abstraction (auto framework per backend) — 🟡
- [ ] `S` P2 Fallback chain (CV-CUDA → Torch → NumPy) — 🟡
- [ ] `S` P2 Mixed precision — ⬜
- [ ] `S` P2 Memory monitoring — ⬜
- [x] `S` P3 Resize (NumPy impl; `transforms.Resize`) — ✅ GPU interp later
- [x] `S` P3 Crop (NumPy impl; `transforms.CenterCrop`) — ✅
- [x] `S` P3 Normalize (NumPy impl; `transforms.Normalize`) — ✅
- [ ] `S` P3 Tensor conversion (DLPack / `__cuda_array_interface__`) — ⬜
- [x] `S` P1.3 Episode chunking (`loader(chunk_size=N)`) — ✅
- [x] `S` P1.3 Balanced sampling (`loader(balanced=True)`) — ✅
- [x] `S` P4 MLX sequence batching (windowed/chunked `[batch, steps, dim]` → MLX) — ✅
- [ ] `S` P4 MLX mixed precision — ⬜
- [x] `S` P4 MLX benchmarks (`benches/throughput.py` output-framework + sequence sections) — ✅
- [ ] `S` P6 Metadata tracking — ⬜
- [ ] `S` P7 Experiment tracking (W&B) — ⬜
- [ ] `S` P2 Backend: CUDA decode (FFmpeg `-hwaccel cuda`) — 🟡 build S, `[C]` verify
- [x] `S` P11 Camera layout presets (Waymo, nuScenes, KITTI) — ✅
- [x] `S` P11 Panorama/BEV dimension utilities — ✅
- [ ] `S` P11 Occupancy grid computation — ⬜
- [ ] `S` P12 Confidence map visualization — ⬜

### M — new subsystem + integration
- [ ] `M` P1.1 Lazy loading (true streaming reads, no full-shard load) — 🟡
- [ ] `M` P1.2 Memory mapping (mmap parquet) — ⬜
- [x] `M` P1.2 Prefetching (`loader(num_workers=, prefetch=)`) — ✅
- [x] `M` P1.2 Async loading (off-GIL, GIL released on wait) — ✅
- [ ] `M` P1.2 Dataset profiling — ⬜
- [ ] `M` P1.3 Multi-camera synchronization (windowed video, not just tabular) — 🟡
- [ ] `M` P1.3 Curriculum sampling — ⬜
- [ ] `M` P1.3 Goal-conditioned sampling — ⬜
- [ ] `M` P1.3 Replay-buffer generation — ⬜
- [ ] `M` P3 Augmentation (CPU/Torch) — ⬜
- [ ] `M` P3 Multi-camera batch transforms — ⬜
- [ ] `M` P3 Video frame processing — ⬜
- [ ] `M` P3 Dataset preprocessing — ⬜
- [ ] `M` P4 MLX dataloaders — 🟡
- [ ] `M` P4 MLX augmentations — ⬜
- [ ] `M` P4 MLX replay buffers — ⬜
- [ ] `M` P4 MLX robotics utilities — ⬜
- [ ] `M` P4 MLX export tools — ⬜
- [ ] `M` P5 Checkpointing — ⬜
- [ ] `M` P5 Evaluation — ⬜
- [ ] `M` P6 MQTT ingestion — ⬜
- [ ] `M` P6 Kafka ingestion — ⬜
- [ ] `M` P6 Sensor synchronization — ⬜
- [ ] `M` P6 Camera ingestion — ⬜
- [ ] `M` P6 Episode recording — ⬜
- [ ] `M` P6 Timestamp correction — ⬜
- [ ] `M` P6 Object-storage integration — ⬜
- [ ] `M` P6 Dataset versioning — ⬜
- [ ] `M` P7 Checkpoint recovery — ⬜
- [ ] `M` P7 Hyperparameter tuning — ⬜
- [ ] `M` P7 Cluster monitoring — ⬜
- [x] `M` P2 Backend benchmarking — ✅ sync-vs-prefetch harness (`benches/throughput.py`); NVIDIA `[C]` pending
- [ ] `M` P2 Backend: CV-CUDA — ⬜ `[C]`
- [ ] `M` P3 Benchmark suite + throughput metrics — ⬜ `[C]`
- [ ] `M` P7 RunPod (templates / launch scripts) — ⬜ `[C]`
- [ ] `M` P7 Slurm — ⬜ `[C]`
- [x] `M` P11 Cylindrical panoramic stitching (Phase 1) — ✅
- [x] `M` P11 Laplacian pyramid blending (Phase 2) — ✅
- [x] `M` P11 BEV projection (Phase 3) — ✅
- [ ] `M` P11 GPU acceleration (CuPy NVIDIA backend) — ⬜ `[C]`
- [ ] `M` P11 GPU acceleration (MLX Apple Silicon backend) — ⬜
- [ ] `M` P11 Optical flow seam tracking — ⬜
- [ ] `M` P11 Waymo/nuScenes dataset loaders — ⬜
- [ ] `M` P12 Lidar point cloud fusion — ⬜
- [ ] `M` P12 Radar velocity fusion — ⬜

### L — large / architecture / native
- [x] `L` P1.2 Multiprocess workers (off-GIL worker pool, `num_workers`) — ✅
- [ ] `L` P1.1 Streaming mode (HF Hub partial download) — ⬜
- [ ] `L` P4 MLX inference — ⬜
- [ ] `L` P5 Behavior Cloning — ⬜
- [ ] `L` P5 Imitation learning — ⬜
- [ ] `L` P5 Offline RL — ⬜
- [ ] `L` P5 Transformer policies — ⬜
- [ ] `L` P5 Fine-tuning — ⬜
- [ ] `L` P6 Dataset export (write LeRobotDataset v3.x) — ⬜
- [ ] `L` P7 Ray — ⬜
- [ ] `L` P7 Distributed dataloading — ⬜
- [ ] `L` P7 Multi-GPU — ⬜ `[C]`
- [ ] `L` P7 Kubernetes — ⬜ `[C]`

### XL — research / blocked / very large
- [ ] `XL` P2 GPU-resident zero-copy decode (Video Codec SDK → DLPack) — ⬜ `[C]`
- [ ] `XL` P4 True zero-copy MLX (decode → IOSurface → MLX) — 🟡 blocked on `mlx#2855`
- [ ] `XL` P4 MLX distributed training — ⬜
- [ ] `XL` P5 ACT — ⬜
- [ ] `XL` P5 Diffusion Policies — ⬜
- [ ] `XL` P5 Vision-Language-Action models — ⬜
- [ ] `XL` P8 "Train Anywhere" productized (one-script-six-targets, packaged + documented) — ⬜

### Already done ✅ (for reference)
- [x] P1.1 Load LeRobotDataset v3.x · Frame extraction · Action/state extraction · Metadata · `validate()`
- [x] P1.1 `ds.stats()` · `train_val_split` + `loader(episodes=)` · `ds.episodes()` · normalization · checkpoint/resume
- [x] P1.2 Caching (frame LRU + shard cache) · Batch assembly · Frame indexing · Sharding · **off-GIL prefetch pipeline** · throughput harness
- [x] P1.3 Sequence windows · Future-prediction windows · State–action alignment · Temporal batching · **episode-chunking sampler** · **MLX sequence batches**
- [x] P2 Backends: CPU · Torch · MLX · **JAX** output · profiling (`stats`/`on_batch`) · MLX benchmarks
- [x] P3/Tier-1 **MCAP → columnar** (`convert_mcap`: JSON + protobuf + cdr/ros2msg) · **ROS 2 bag** (`convert_ros2_bag`) · auto **metadata.json**/**stats.json**
- [x] P2/Tier-1 **Robotics DataFrame** (`RoboticsDataFrame`: per-topic access, `slice`, as-of `align`) · runtime deps (numpy/pyarrow) declared

---

## Contributor & user priorities (what defines / differentiates PyRoboFrames)

A priority lens (from a contributor/user view) over the vision below: what makes PyRoboFrames
*itself*, then what makes it *compelling*. Status from the 2026-06-25 audit.

### Tier 1 — Core identity (what PyRoboFrames *is*)
- [x] MCAP → columnar (Parquet) conversion — ✅ JSON + **protobuf** + **cdr/ros2msg** (`convert_mcap`)
- [x] ROS 2 bag (`.db3`) → columnar conversion — ✅ `convert_ros2_bag`
- [x] Robotics DataFrame abstraction (typed, time-indexed, multi-sensor) — ✅ `RoboticsDataFrame`
- [x] Time-synchronized sensor fusion — ✅ `RoboticsDataFrame.align()` as-of join; **MultimodalDataFrame** (v0.4.2) RGB+depth+IMU
- [x] Parquet-backed storage — ✅ reads/writes LeRobot parquet + converter Parquet/metadata
- [ ] MQTT / Kafka ingestion — ⬜ (§P7, deferred)
- [x] LeRobot interoperability — ✅ read v3.0 (local + Hub streaming); write-back (v0.4.1+)
- [x] MLX / PyTorch / JAX data loaders — ✅ MLX/Torch/JAX output + `DataLoader` (native on-device transforms §P6)
- [x] Automotive perception pipeline — ✅ v0.5.0: panoramic stitching (Phase 1-3), BEV projection, camera calibration

### Tier 2 — Differentiators (what makes it *compelling*)
- [ ] SAM / SAM2 segmentation integration — ⬜ (v0.5.1+)
- [ ] Grounding DINO (open-vocab detection) integration — ⬜ (v0.5.1+)
- [ ] CLIP embeddings — ⬜ (v0.5.1+)
- [ ] Automatic annotation pipelines — ⬜ (v0.5.2+)
- [ ] Vision-language dataset generation — ⬜ (v0.5.2+)
- [x] Autonomous driving 360° perception — ✅ v0.5.0: panorama + BEV for AV datasets
- [ ] Real-time 3D object detection (Waymo/nuScenes) — ⬜ (v0.5.2+)
- [ ] Occupancy mapping & lidar fusion — ⬜ (v0.5.2+)

> Read against the near-term plan above: **MCAP → columnar** has shipped (JSON topics, §P1 finishes
> it); the **Robotics DataFrame** (§P2) is the next headline; **MLX/PyTorch/JAX loaders** are done as
> output adapters, with native on-device transforms in §P6. **Kafka/MQTT** (§P7) and the Tier-2
> **vision-model integrations** (§P8) are larger, mostly-Python milestones now placed in the plan.

## Long-range vision (full product surface)

A superset of the prioritized backlog above, capturing where PyRoboFrames could go as a complete
robotics data platform. Status from the 2026-06-27 audit (✅ done · 🟡 partial · ⬜ not started).
Most of this is **not yet scheduled** — it's the vision, not a commitment; the prioritized P0–P10
plan and ease-sorted backlog above remain the near-term work.

### Core data layer
- [ ] Native Parquet-based robotics dataset format — ⬜ (§P3)
- [x] MCAP → PyRoboFrames converter — 🟡 JSON topics done (`convert_mcap`); protobuf/ros2msg pending (§P1)
- [ ] ROS 2 bag → PyRoboFrames converter — ⬜ (§P1)
- [ ] Hugging Face LeRobotDataset importer — 🟡 local path today; Hub download/stream ⬜
- [ ] Dataset versioning and snapshots — ⬜
- [ ] Time-synchronized multi-sensor indexing — 🟡 episode/camera ts sync exists
- [ ] Lazy loading for large datasets — 🟡 per-shard
- [ ] Memory-mapped dataset access — ⬜
- [x] Dataset schema validation — ✅ (`ds.validate()`)
- [ ] Automatic metadata generation — ⬜

### Video & vision
- [x] MP4-backed video storage — ✅ (reads LeRobot mp4)
- [x] Frame-level random access — ✅
- [x] Video timestamp synchronization — ✅
- [x] Multi-camera dataset support — ✅
- [x] Video compression benchmarking — ✅ (H.264/HEVC/AV1, v0.4.1)
- [ ] Image augmentation pipeline — ⬜
- [ ] Vision-language annotation support — ⬜
- [ ] Object-detection label integration — ⬜
- [ ] Segmentation mask support — ⬜
- [x] Depth camera support — ✅ (XYZ/PLY/PCD, v0.4.1)
- [x] Panoramic video stitching — ✅ (cylindrical, v0.5.0)
- [x] Multi-view BEV projection — ✅ (3D perception, v0.5.0)
- [ ] Occupancy grid mapping — ⬜ (v0.5.1)
- [ ] Optical flow estimation — ⬜ (v0.5.1)

### Sensor fusion
- [x] IMU frame abstraction — 🟡 (v0.4.2: gyro + accel data, motion compensation)
- [ ] GPS trajectory abstraction — ⬜
- [ ] LiDAR point-cloud support — ⬜ (planned v0.5.2)
- [ ] Event-camera support — ⬜
- [ ] Audio-stream support — ⬜
- [x] Sensor calibration registry — ✅ (v0.4.1: intrinsics, extrinsics, poses)
- [ ] Sensor health monitoring — ⬜
- [x] Missing-data interpolation — ✅ (v0.4.2: backward as-of join, tolerance filtering)
- [x] Time-series resampling engine — ✅ (v0.4.2: MultimodalDataFrame)
- [x] Multi-rate sensor alignment — ✅ (v0.4.2: time-sync, 50ms tolerance default)
- [ ] Radar velocity fusion — ⬜ (planned v0.5.2)
- [ ] Depth-aware 3D fusion — ⬜ (planned v0.5.2)

### Streaming & edge
- [ ] MQTT data-source connector — ⬜
- [ ] Apache Kafka connector — ⬜
- [ ] Apache Pulsar connector — ⬜
- [ ] WebSocket stream connector — ⬜
- [ ] PyFlink integration — ⬜
- [ ] PySpark integration — ⬜
- [ ] Real-time feature extraction — ⬜
- [ ] Stream-to-dataset writer — ⬜
- [ ] Edge buffering support — ⬜
- [ ] Offline-first synchronization — ⬜

### Storage
- [ ] MinIO native backend — ⬜
- [ ] S3-compatible backend — ⬜
- [x] Local filesystem backend — ✅
- [x] Dataset sharding support — ✅ (reads chunked shards)
- [ ] Incremental dataset append — ⬜
- [ ] Cold-storage archival mode — ⬜
- [ ] Dataset compaction utilities — ⬜
- [ ] Storage cost estimator — ⬜
- [ ] Deduplication engine — ⬜
- [ ] Compression benchmarking toolkit — ⬜

### Data engineering
- [ ] SQL-like robotics queries — ⬜
- [ ] Pandas-compatible API — ⬜
- [ ] Polars-compatible API — ⬜
- [ ] Dataset joins across sensors — ⬜
- [x] Temporal window operations — ✅ (`delta_timestamps`)
- [ ] Rolling statistics engine — ⬜
- [ ] Event-detection pipeline — ⬜
- [ ] Feature-engineering toolkit — ⬜
- [ ] Data-quality scoring — ⬜ (validate is integrity, not scoring)
- [ ] Dataset lineage tracking — ⬜

### ML & AI
- [ ] PyTorch dataset adapter — 🟡 torch output today
- [ ] MLX dataset adapter — 🟡 mlx output today
- [x] JAX dataset adapter — 🟡 jax output today (`output="jax"`)
- [ ] TensorFlow dataset adapter — ⬜
- [ ] RL replay-buffer export — ⬜
- [ ] Imitation-learning dataset export — ⬜
- [ ] VLA dataset export — ⬜
- [ ] Foundation-model training support — ⬜
- [ ] Dataset tokenization pipeline — ⬜
- [ ] Distributed training dataloader — 🟡 off-GIL prefetch in-process; multi-node ⬜

### Apple Silicon first
- [ ] MLX-native training pipeline — ⬜
- [ ] Unified-memory optimizations — ⬜
- [ ] Metal GPU acceleration — ⬜
- [ ] Zero-copy Arrow integration — ⬜
- [ ] Apple Neural Engine experimentation — ⬜
- [ ] MacBook dataset profiling tools — 🟡 throughput harness
- [ ] Mac Studio optimization suite — ⬜
- [ ] Local-first robotics workflows — 🟡 runs fully local today
- [ ] Energy-efficient training benchmarks — ⬜
- [ ] CUDA vs MLX comparison tooling — ⬜

### Developer experience
- [ ] One-line dataset creation API — ⬜
- [ ] Interactive dataset explorer — ⬜
- [ ] Jupyter notebook integration — 🟡 usable as a library
- [ ] VS Code extension — ⬜
- [ ] Dataset visualizer dashboard — ⬜
- [ ] Automatic schema documentation — ⬜
- [ ] CLI toolkit — ⬜
- [ ] Dataset debugging tools — 🟡 `ds.validate()`
- [x] Benchmark suite — ✅ (`benches/throughput.py`)
- [ ] Project templates — ⬜
