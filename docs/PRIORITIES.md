# PyRoboFrames Priorities — At a Glance

High-level summary of work organized by priority tier. For detailed breakdown, see [`ROADMAP.md`](./ROADMAP.md).

---

## 🎯 Current Status: v0.1.11+

**Shipped (P0–P6 + P9 build/benchmark):**
- ✅ LeRobot v3.0 dataloader (state/action + camera frames, temporal windows, off-GIL prefetch)
- ✅ MCAP ingestion (JSON, protobuf, CDR/ros2msg)
- ✅ ROS 2 bag converter (`.db3` SQLite + embedded message definitions)
- ✅ Robotics DataFrame (time-indexed, multi-sensor, slice/align/resample/save)
- ✅ LeRobot write-back + HF Hub importer (full-download or partial-streaming)
- ✅ Memory-mapped shards + curriculum/goal-conditioned sampling
- ✅ Backend parity (NumPy / MLX / PyTorch / JAX output, unified abstraction)
- ✅ **MLX/Torch native transforms** (Resize, Normalize with device dispatch)
- ✅ **HF Hub partial-streaming** (selective episode download for faster startup)
- ✅ **NVIDIA throughput benchmark** (FFmpeg baseline + frame timing)
- ✅ NVDEC decode path (built feature-gated, awaiting GPU verification)

**In progress or blocked:**
- 🟡 Apple Silicon zero-copy MLX (blocked on [mlx#2855](https://github.com/ml-explore/mlx/issues/2855))
- 🟡 NVIDIA CUDA/NVDEC functional verification (needs GPU hardware)
- ⬜ CV-CUDA operators (requires NVIDIA hardware)

---

## 📋 Completed & Next Priorities

### ✅ Just Completed (0.1.11+)
| Priority | Item | Effort | Impact | Test | Status |
|----------|------|--------|--------|------|--------|
| **P7a** | **HF Hub partial-streaming** | S | High | ✓ | ✅ |
| **P6a** | **MLX / MPS native transforms** | M | High | ✓ | ✅ |
| **P9a** | **NVIDIA throughput benchmark** | M | Medium | [C] | ✅ |

### Next Must-do (high-impact, low-effort)
| Priority | Item | Effort | Impact | Test | Status |
|----------|------|--------|--------|------|--------|
| **P6b** | **Row-group-level lazy Parquet reads** | M | High | ✓ | ⬜ |
| **P9b** | **CV-CUDA transform operators** | L | High | [C] | ⬜ |
| **P8** | **Tier-2 vision** (CLIP → SAM → Grounding DINO) | L | High | ✓ | ⬜ |

### Nice-to-have (larger scope)
| Priority | Item | Effort | Impact | Test | Status |
|----------|------|--------|--------|------|--------|
| **P8** | **Tier-2 vision** (CLIP → SAM → Grounding DINO) | L | High | ✓ | ⬜ |
| **P10** | **Scale & research** (Ray, Slurm, multi-node) | L | Medium | ~test | ⬜ |
| **P7** | **Streaming** (MQTT / Kafka) | L | Medium | ~test | ⬜ |

---

## 🏗️ Effort Scale

| Symbol | Time | Examples |
|--------|------|----------|
| **XS** | ≤1 day | Wire existing code, expose method, simple config |
| **S** | 1–2 days | Small module, one feature flag, single-file logic |
| **M** | 3–7 days | Subsystem integration, new backend, benchmarking suite |
| **L** | 1–2+ weeks | Architecture change, native implementation, new protocol |
| **XL** | Multi-week+ | Research, blocked work, multi-team effort |

---

## 🧪 Verification Tiers

| Tier | Hardware | Meaning | Status |
|------|----------|---------|--------|
| **A** | Any CPU / this Mac | Build and functionally test now | ✅ Primary |
| **B** | Apple-Silicon GPU (MLX/MPS) | Testable on maintainer's Mac | 🟡 Partial |
| **C** | NVIDIA GPU (RTX/H100/RunPod) | Build + lint now; verify later | ⬜ Deferred |

All Tier-A and Tier-B items stay verifiable on CI. Tier-C items are feature-gated with CPU fallbacks.

---

## 📊 Feature Coverage Matrix (v0.1.10)

### Data layer
| Feature | Status | Notes |
|---------|--------|-------|
| LeRobot v3.0 read | ✅ | Local + HF Hub download |
| LeRobot v3.0 write | ✅ | `write_lerobot_dataset()` |
| Native Parquet storage | ✅ | Via `RoboticsDataFrame.save()` |
| MCAP → Parquet | ✅ | JSON, protobuf, CDR/ros2msg |
| ROS 2 bag → Parquet | ✅ | `.db3` SQLite format |
| Metadata + stats | ✅ | Auto-generated for converters |
| HF Hub full-download | ✅ | `download_lerobot_dataset()` (default) |
| HF Hub partial-streaming | ✅ | `episodes=[...]` for selective pre-download |
| Lazy Parquet row-groups | 🟡 | Per-shard only; row-group streaming ⬜ |

### Compute & loader
| Feature | Status | Notes |
|---------|--------|-------|
| Off-GIL prefetch | ✅ | `num_workers=N` (2.7× on decode) |
| Memory-mapped shards | ✅ | Lower RSS, random access |
| Temporal windowing | ✅ | `delta_timestamps` for video sequences |
| Multi-camera sync | ✅ | Windowed frame extraction |
| Curriculum sampling | ✅ | Difficulty-ordered episodes |
| Goal-conditioned | ✅ | Append final-frame goals |
| Balanced sampling | ✅ | Equal episodes per batch |
| Train/val split | ✅ | `ds.train_val_split` |

### Output formats
| Backend | Status | Notes |
|---------|--------|-------|
| NumPy | ✅ | Default, CPU-only |
| PyTorch | ✅ | CPU + CUDA (via device) |
| MLX | ✅ | Apple Silicon, sequence batches |
| JAX | ✅ | CPU + GPU via XLA |
| Backend auto-select | ✅ | `resolve_device("auto")` |
| Transform fallback chain | ✅ | CV-CUDA → MLX → Torch → NumPy |
| Native MLX transforms | ✅ | Resize (bilinear/nearest), Normalize |
| Native Torch transforms | ✅ | Resize (via F.interpolate), Normalize |
| Native MPS transforms | ✅ | Via Torch MPS backend |

### Decoding
| Decoder | Status | Notes |
|---------|--------|-------|
| FFmpeg (CPU) | ✅ | Software decode, fallback |
| VideoToolbox (macOS) | 🟡 | Built feature-gated; zero-copy MLX blocked |
| NVDEC (NVIDIA) | 🟡 | Built feature-gated; functional verify pending |
| NVIDIA benchmark | ✅ | `benches/nvidia_benchmark.py` measures FFmpeg baseline |
| MLX zero-copy | 🟡 | Blocked on mlx#2855 (IOSurface support) |

### Data platform
| Feature | Status | Notes |
|---------|--------|-------|
| RoboticsDataFrame | ✅ | `slice()`, `align()`, `resample()`, `save()` |
| Time-indexed access | ✅ | Per-topic `TopicFrame` views |
| As-of join (align) | ✅ | Backward join with tolerance |
| Multi-rate resample | ✅ | "previous", "nearest", "linear" methods |
| Schema validation | ✅ | `ds.validate()` |
| Statistics profiling | ✅ | `ds.stats()` → mean, std, min, max, etc. |

---

## 🔄 Open Blockers & Deferred

### Blocked
- **Apple Silicon zero-copy MLX** — Waiting on [mlx#2855](https://github.com/ml-explore/mlx/issues/2855) (IOSurface → MLX integration)
- **NVIDIA functional verification** — NVDEC built, waiting for GPU hardware sign-off

### Deferred (next batch)
- **HF Hub partial-streaming** — Download only needed shards (vs full dataset)
- **Row-group-level lazy Parquet** — True streaming reads without shard load
- **MLX/MPS native transforms** — On-device Resize/Crop/Normalize (NumPy path works)
- **CV-CUDA operators** — Requires NVIDIA hardware + verification

---

## 💡 Quick Win Ideas

Grab these if looking for small, high-impact contributions:

| Idea | Effort | Impact | Notes |
|------|--------|--------|-------|
| Publish throughput benchmark | S | High | Timing already instrumented; just document |
| Add dataset examples | S | High | LeRobot dataset download + load example |
| MLX transform kernels | M | High | Resize/Crop/Normalize in MLX (seam ready) |
| MQTT connector prototype | M | Medium | Proof-of-concept streaming ingestion |
| Docker / DevContainer setup | S | Medium | Reproducible dev environment |

---

## 📚 Related Docs

- **[`ROADMAP.md`](./ROADMAP.md)** — Full prioritized plan (P0–P10), verification tiers, effort estimates
- **[`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)** — Original v0.1 build sequence
- **[`COMPARISON.md`](./COMPARISON.md)** — Competitive analysis & adopted techniques
- **[`ARCHITECTURE.md`](../ARCHITECTURE.md)** — Design decisions and the gap we're filling
