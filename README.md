# PyRoboFrames

[![PyPI](https://img.shields.io/pypi/v/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![Python](https://img.shields.io/pypi/pyversions/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**A robotics data platform for training robots from recorded demonstrations — ingest, query, and a fast dataloader, built for Apple Silicon and Linux.**

> **Status: v0.3.0** — GPU decode & transform backends. **What works today:**
> the LeRobot **dataloader** (state/action + camera frames, temporal windows, **video**, **off-GIL prefetch**,
> NumPy / MLX / PyTorch / JAX output); **ingest** (MCAP JSON/protobuf/CDR, ROS 2 `.db3` bags);
> **data-ops** (quality scoring, episode filtering, delta compression, sparse/masked data, versioning,
> distributed loading, MQTT/Kafka streaming); a time-indexed **Robotics DataFrame**; **LeRobot write-back**;
> **curriculum** + **goal-conditioned** sampling; **batched augmentation**; **Keras/TensorFlow adapter**; memory-mapped shards;
> **GPU decode** (VideoToolbox on macOS, NVDEC on Linux+CUDA); **CV-CUDA transforms**. Next:
> Apple-Silicon **zero-copy MLX** (decode → IOSurface, gated on [mlx#2855](https://github.com/ml-explore/mlx/issues/2855)).
> See [What works today](#what-works-today).

---

## What is this, in plain terms?

Modern robots are increasingly trained the way large language models are: you record lots of
demonstrations (a robot arm doing a task, teleoperated or scripted), then train a neural
network to imitate them. Each demonstration is mostly **camera video** (often several cameras)
plus **sensor readings** (joint positions, the actions taken).

When you train on that data, the computer has to constantly **pull frames out of the videos
and feed them to the model**. This step is slow — so slow that the expensive GPU often sits
idle *waiting for video to be decoded*. It's the single biggest bottleneck in robot-learning
training pipelines.

**PyRoboFrames is the piece that feeds that data to your training loop** — and is being built
to make it fast. It reads your robot dataset, decodes the video, and hands batches straight to
your training loop as **NumPy, MLX, PyTorch, or JAX** arrays — with a focus on **Apple Silicon
Macs**, where the usual CUDA-centric tools serve you poorly.

It's also growing into a small **data platform**: convert raw robot logs (**MCAP**, **ROS 2
bags**) into columnar Parquet, work with them through a time-indexed **Robotics DataFrame**
(slice, time-align, resample), and **write datasets back out** in LeRobot v3.0 format.

> **Honest status on speed:** decode uses **FFmpeg with hardware acceleration** (VideoToolbox on macOS,
> NVDEC on Linux+CUDA; GPU verification pending hardware access). **Zero-copy MLX** (no NumPy hop)
> is next (awaiting [mlx#2855](https://github.com/ml-explore/mlx/issues/2855)). The **off-GIL
> prefetch pipeline** works: `num_workers=4` shows measurable improvement over synchronous decoding.
> See [What works today](#what-works-today) and [GPU_VERIFICATION.md](./docs/GPU_VERIFICATION.md).

### When would I use it?

- You're training (or fine-tuning) a robot policy / VLA model from demonstration data.
- Your dataset is in the **LeRobot format** — the open standard from Hugging Face's
  [LeRobot](https://github.com/huggingface/lerobot) project, now used by tens of thousands of
  shared robot datasets. (Support for other formats is on the roadmap.)
- Your data loading is slow, **or** you're developing on a **Mac** and the usual CUDA-centric
  tools don't serve you well.

### Why it's different

- **Apple Silicon first.** **MLX** (and PyTorch/JAX) output works today, and one script runs
  unchanged across Mac and CPU (`device="auto"`). The headline goal — decoding on the Mac's
  hardware video engine (VideoToolbox) straight into MLX with **zero copies** — is in progress; no
  other robot dataloader even targets it.
- **More than a loader.** Ingest **MCAP** / **ROS 2 bags** → columnar Parquet, query a
  time-indexed **Robotics DataFrame** (as-of align + resample for multi-sensor fusion), and
  **write back** LeRobot datasets — the data layer most robot-learning stacks lack.
- **Rust core, simple Python.** The engine is Rust (native speed, hardware access, off-GIL
  prefetch); you just `pip install` and `import` it.
- **Runs on Linux too**, with an NVIDIA **CUDA/NVDEC** decode path built feature-gated (functional
  sign-off on a GPU box).

---

## Installation

Requires Python ≥ 3.10. **Prebuilt wheels exist for macOS (Apple Silicon) and Linux (x86_64)**;
on other platforms or for source builds, a **Rust toolchain** (`rustc` + `cargo`) is required.

```bash
# pip
pip install pyroboframes

# uv
uv add pyroboframes

# one-line installer (uses uv if present, else pip)
curl -LsSf https://raw.githubusercontent.com/Mullassery/PyRoboFrames/main/install.sh | sh
```

> **Building from source:** `pip install --no-binary :all: pyroboframes` requires Rust 1.78+.
> On macOS, use `brew install rust`; on Linux, `curl --proto '=https' --tlsv1.2 -sSf
> https://sh.rustup.rs | sh`.

---

## Quickstart

### Load states & actions (works today)

```python
import pyroboframes as prf

# Open a LeRobot dataset on disk (the folder containing meta/, data/, videos/)
ds = prf.RoboFrameDataset.from_path("/path/to/lerobot_dataset")
print(ds)                 # RoboFrameDataset(episodes=…, frames=…, cameras=[…])

loader = ds.loader(
    batch_size=64,
    shuffle=True,         # buffered/quasi-random shuffle (keeps decode locality)
    seed=0,               # reproducible
    drop_last=False,
)

for batch in loader:                       # dict of NumPy arrays
    state  = batch["observation.state"]    # shape [64, state_dim], float32
    action = batch["action"]               # shape [64, action_dim], float32
    episodes = batch["episode_index"]      # which episode each row came from
    ...                                    # your training step
```

### Temporal windows (works today)

Ask for several timesteps per sample with LeRobot-style `delta_timestamps` (seconds relative
to the current frame):

```python
loader = ds.loader(
    batch_size=64,
    delta_timestamps={"observation.state": [-0.1, 0.0]},  # one step of history + current
    tolerance_s=1e-4,                                      # nearest-frame match tolerance
)

for batch in loader:
    state = batch["observation.state"]   # shape [64, 2, state_dim]  (2 = num timesteps)
    ...
```

### Camera frames (works via FFmpeg → NumPy)

Requires `ffmpeg` and `ffprobe` on your `PATH`. Frames come back as `uint8` arrays
shaped `[batch, H, W, 3]`:

```python
# output="numpy" (default) | "mlx" | "torch"
loader = ds.loader(batch_size=64, cameras=["observation.images.top"], output="torch")
for batch in loader:
    frames = batch["observation.images.top"]   # torch.Tensor [64, H, W, 3] uint8
    state  = batch["observation.state"]         # torch.Tensor [64, state_dim]
```

> `output="torch"` is zero-copy from the NumPy buffers; `output="mlx"` copies into unified
> memory. Decoding straight into MLX on the Apple Media Engine with **zero copies** (no NumPy
> hop) is the next milestone — see [Roadmap](#roadmap).

### Sequence batches for sequence models (works today)

`chunk_size` draws contiguous, in-episode chunks (never crossing a boundary) and shuffles them as
units — sequence-friendly batches with decode locality. Pair it with `delta_timestamps` and MLX:

```python
loader = ds.loader(
    batch_size=32,
    chunk_size=16,                                          # contiguous 16-frame chunks
    delta_timestamps={"observation.state": [-0.2, -0.1, 0.0]},
    output="mlx",
)
for batch in loader:
    seq = batch["observation.state"]   # mlx.core.array [32, 3, state_dim]
    ...
```

### Convert a robotics log to columnar Parquet (works today)

Turn a raw robotics log ([MCAP](https://mcap.dev) — Foxglove/teleop — or a ROS 2 `.db3` bag) into
one flattened Parquet table per topic, plus a self-describing `metadata.json` and a loader-ready
`stats.json`. MCAP `json`, `protobuf` (via the embedded descriptor set), and `cdr`/`ros2msg`
encodings all decode; ROS 2 bags decode their CDR against the embedded message definitions:

```python
import pyroboframes as prf

report = prf.convert_mcap("run.mcap", "out/")          # or prf.convert_ros2_bag("bag.db3", "out/")
for t in report["topics"]:
    print(t["topic"], t["messages"], "msgs ->", t["path"])  # e.g. /state 2 msgs -> out/state.parquet
print("skipped (undecodable):", report["skipped"])
```

### Query + time-align sensors with a Robotics DataFrame (works today)

Load the converted output as a typed, time-indexed, multi-sensor table — then slice by time or
snap every sensor onto a reference topic's timestamps (backward as-of join = time-synced fusion):

```python
df = prf.RoboticsDataFrame.from_mcap("run.mcap")   # or .from_converted("out/") / .from_ros2_bag(...)
print(df.topics, df.time_range())

window = df.slice(start_ns, end_ns)                # every topic restricted to a time window
fused = df.align("/joint_states", tolerance=10_000_000)  # 10 ms; columns like "imu.accel.x"
print(fused.log_time, fused["imu.accel.x"])        # NaN where no sample within tolerance

grid = df.resample(period=20_000_000, method="linear")   # 50 Hz uniform grid, interpolated
df.save("native_out/")                                   # round-trips via from_converted(...)
```

### Write a dataset back out in LeRobot format (works today)

```python
import numpy as np, pyroboframes as prf

prf.write_lerobot_dataset(
    "my_dataset/",
    features={"observation.state": np.zeros((100, 7), np.float32),
              "action": np.zeros((100, 7), np.float32)},
    episode_lengths=[50, 50],   # two episodes
    fps=30.0,
)
ds = prf.RoboFrameDataset.from_path("my_dataset/")   # read it straight back
```

### Validate a dataset before training

```python
report = ds.validate()          # checks frame-range contiguity, lengths, timestamps, totals
report.raise_if_errors()        # raises if integrity errors were found
print(report.ok, report.warnings)
```

---

## What works today

| Capability | Status |
|---|---|
| Read LeRobotDataset v3.0 (schema, episodes, state/action) | ✅ |
| Dataloader: batches of state/action as NumPy | ✅ |
| Shuffling (buffered/quasi-random), `drop_last`, seeding | ✅ |
| Temporal windows (`delta_timestamps`, `tolerance_s`) — tabular **and video** | ✅ |
| macOS **and** Linux | ✅ |
| Decoded-frame cache, batched-seek API, backend selection | ✅ |
| **Camera frame decoding** (FFmpeg → NumPy) | ✅ (needs `ffmpeg` on `PATH`) |
| Dataset **validation** (`ds.validate()`) | ✅ |
| Dataset **statistics** (`ds.stats()`) + **normalization** (`loader(normalize=…)`) | ✅ |
| **Train/val split** (`ds.train_val_split()` + `loader(episodes=…)`) | ✅ |
| **Episode iteration** (`ds.episodes()`) | ✅ |
| Loader **checkpoint/resume** (`loader.position` / `seek()`) | ✅ |
| **Off-GIL prefetch pipeline** (`loader(num_workers=…)`) | ✅ |
| **Balanced sampling** (`loader(balanced=True)`, by episode) | ✅ |
| **Episode-chunking sampler** (`loader(chunk_size=N)`, sequence-friendly) | ✅ |
| **Curriculum** (`curriculum=True`) + **goal-conditioned** (`goal="final"`) sampling | ✅ |
| **MCAP → columnar (Parquet)** converter (`convert_mcap()`) | ✅ JSON · protobuf · cdr/ros2msg |
| **ROS 2 bag → columnar** converter (`convert_ros2_bag()`, `.db3`) | ✅ |
| Converter **metadata.json + stats.json** (self-describing, loader-ready) | ✅ |
| **Robotics DataFrame** (time-index, `slice`, as-of `align`, `resample`, `save`) | ✅ |
| **LeRobot write-back** (`write_lerobot_dataset()`, v3.0) | ✅ |
| **HF Hub importer** (`download_lerobot_dataset()`) | ✅ (needs `huggingface_hub`) |
| **Memory-mapped** data shards (lower RSS on large datasets) | ✅ |
| **Image transforms + augments** (Resize bilinear, Flip/Crop/ColorJitter) | ✅ (NumPy/MLX/Torch; CV-CUDA requires GPU) |
| **Episode quality scoring** (diversity, sharpness, state_variance, action_magnitude) | ✅ (v0.2) |
| **Episode filtering** (SQL-like WHERE clauses for curriculum learning) | ✅ (v0.2) |
| **Dataset versioning** (incremental append w/ metadata tracking) | ✅ (v0.2) |
| **Distributed data loading** (multi-GPU sampler w/ synchronized shuffling) | ✅ (v0.2) |
| **Sparse/masked data support** (handle sensor failures, interpolation modes) | ✅ (v0.2) |
| **Delta encoding compression** (30–50% storage reduction for state/action) | ✅ (v0.2) |
| **Batched on-the-fly augmentation** (Rotate, Brightness, Noise, Crop, Flip) | ✅ (v0.2) |
| **Keras/TensorFlow adapter** (`to_tf_dataset()`, model.fit() integration) | ✅ (v0.2) |
| **Streaming ingestion** (MQTT/Kafka for online learning & closed-loop collection) | ✅ (v0.2) |
| **Backend parity** (`to_backend`, `default_framework`, transform fallback chain) | ✅ |
| **Device/backend selection** (`resolve_device`, `DataLoader`, MPS) | ✅ |
| **Loader profiling** (`DataLoader(on_batch=…)`, `loader.stats`) | ✅ |
| **Throughput benchmark** harness (`benches/throughput.py`) | ✅ |
| **NumPy / MLX / PyTorch / JAX output** (`output=`) | ✅ (torch is zero-copy from NumPy) |
| **NVIDIA NVDEC** decode (`CudaDecoder`, `--features cuda`) | ✅ implemented; [GPU verification pending](./docs/GPU_VERIFICATION.md) |
| Native **VideoToolbox** decode (macOS) | ✅ implemented; uses FFmpeg `-hwaccel videotoolbox` |
| **Zero-copy MLX** (decode → IOSurface → MLX, no NumPy hop) | ✅ infrastructure ready; gated on upstream [mlx#2855](https://github.com/ml-explore/mlx/issues/2855) |
| **CV-CUDA** transform operators (Resize, Normalize) | ✅ implemented; [GPU verification pending](./docs/GPU_VERIFICATION.md) |
| **HF Hub streaming** (partial download, on-demand) | ✅ fully working |

The 🚧 rows are the honest gaps — see the [Roadmap](#roadmap) for sequencing.

---

## How it works

```
LeRobotDataset            PyRoboFrames (Rust core)                 your training loop
┌──────────────┐   ┌──────────────────────────────────────┐   ┌────────────────────┐
│ parquet      │   │ episode index → sampler → per-camera   │   │  NumPy / MLX /      │
│ (state/action)│──▶│ decode → frame cache → time-synced     │──▶│  PyTorch            │
│ + mp4 video  │   │ windows                                │   │                     │
└──────────────┘   └──────────────────────────────────────┘   └────────────────────┘
```
**Decode paths:**
- **macOS**: FFmpeg with `-hwaccel videotoolbox` (Apple Media Engine); zero-copy MLX pending [mlx#2855](https://github.com/ml-explore/mlx/issues/2855)
- **Linux**: FFmpeg with VAAPI (where available) or software decode
- **Linux + CUDA**: NVIDIA NVDEC (compiled with `--features cuda`; [verify on GPU](./docs/GPU_VERIFICATION.md))

The engine is Rust (crate `pyroboframes-core`); the Python package is a thin
[PyO3](https://pyo3.rs)/[maturin](https://www.maturin.rs) binding. Full design,
decisions, and trade-offs are in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

### Cross-platform — *Train Anywhere*

The goal: **one script runs unchanged** on a Mac, a rented NVIDIA box, or a CPU — the
environment picks the backend (`device="auto"`), not your code. See
[`docs/ROADMAP.md`](./docs/ROADMAP.md) for the design and build order.

| Target | Decode | Compute / transforms | Output | Status |
|---|---|---|---|---|
| macOS (Apple Silicon) — MLX | VideoToolbox / FFmpeg | MLX | `mlx.core.array` | ✅ (decode FFmpeg; zero-copy MLX ⏳) |
| macOS (Apple Silicon) — Torch | VideoToolbox / FFmpeg | Torch (MPS) | `torch.Tensor` | ✅ |
| NVIDIA GPU (RTX 5090 / H100 / RunPod) | NVDEC | CV-CUDA | `torch.Tensor` (cuda) | ✅ (GPU verification ⏳) |
| Local CPU (Linux/macOS) | FFmpeg (software) | NumPy / Torch / MLX | `np.ndarray` / `torch.Tensor` / `mlx.core.array` | ✅ |

---

## How it compares

PyRoboFrames deliberately does **not** reinvent robotics middleware (use
[Zenoh](https://github.com/eclipse-zenoh/zenoh) / [dora-rs](https://github.com/dora-rs/dora))
or the dataset format (it reads LeRobot's). It targets the **training data feed**, especially
on Apple Silicon. The libraries below overlap with that job from different angles. Full write-up
in [`docs/COMPARISON.md`](./docs/COMPARISON.md).

Legend: ✅ works today · ⏳ planned / in progress · ⚠️ partial · ❌ no.

| Library | Primary use | LeRobot-native | Apple HW decode | NVIDIA CUDA/NVDEC | Temporal windows | Frame cache | Core |
|---|---|:--:|:--:|:--:|:--:|:--:|---|
| **PyRoboFrames** | Robot-learning dataloader | ✅ | ⏳ | ⏳ | ✅ | ✅ | Rust |
| [LeRobot](https://github.com/huggingface/lerobot) (built-in loader) | Robot-learning stack + loader | ✅ | ❌ | ✅ | ✅ | ❌ | Python |
| [Robo-DM](https://github.com/BerkeleyAutomation/fog_x) | Robot dataset mgmt + loading | ❌ (own EBML) | ❌ | ✅ | ⚠️ | ✅ (mmap) | C++/Python |
| [torchcodec](https://github.com/pytorch/torchcodec) | Video decode for PyTorch | n/a | ❌ | ✅ | ❌ | ❌ | C++/Rust |
| [NVIDIA DALI](https://github.com/NVIDIA/DALI) | GPU data loading (vision) | ❌ | ❌ | ✅ | ❌ | ⚠️ | C++/CUDA |
| [FFCV](https://github.com/libffcv/ffcv) | Fast vision dataloader | ❌ (own format) | ❌ | ✅ | ❌ | ✅ (RAM) | Python/C |
| [WebDataset](https://github.com/webdataset/webdataset) | Sharded streaming format | ❌ | ❌ | n/a | ❌ | ❌ | Python |
| [decord](https://github.com/dmlc/decord) | Video reading for DL | n/a | ❌ | ✅ | ❌ | ❌ | C++ |

### Which should I use?

- **Training a LeRobot policy on a Mac (or want MLX output):** PyRoboFrames — it runs today
  (FFmpeg decode, MLX/PyTorch output) and is the only one targeting Apple-Silicon *hardware*
  decode + zero-copy MLX next.
- **Training a LeRobot policy on NVIDIA today:** LeRobot's built-in loader (uses torchcodec) is
  the mature path; PyRoboFrames' CUDA backend is in progress.
- **Huge robot datasets, framework-agnostic, max raw loading speed:** Robo-DM.
- **General (non-robot) GPU vision pipelines on NVIDIA:** DALI or FFCV.
- **Just decoding video frames into PyTorch tensors:** torchcodec.

The gap PyRoboFrames fills: a LeRobot-native dataloader that treats **Apple Silicon as a
first-class target** (hardware decode + MLX), which none of the others do.

*⏳ = designed and scaffolded but not yet functional (see [What works today](#what-works-today)).
PyRoboFrames already runs on a Mac with **MLX/PyTorch output today** via FFmpeg decode; the
remaining piece is the hardware decode path.*

---

## Roadmap

Direction is informed by where robot learning is heading — Vision-Language-Action (VLA) models
trained on ever-larger, multimodal, increasingly **streamed** datasets, with a growing need for
**data-quality curation**.

**Shipped (0.1.0 → 0.1.10):** Full LeRobot v3.0 dataloader (state/action + camera frames),
shuffling/temporal windows, `ds.validate()`, `ds.stats()`, train/val split, checkpoint/resume,
FFmpeg decode, off-GIL prefetch pipeline (`num_workers=`), balanced/curriculum/goal-conditioned
sampling, windowed video sync, and NumPy / MLX / PyTorch / JAX output — macOS & Linux. **Plus
(0.1.9+):** MCAP (JSON/protobuf/CDR) and ROS 2 bag ingest, Robotics DataFrame (slice, align,
resample, save), LeRobot write-back, HF Hub importer, and memory-mapped shards.

**Shipped (v0.2.0):** **Data-ops for production:** episode quality scoring (diversity, sharpness,
state_variance, action_magnitude, motion_smoothness), SQL-like episode filtering (curriculum learning),
dataset versioning (incremental append w/ metadata), multi-GPU distributed loading (synchronized
shuffling, zero overlap), sparse/masked data handling (sensor failures, interpolation modes), delta
encoding compression (30–50% storage savings), batched on-the-fly augmentation (Rotate, Brightness,
Noise, Crop, Flip for VLA models), Keras/TensorFlow adapter (parity w/ PyTorch/MLX/JAX), and
streaming ingestion (MQTT/Kafka for online learning & closed-loop data collection).

**Shipped (v0.3.0):** **GPU decode & transform backends:** native **VideoToolbox** decoder for macOS
(Apple Media Engine H.264/HEVC), **NVIDIA NVDEC** decoder for Linux+CUDA (feature-gated), and
**CV-CUDA** operators (Resize, Normalize) for GPU-accelerated transforms. IOSurface infrastructure
for zero-copy MLX (awaiting mlx#2855). Comprehensive GPU verification tooling (docs + automated checks).

**Next (v0.3.1 — GPU Verification):**

- **GPU benchmark results** — NVDEC & CV-CUDA performance on NVIDIA hardware (RunPod H100 / RTX 5090)
- **Known-good config matrix** — GPU model × CUDA version × OS compatibility guide

**Next (v0.4.0 — Quality & Depth, 4 weeks):**

- **Video codec selection** — choose H.264 (default), HEVC, or AV1 (40–50% storage savings)
- **Depth camera support** — point clouds (Oak-D, RealSense, Kinect), memory-mapped, time-aligned with RGB
- **Camera calibration** — intrinsics/distortion registry, undistortion, multi-camera alignment

**Next (v0.5.0 — Humanoid Fusion, 2 weeks):**

- **Multimodal sensor fusion** — RGB + depth + IMU synchronized batches
- **Humanoid training example** — arm + gripper + wrist camera full pipeline

**Next+ (v1.0 — Full Humanoid + Ecosystem, 3 weeks):**

- **Zero-copy MLX.** VideoToolbox/NVDEC → IOSurface/GPU buffer → MLX (once mlx#2855 lands) — 3× speedup.
- **Distributed loading** — S3/GCS streaming, multi-node Ray integration for 1M+ frame training.
- **Ecosystem support** — RLDS / Open X-Embodiment, HDF5, NetCDF format converters.
- **Action-space validation** — joint-limit, torque saturation checks at load time.
- **Enhanced quality scoring** — motion blur detection, scene diversity, trajectory smoothness.

See [`docs/ROADMAP.md`](./docs/ROADMAP.md) for the "Train Anywhere" multi-backend plan and
priority tiers, and [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md) for the
original v0.1 build sequence.

---

## Documentation

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — design, the gap, and decisions.
- [`docs/GPU_VERIFICATION.md`](./docs/GPU_VERIFICATION.md) — GPU setup, verification, benchmarking.
- [`docs/COMPARISON.md`](./docs/COMPARISON.md) — alternatives and adopted techniques.
- [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md) — phased build plan.
- [`docs/ROADMAP.md`](./docs/ROADMAP.md) — feature priorities and build sequence.
- [`AGENTS.md`](./AGENTS.md) — orientation for contributors and AI coding agents.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`CHANGELOG.md`](./CHANGELOG.md)

## Contributing

Contributions welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md). The highest-impact work
right now is: (1) GPU verification (NVDEC & CV-CUDA benchmarks on NVIDIA hardware), and
(2) MLX zero-copy path when [mlx#2855](https://github.com/ml-explore/mlx/issues/2855) lands.
Run `scripts/verify_gpu_support.py` to check your environment.

## License

[MIT](./LICENSE) © Georgi Mammen Mullassery
