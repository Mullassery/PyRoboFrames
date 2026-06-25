# PyRoboFrames

[![PyPI](https://img.shields.io/pypi/v/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![Python](https://img.shields.io/pypi/pyversions/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**A fast dataloader for training robots from recorded demonstrations — built for Apple Silicon, and Linux too.**

> ⚠️ **Early / experimental** (`0.x`, expect API changes). The LeRobot **dataloader works today**;
> the hardware-decode / **zero-copy-MLX** fast path and the **parallel prefetch pipeline** are still
> in progress, and throughput isn't benchmarked yet. See [What works today](#what-works-today).

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
your training loop as **NumPy, MLX, or PyTorch** arrays — with a focus on **Apple Silicon Macs**,
where the usual CUDA-centric tools serve you poorly.

> **Honest status on speed:** decode today is **FFmpeg** (the Apple Media-Engine hardware path is
> still in progress). The **off-GIL prefetch pipeline** now works (`num_workers=`): on synthetic
> data it scales the FFmpeg camera-decode epoch **~2.7× with 4 workers** vs synchronous
> ([`benches/throughput.py`](./benches/throughput.py)) — a relative sync-vs-prefetch signal on one
> machine, not yet an absolute benchmark vs other libraries. See
> [What works today](#what-works-today).

### When would I use it?

- You're training (or fine-tuning) a robot policy / VLA model from demonstration data.
- Your dataset is in the **LeRobot format** — the open standard from Hugging Face's
  [LeRobot](https://github.com/huggingface/lerobot) project, now used by tens of thousands of
  shared robot datasets. (Support for other formats is on the roadmap.)
- Your data loading is slow, **or** you're developing on a **Mac** and the usual CUDA-centric
  tools don't serve you well.

### Why it's different

- **Apple Silicon first.** **MLX** (and PyTorch) output works today. The headline goal —
  decoding on the Mac's hardware video engine (VideoToolbox) straight into MLX with **zero
  copies** — is in progress; no other robot dataloader even targets it.
- **Rust core, simple Python.** The engine is Rust (native speed, hardware access, room to go
  off-GIL); you just `pip install` and `import` it. The parallel prefetch pipeline that turns
  that into end-to-end throughput is on the roadmap, not wired yet.
- **Runs on Linux too** (NVIDIA CUDA/NVDEC support is planned).

> ## Status: early (`0.1.3`, `0.x` — expect API changes)
> **Works today** on any LeRobotDataset v3.0: the **dataloader** (state/action **and camera
> frames**), shuffling, temporal windows, `validate()`, **dataset stats** (`ds.stats()`),
> **train/val split**, **checkpoint/resume**, and **NumPy / MLX / PyTorch output**. Camera frames
> decode via **FFmpeg**.
> **Not yet:** the Apple-Silicon **zero-copy MLX** path (decode → IOSurface → MLX), the native
> **VideoToolbox / NVDEC** backends, and the **parallel prefetch pipeline** — the loader runs
> single-threaded today and **throughput is not yet benchmarked**. See
> [What works today](#what-works-today).

---

## Installation

Requires Python ≥ 3.10.

```bash
# pip
pip install pyroboframes

# uv
uv pip install pyroboframes
#   or, in a uv project:
uv add pyroboframes

# one-line installer (uses uv if present, else pip)
curl -LsSf https://raw.githubusercontent.com/Mullassery/PyRoboFrames/main/install.sh | sh
```

Prebuilt wheels are published for **macOS (Apple Silicon)**; on other platforms pip builds
from the source distribution (a Rust toolchain is required for that until more wheels ship).

> The `curl` one-liner fetches [`install.sh`](./install.sh) from this repo; it needs the
> repository to be public.

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
| Temporal windows (`delta_timestamps`, `tolerance_s`) | ✅ |
| macOS **and** Linux | ✅ |
| Decoded-frame cache, batched-seek API, backend selection | ✅ |
| **Camera frame decoding** (FFmpeg → NumPy) | ✅ (needs `ffmpeg` on `PATH`) |
| Dataset **validation** (`ds.validate()`) | ✅ |
| Dataset **statistics** (`ds.stats()`) + **normalization** (`loader(normalize=…)`) | ✅ |
| **Train/val split** (`ds.train_val_split()` + `loader(episodes=…)`) | ✅ |
| **Episode iteration** (`ds.episodes()`) | ✅ |
| Loader **checkpoint/resume** (`loader.position` / `seek()`) | ✅ |
| **Off-GIL prefetch pipeline** (`loader(num_workers=…)`) | ✅ |
| **Throughput benchmark** harness (`benches/throughput.py`) | ✅ |
| **NumPy / MLX / PyTorch output** (`output=`) | ✅ (torch is zero-copy from NumPy) |
| Native **VideoToolbox / NVDEC** decode | 🚧 |
| **Zero-copy MLX** (decode → IOSurface → MLX, no NumPy hop) | 🚧 (upstream `mlx#2855`) |
| **CUDA / CV-CUDA** compute · **MPS** output · **HF Hub streaming** | 🚧 |

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
Decode today uses FFmpeg; the Apple VideoToolbox / NVIDIA NVDEC hardware paths are planned.

The engine is Rust (crate `pyroboframes-core`); the Python package is a thin
[PyO3](https://pyo3.rs)/[maturin](https://www.maturin.rs) binding. Full design,
decisions, and trade-offs are in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

### Cross-platform — *Train Anywhere*

The goal: **one script runs unchanged** on a Mac, a rented NVIDIA box, or a CPU — the
environment picks the backend (`device="auto"`), not your code. See
[`docs/ROADMAP.md`](./docs/ROADMAP.md) for the design and build order.

| Target | Decode | Compute / transforms | Output | Status |
|---|---|---|---|---|
| macOS (Apple Silicon) — MLX | FFmpeg | MLX | `mlx.core.array` | ✅ output · ⏳ transforms |
| macOS (Apple Silicon) — MPS | FFmpeg | Torch (MPS) | `torch.Tensor` | ⏳ |
| RTX 5090 / H100 / RunPod | NVDEC | **CV-CUDA** | `torch.Tensor` (cuda) | ⏳ |
| Local CPU | FFmpeg (software) | NumPy / Torch | `np.ndarray` / `torch.Tensor` | ✅ |
| macOS (Apple Silicon) | FFmpeg | — | NumPy · MLX · PyTorch | ✅ · VideoToolbox→zero-copy MLX ⏳ |

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

**Shipped so far (0.1.0 → 0.1.3):** dataloader (state/action + camera frames), buffered shuffle,
temporal windows, `ds.validate()`, **`ds.stats()`**, **train/val split** (`train_val_split` +
`loader(episodes=…)`), **checkpoint/resume**, FFmpeg decode, and NumPy / MLX / PyTorch output —
macOS & Linux. *(All single-threaded; no throughput benchmarks published yet.)*

**Next up:**

- **Performance — the actual speed story.** Wire the off-GIL **parallel prefetch + worker
  pipeline** (today these are config fields only), then publish a reproducible **throughput
  benchmark** vs the FFmpeg/CPU baseline. This is what justifies the word "fast"; until it lands,
  the claim stays a goal.
- **Train Anywhere (multi-backend core).** One script, unchanged, across MacBook (MLX / MPS),
  NVIDIA (RTX 5090 / H100 / RunPod, via **CV-CUDA** + NVDEC), and CPU — the runtime auto-selects
  the backend. Sequenced **test-first**: the backend-detection seam, the unified tensor/transforms
  abstraction, and the CPU/MPS/MLX paths (verifiable on a Mac) land before the NVIDIA paths
  (built feature-gated, functionally signed off on a GPU box). Full plan + priority tiers in
  [`docs/ROADMAP.md`](./docs/ROADMAP.md).
- **0.1.x — The Apple fast path.** Native **VideoToolbox** (macOS) hardware decode → **zero-copy
  MLX** (no NumPy hop, gated on [mlx#2855](https://github.com/ml-explore/mlx/issues/2855)) and
  NVIDIA **NVDEC** on Linux; a published decode-throughput benchmark vs. the FFmpeg/CPU baseline.
- **0.2 — Streaming.** Stream datasets directly from the Hugging Face Hub without a full download
  (à la LeRobot's `StreamingLeRobotDataset`).
- **0.3 — More formats.** MCAP, RLDS / Open X-Embodiment, and HDF5 ingestion behind the same
  loader API.
- **0.4 — Data-quality curation.** Beyond `validate()`: trajectory **scoring** (jitter, diversity,
  sharpness, state-variance) to filter low-quality segments before training.
- **0.5+ — Scale.** Multi-GPU / multi-Mac distributed loading, on-the-fly augmentation, and
  interop with synthetic-data / world-model pipelines.

See [`docs/ROADMAP.md`](./docs/ROADMAP.md) for the "Train Anywhere" multi-backend plan and
priority tiers, and [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md) for the
original v0.1 build sequence.

---

## Documentation

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — design, the gap, and decisions.
- [`docs/COMPARISON.md`](./docs/COMPARISON.md) — alternatives and adopted techniques.
- [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md) — phased build plan.
- [`AGENTS.md`](./AGENTS.md) — orientation for contributors and AI coding agents.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`CHANGELOG.md`](./CHANGELOG.md)

## Contributing

Contributions welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md). The highest-impact work
right now is the video-decode backends and the MLX zero-copy path
([mlx#2855](https://github.com/ml-explore/mlx/issues/2855)).

## License

[MIT](./LICENSE) © Georgi Mammen Mullassery
