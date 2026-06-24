# PyRoboFrames

[![PyPI](https://img.shields.io/pypi/v/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![Python](https://img.shields.io/pypi/pyversions/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**A fast dataloader for training robots from recorded demonstrations — built for Apple Silicon, and Linux too.**

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

**PyRoboFrames is the piece that makes that data feed fast.** It reads your robot dataset,
decodes the video on dedicated hardware, and hands batches straight to your training loop —
with special care for **Apple Silicon Macs**, where the usual tools waste the Mac's video
engine and run everything on the CPU.

### When would I use it?

- You're training (or fine-tuning) a robot policy / VLA model from demonstration data.
- Your dataset is in the **LeRobot format** — the open standard from Hugging Face's
  [LeRobot](https://github.com/huggingface/lerobot) project, now used by tens of thousands of
  shared robot datasets. (Support for other formats is on the roadmap.)
- Your data loading is slow, **or** you're developing on a **Mac** and the usual CUDA-centric
  tools don't serve you well.

### Why it's different

- **Apple Silicon first.** It uses the Mac's hardware video engine (VideoToolbox) and Apple's
  ML framework (MLX) with zero-copy hand-off — a path no other robot dataloader targets.
- **Fast core, simple Python.** The engine is Rust (no GIL, hardware access); you just
  `pip install` and `import` it.
- **Runs on Linux too**, including NVIDIA CUDA/NVDEC when present.

> ## Status: early (`0.1.1`, `0.x` — expect API changes)
> Works today on any LeRobotDataset v3.0: the **dataloader** (state/action **and camera
> frames**), shuffling, temporal windows, and **`validate()`**. Camera frames decode via
> **FFmpeg** and return as NumPy. Still in progress: the Apple-Silicon **zero-copy MLX** path
> and the native VideoToolbox / NVDEC backends. See [What works today](#what-works-today).

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
loader = ds.loader(batch_size=64, cameras=["observation.images.top"])
for batch in loader:
    frames = batch["observation.images.top"]   # np.uint8, [64, H, W, 3]
    state  = batch["observation.state"]
    # one line to your framework:
    #   torch.from_numpy(frames)   # PyTorch
    #   mlx.core.array(frames)     # MLX
```

> Decoding straight into MLX on the Apple Media Engine with **zero copies** is the next
> milestone (see [Roadmap](#roadmap)); today frames are FFmpeg-decoded and returned as NumPy.

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
| Native **VideoToolbox / NVDEC** decode | 🚧 |
| **Zero-copy MLX** output | 🚧 (NumPy → `mlx.core.array` in one line) |
| **PyTorch / CUDA** output | 🚧 (NumPy → `torch.from_numpy` in one line) |

---

## How it works

```
LeRobotDataset            PyRoboFrames (Rust core)                 your training loop
┌──────────────┐   ┌──────────────────────────────────────┐   ┌────────────────────┐
│ parquet      │   │ episode index → sampler → per-camera   │   │  MLX  (Apple) /     │
│ (state/action)│──▶│ hardware decode → frame cache →        │──▶│  NumPy / PyTorch    │
│ + mp4 video  │   │ time-synced windows                    │   │                     │
└──────────────┘   └──────────────────────────────────────┘   └────────────────────┘
```

The engine is Rust (crate `pyroboframes-core`); the Python package is a thin
[PyO3](https://pyo3.rs)/[maturin](https://www.maturin.rs) binding. Full design,
decisions, and trade-offs are in [`ARCHITECTURE.md`](./ARCHITECTURE.md).

### Cross-platform

| Platform | Decode | Output |
|---|---|---|
| macOS (Apple Silicon) | VideoToolbox (Media Engine) | MLX (zero-copy), NumPy |
| Linux | FFmpeg (VAAPI / software) | NumPy, PyTorch |
| Linux + CUDA (`--features cuda`) | NVIDIA NVDEC | PyTorch (CUDA) |

---

## How it compares

PyRoboFrames deliberately does **not** reinvent robotics middleware (use
[Zenoh](https://github.com/eclipse-zenoh/zenoh) / [dora-rs](https://github.com/dora-rs/dora))
or the dataset format (it reads LeRobot's). It targets the **training data feed**, especially
on Apple Silicon. The libraries below overlap with that job from different angles. Full write-up
in [`docs/COMPARISON.md`](./docs/COMPARISON.md).

Legend: ✅ yes · ⚠️ partial / with caveats · ❌ no · *(t)* = PyRoboFrames target, in progress.

| Library | Primary use | LeRobot-native | Apple HW decode + MLX | NVIDIA CUDA/NVDEC | Temporal windows | Frame cache | Core |
|---|---|:--:|:--:|:--:|:--:|:--:|---|
| **PyRoboFrames** | Robot-learning dataloader | ✅ | ✅ *(t)* | ✅ *(t)* | ✅ | ✅ | Rust |
| [LeRobot](https://github.com/huggingface/lerobot) (built-in loader) | Robot-learning stack + loader | ✅ | ❌ | ✅ | ✅ | ❌ | Python |
| [Robo-DM](https://github.com/BerkeleyAutomation/fog_x) | Robot dataset mgmt + loading | ❌ (own EBML) | ❌ | ✅ | ⚠️ | ✅ (mmap) | C++/Python |
| [torchcodec](https://github.com/pytorch/torchcodec) | Video decode for PyTorch | n/a | ❌ | ✅ | ❌ | ❌ | C++/Rust |
| [NVIDIA DALI](https://github.com/NVIDIA/DALI) | GPU data loading (vision) | ❌ | ❌ | ✅ | ❌ | ⚠️ | C++/CUDA |
| [FFCV](https://github.com/libffcv/ffcv) | Fast vision dataloader | ❌ (own format) | ❌ | ✅ | ❌ | ✅ (RAM) | Python/C |
| [WebDataset](https://github.com/webdataset/webdataset) | Sharded streaming format | ❌ | ❌ | n/a | ❌ | ❌ | Python |
| [decord](https://github.com/dmlc/decord) | Video reading for DL | n/a | ❌ | ✅ | ❌ | ❌ | C++ |

### Which should I use?

- **Training a LeRobot policy on a Mac (or want MLX):** PyRoboFrames — it's the only one
  targeting Apple Silicon hardware decode + zero-copy MLX.
- **Training a LeRobot policy on NVIDIA today:** LeRobot's built-in loader (uses torchcodec) is
  the mature path; PyRoboFrames' CUDA backend is in progress.
- **Huge robot datasets, framework-agnostic, max raw loading speed:** Robo-DM.
- **General (non-robot) GPU vision pipelines on NVIDIA:** DALI or FFCV.
- **Just decoding video frames into PyTorch tensors:** torchcodec.

The gap PyRoboFrames fills: a LeRobot-native dataloader that treats **Apple Silicon as a
first-class target** (hardware decode + MLX), which none of the others do.

*"(target)" = designed and scaffolded; video-decode backends are in progress (see status above).*

---

## Roadmap

Direction is informed by where robot learning is heading — Vision-Language-Action (VLA) models
trained on ever-larger, multimodal, increasingly **streamed** datasets, with a growing need for
**data-quality curation**.

- **0.1.x — Make frames real.** VideoToolbox (macOS) and FFmpeg (Linux) decode; zero-copy MLX
  output; a published decode-throughput benchmark vs. the PyAV/CPU baseline.
- **0.2 — Streaming & PyTorch.** Stream datasets directly from the Hugging Face Hub without a
  full download (à la LeRobot's `StreamingLeRobotDataset`); PyTorch-MPS/CUDA output via DLPack.
- **0.3 — More formats.** MCAP, RLDS / Open X-Embodiment, and HDF5 ingestion behind the same
  loader API.
- **0.4 — Data quality.** `ds.validate()` plus trajectory **scoring/curation** (jitter,
  diversity, sharpness, state-variance) to filter low-quality segments before training —
  increasingly essential as datasets scale.
- **0.5+ — Scale.** Multi-GPU / multi-Mac distributed loading, on-the-fly augmentation, and
  interop with synthetic-data / world-model pipelines.

See [`docs/IMPLEMENTATION_PLAN.md`](./docs/IMPLEMENTATION_PLAN.md) for the near-term build plan.

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
