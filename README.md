# PyRoboFrames

**The fast path from robot recordings to training batches.** A Rust-backed dataloader for LeRobot v3.0 and friends — in-process video decode, zero-copy arrays, and one API for NumPy, PyTorch, JAX, and MLX.

[![CI](../PyRoboFrames/actions/workflows/ci.yml/badge.svg)](../PyRoboFrames/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyroboframes)](https://pypi.org/project/pyroboframes/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-CUDA%20path%20experimental-76B900?logo=nvidia&logoColor=white)](#nvidia-ecosystem)

> **Why it exists:** robot-learning training loops stall on data loading, not model compute. PyRoboFrames moves dataset reading, temporal windowing, and camera decode into Rust, so a Python training loop gets aligned state/action/video batches without an `ffmpeg` subprocess per frame.

## Quickstart

```bash
pip install pyroboframes
```

```python
import pyroboframes as prf

ds = prf.RoboFrameDataset.from_path("/path/to/lerobot_dataset")
print(ds.num_frames, ds.num_episodes, ds.fps, ds.cameras)

loader = ds.loader(
    batch_size=32,
    shuffle=True,
    cameras=["observation.images.top"],
    output="numpy",  # or "torch" / "mlx" / "jax"
)
for batch in loader:
    batch["observation.state"]        # [32, state_dim] float32
    batch["action"]                   # [32, action_dim] float32
    batch["observation.images.top"]   # [32, H, W, 3] uint8
```

No dataset handy? Pull one from the Hub:

```python
ds = prf.RoboFrameDataset.from_path(
    prf.download_lerobot_dataset("lerobot/aloha_mobile_cabinet")
)
```

## NVIDIA ecosystem

PyRoboFrames is **NVIDIA-optional by design**: the default install needs no CUDA, no GPU, and no NVIDIA driver. The GPU seams that exist today are real code, but they are not the default path.

| Component | Status | What it actually does |
|---|---|---|
| CuPy / CV-CUDA transforms | **Experimental**, optional | GPU-accelerated image transforms with CPU fallbacks (`gpu_acceleration.py`, `cvcuda_transforms.py`). |
| NVDEC / CUDA video decode | **Experimental**, hardware-unverified | Feature-gated Rust decode path. The code documents that functional verification is deferred to NVIDIA hardware; no zero-copy CUDA buffer handoff yet. |
| Linux CUDA wheels | **Not published** | CI builds Linux/Windows wheels as artifacts; PyPI currently serves macOS arm64. |
| SANA-WM synthetic dataset ingestion | **Not integrated** | No SANA-WM loader, generator, or generated-video ingestion ships. Generated frames would need provenance before being treated as sensor data. |

**The honest headline:** the best-tested hardware path today is Apple VideoToolbox on macOS arm64. NVIDIA users can install from source and opt into CUDA features, but should treat those paths as experimental until hardware verification lands.

## At a glance

| Area | Status |
|---|---|
| LeRobot v3.0 reader + batched loader | **Stable** |
| VideoToolbox in-process decode (macOS arm64) | **Stable** |
| HDF5 / NetCDF / RLDS / MCAP / ROS 2 conversion | **Stable**, optional deps |
| NumPy / PyTorch / JAX / MLX output | **Stable** |
| `numpy_zerocopy` per-frame output | **Stable** |
| torchcodec fallback decode | **Optional**, Linux/CI path |
| CUDA transforms / NVDEC decode | **Experimental**, optional |
| Zero-copy CUDA handoff | **Not implemented** |
| SANA-WM generation or ingestion | **Not shipped** |
| Linux / Windows PyPI wheels | **Built by the release workflow; not on PyPI yet** |

## What it is

PyRoboFrames loads robot-learning datasets fast. The heavy lifting — dataset reading, video decode, temporal windowing — is a compiled Rust extension (`pyroboframes._core`, PyO3/maturin); the Python layer is the ergonomic surface (`RoboFrameDataset`, `DataLoader`, format converters, device adapters). It targets robot learning's LeRobot-style episodic data (state/action/video aligned by frame index) and its differentiator is in-process hardware video decode via Apple VideoToolbox — not an `ffmpeg` subprocess per frame.

## Features

- **LeRobot v3.0 native** — direct Rust reader for the `meta/`, `data/`, `videos/` layout; batched iteration with on-the-fly camera decode.
- **Multiple array backends** — `output="numpy" | "torch" | "mlx" | "jax"`; `output="numpy_zerocopy"` returns per-frame `[H, W, 3]` arrays with no shared-batch copy (works with `torch.from_dlpack` / `jax.dlpack.from_dlpack`).
- **In-process hardware decode (macOS)** — real `VTDecompressionSession` returning IOSurface-backed `CVPixelBuffer`s; H.264/HEVC (hev1) zero-copy, transparent `ffmpeg`+`dav1d` fallback for AV1.
- **Format conversion** — bring HDF5 (ROBOMIMIC/ACT), NetCDF, RLDS (Open X-Embodiment), MCAP, or ROS 2 bags into LeRobot format via `convert_*` / `RLDSDataset.from_tfds()`.
- **Cloud and Hub** — `download_lerobot_dataset()` from the Hugging Face Hub; `RemoteDataset.from_s3()/from_gcs()`.
- **Unified registry** — `load_dataset(path, format=...)` across supported formats; missing optional deps raise a clear `ImportError` with an install hint.
- **Ingestion hardening** — bounded, deterministic Hypothesis property tests replay truncated/bit-flipped/random-header HDF5 and NetCDF fixtures through the real loaders on every test run; MP4 decode is exercised in a timeout-isolated subprocess when the optional `torch`/`torchcodec` stack is installed. Corrupt inputs raise `ValueError`/`OSError`, not backend internals or hangs.

## Who it's for / Use cases

- **Training-loop data loading** — iterate a LeRobot dataset in batches (state/action + decoded camera frames) into NumPy/PyTorch/JAX/MLX. Primary, best-tested path.
- **Normalizing mixed recordings** — convert HDF5/NetCDF/RLDS/MCAP/ROS 2 into one LeRobot format so downstream code only knows one reader.
- **Apple Silicon training where camera decode is the bottleneck** — the in-process VideoToolbox path avoids the subprocess-copy cost of per-frame `ffmpeg`.

## Optional torchcodec fallback

On Linux/CI, decode an RGB frame range with PyTorch's torchcodec:

```bash
pip install "pyroboframes[torchcodec]"
```

```python
from pyroboframes import decode_video_frames

frames = decode_video_frames("episode.mp4", start_seconds=1.0, end_seconds=2.0)
# (N, H, W, 3) uint8
```

This is a standalone helper; the native dataset pipeline remains VideoToolbox-first. Missing torchcodec raises `TorchCodecUnavailable`.

## Installation

```bash
pip install pyroboframes
```

Installs a prebuilt PyPI wheel on **macOS arm64**; Linux/Windows wheels are CI artifacts today. On other platforms install from source (Rust toolchain; `ffmpeg`/`ffprobe` on `PATH` for the default build):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install pyroboframes
```

Optional extras/dependencies, depending on format and backend: `pyroboframes[mlx]`, `h5py` (HDF5), `xarray netCDF4` (NetCDF), `tensorflow_datasets` (RLDS), `fsspec s3fs gcsfs` (cloud), `torch`/`jax` (array backends), `scipy scikit-learn` (GPU transforms and 3D occupancy morphology).

## Usage

```python
# Zero-copy per-frame arrays (no shared-batch copy); not available with num_workers > 0
loader = ds.loader(batch_size=32, cameras=["observation.images.top"], output="numpy_zerocopy")

# Convert a foreign recording into LeRobot format
prf.convert_hdf5("/data/robomimic_task.hdf5", "/data/lerobot_out")
```

## What it is not

- **PyPI wheels are macOS (Apple Silicon) only.** CI builds and smoke-tests Linux x86_64/aarch64 and Windows x86_64 wheels as artifacts, but they are not yet published to PyPI. Other platforms install from the source distribution and need a Rust toolchain (and, for the default build, `ffmpeg`/`ffprobe` on `PATH`).
- **Not a true remote-streaming loader.** Cloud readers download to a local cache first.
- **Video decode does not cover everything.** HEVC is limited to `hev1`-tagged files (not `hvc1`), the native path has no full B-frame reorder buffer, and AV1 uses software `dav1d` via subprocess.
- **CUDA decode downloads frames to host memory** — not yet a zero-copy CUDA buffer handoff.
- **`convert_mcap` has an open upstream crash** in `mcap` 0.25.0 on malformed input; it is isolated behind `catch_unwind` so callers get a clean error, but the fuzz smoke test cannot pass until upstream fixes it.
- **Not a general-purpose video toolkit** — scope is robot-learning episodic data.

## Project status

Released and in use, version **2.7.1**. 300+ Python tests (including deterministic HDF5/NetCDF ingest property tests) and 300+ Rust unit+integration tests at this revision. Two shipped incidents are disclosed rather than hidden: the v2.5.0 PyPI wheel was broken (no compiled extension; fixed in v2.5.1), and VideoToolbox hardware decode was dead code in every wheel before v2.5.0 (the Cargo feature was missing from the maturin features list; fixed in v2.5.0). `scripts/release.sh` now guards against the former. See [docs/internal/ROADMAP_HONEST.md](docs/internal/ROADMAP_HONEST.md) for the solid-vs-rough breakdown.

## For AI agents

Canonical facts for automated tooling and coding agents:

| Fact | Value |
|---|---|
| Package | `pyroboframes` |
| Python | `>=3.10` |
| Core | Rust/PyO3 extension `pyroboframes._core` |
| Main entry points | `RoboFrameDataset.from_path()`, `.loader()`, `load_dataset()`, `convert_hdf5()`, `convert_mcap()` |
| Source layout | `crates/` (Rust), `python/pyroboframes/` (Python surface) |
| Python tests | `pip install -e ".[dev]"` then `pytest tests/ -v` |
| Rust tests | `cargo test --release --features ffmpeg` |
| Lint / format | `ruff check .`, `ruff format --check .`, `cargo clippy --release -p pyroboframes-core -- -D warnings` |
| GPU needed? | **No** for core, tests, or default install |
| Optional deps | Lazily imported; missing extras raise `ImportError` with an install hint |
| Generated data | Not produced. SANA-WM ingestion is not integrated. |

Agent-friendly docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/CROSS_PROJECT_INTEGRATION.md](docs/CROSS_PROJECT_INTEGRATION.md), [docs/internal/ROADMAP_HONEST.md](docs/internal/ROADMAP_HONEST.md).

## Documentation

- [docs/](docs/) — architecture, camera calibration, depth cameras, multimodal fusion, and more
- [docs/GR00T_WORKFLOW.md](docs/GR00T_WORKFLOW.md) — validate and load GR00T LeRobot v2.1 datasets
- [.github/INSTALL.md](.github/INSTALL.md) — platform-specific install notes
- [examples/](examples/) — full training-loop scripts (humanoid multimodal fusion, quadruped proprioception)
- [docs/internal/ROADMAP_HONEST.md](docs/internal/ROADMAP_HONEST.md) — solid vs rough, open issues, measurements
- [SECURITY.md](SECURITY.md) — security/compliance posture
- [CHANGELOG.md](CHANGELOG.md) — notable changes

## Known issues and deferred work

**Blocked / failed:**
- Linux/CUDA wheels; zero-copy CUDA; hvc1/AV1.
- The native VideoToolbox/FFmpeg MP4 path is still not fuzzed; only the optional torchcodec fallback has a subprocess-isolated mutation smoke test.

**Rejected / deferred:**
- Remote streaming; cache-first is acceptable.

Full shipped/blocked/rejected detail: [`docs/internal/QUEUE_STATUS.md`](docs/internal/QUEUE_STATUS.md).

## Path sandboxing

`RoboFrameDataset.from_path`, `convert_mcap`, `convert_ros2_bag`, `HDF5Dataset`/`convert_hdf5`, and `NetCDFDataset`/`convert_netcdf` accept `base_dir=` for an explicit containment root. They also accept an opt-in `secure=True` mode: relative paths resolve against the project root (`PYROBOFRAMES_PROJECT_ROOT`, or the current directory) and absolute paths outside it are rejected. `allow_outside_root=True` is the explicit escape hatch; an explicit `base_dir` always remains authoritative. The secure mode is opt-in because flipping the default would break existing callers that legitimately pass absolute paths outside the process working directory.

## Contributing

Bug reports and questions: [GitHub Issues](../PyRoboFrames/issues). Releases are cut with `scripts/release.sh` (not `maturin build` directly).

## Part of the robotics & autonomy family

PyRoboFrames is one of six complementary, composable tools in the Mullassery **robotics & autonomy** family. Together they form a single stack: **perceive → map → simulate → learn → debug → coordinate the fleet**.

| Tool | Role | What it does |
|---|---|---|
| [PyRoboVision](../PyRoboVision) | Perception | Multi-object tracker (Kalman filter + Hungarian algorithm) and trajectory predictor for robot perception. |
| [PyTerrainMap](../PyTerrainMap) | Mapping / SLAM | Fuses multi-robot terrain observations into a spatial-temporal store, with H3 spatial indexing and SLAM. |
| [PyRoboSimulator](../PyRoboSimulator) | Simulation | World simulator for autonomous systems, with a FastAPI backend. |
| [PyRoboFrames](../PyRoboFrames) | Learning data | **This project** — Rust-backed ML dataloader for robot-learning datasets (native LeRobot format, Apple Silicon hardware video decode). |
| [PyRoboReplay](../PyRoboReplay) | Mission forensics | Forensic-debugging and causal-analysis engine for autonomous robot mission replay. |
| [SyncFleet](../SyncFleet) | Fleet coordination | Rust + SQLite, peer-to-peer state sync for robotics fleets — mutual-TLS 1.3, version-vector replication, mDNS discovery; conflict resolution on the roadmap. |

Each works standalone, but they're complementary by design. [Browse all projects →](../Mullassery-profile)

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
