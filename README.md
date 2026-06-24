# PyRoboFrames

**Zero-copy, hardware-accelerated robot-learning dataloader for Apple Silicon.**

PyRoboFrames feeds robot-learning training loops on Apple Silicon at hardware speed. It
reads robot datasets (LeRobotDataset v3.0, with MCAP planned), decodes their multi-camera
video on the Apple **Media Engine** via VideoToolbox, and hands the frames to **MLX** (and
PyTorch-MPS) as arrays **without a single CPU copy** — turning the data path from the
training bottleneck into a non-event.

> Status: **pre-alpha / scaffolding.** The architecture is designed and the gap is
> validated (see [`ARCHITECTURE.md`](./ARCHITECTURE.md)); the v0.1 implementation is in
> progress. APIs will change. Not yet published to PyPI.

---

## The problem

Robot-learning datasets store observations as **MP4 video** (often several cameras per
episode). During training, every sample requires seeking into those videos and decoding the
right frames. This decode step is the dominant cost of the data pipeline — Hugging Face's
own LeRobot tracker reports training that is *"completely bottlenecked by video decoding even
on servers with hundreds of cores,"* spending more time waiting on the dataloader than on
backprop ([lerobot#1623](https://github.com/huggingface/lerobot/issues/1623)).

On **Apple Silicon** the problem is worse, and avoidably so: the standard Python stack
(torchvision / PyAV / FFmpeg software decode) runs on the CPU and leaves the dedicated
**Media Engine idle**, then copies frames across to the GPU — copies that are pure waste on a
unified-memory machine. Meanwhile the compute side (MLX, M5 Neural Accelerators) is fast and
underfed.

## What PyRoboFrames does

```
LeRobotDataset / MCAP        PyRoboFrames (Rust core)              your training loop
┌───────────────────┐   ┌──────────────────────────────────┐   ┌────────────────────┐
│ parquet (state /  │   │ index → sample → VideoToolbox HW   │   │  MLX  (Neural       │
│ action) + mp4     │──▶│ decode → IOSurface (shared mem) →  │──▶│  Accelerators) or  │
│ video shards      │   │ time-synced windows, no copy       │   │  PyTorch-MPS        │
└───────────────────┘   └──────────────────────────────────┘   └────────────────────┘
```

- **Hardware decode** via Apple VideoToolbox — uses the Media Engine, not the CPU.
- **Zero-copy** — decoded frames live in IOSurface-backed unified memory and are wrapped as
  MLX arrays without a host→device transfer (there is no "device transfer" on unified memory;
  we stop pretending there is).
- **Time-synced windows** — assembles `(multi-camera frames, joint state, action)` windows by
  joining the parquet tabular data with the decoded video at matching timestamps.
- **Built-in validation** — flags missing frames, non-monotonic timestamps, and
  camera/state misalignment before they silently corrupt a training run.

## Why a Rust core with a Python API

The audience is ML researchers, so the product is a `pip`-installable Python package — the
Rust is invisible. Rust is the implementation because the hot path (HW decode, IOSurface
lifetime management, off-GIL prefetch, zero-copy buffer hand-off) is exactly where a safe
systems language with no GIL earns its keep. Same pattern as the author's other engines:
fast Rust core, ergonomic Python shell, one codebase via [PyO3](https://pyo3.rs) +
[maturin](https://www.maturin.rs).

## Installation

> Not yet released. When v0.1 ships:

```bash
pip install pyroboframes        # macOS / Apple Silicon wheels, no Rust toolchain needed
```

Wheels are built for Apple Silicon (primary target) with a portable FFmpeg fallback for
other platforms.

## Quickstart (planned v0.1 API)

```python
import pyroboframes as prf

# Open a LeRobot dataset (local path or Hugging Face Hub repo id)
ds = prf.RoboFrameDataset.from_hub("lerobot/aloha_sim_insertion_human")

# Validate before training
report = ds.validate()
report.raise_if_errors()        # missing frames, timestamp gaps, cam/state mismatch

# Build a dataloader that yields MLX arrays, zero-copy, decoded on the Media Engine
loader = ds.loader(
    batch_size=64,
    cameras=["top", "wrist"],
    window=1,                   # frames of temporal context per sample
    shuffle=True,
    num_workers=4,              # Rust worker pool, runs off the GIL
    output="mlx",               # or "numpy" / "torch" (MPS)
)

for batch in loader:
    frames = batch["observation.images.top"]   # mlx.core.array, already on GPU
    state  = batch["observation.state"]
    action = batch["action"]
    ...                                          # your MLX training step
```

## Supported (target matrix)

| | v0.1 | Planned |
|---|---|---|
| Datasets | LeRobotDataset v3.0 | MCAP, RLDS, HDF5 |
| Codecs (HW) | H.264, HEVC (VideoToolbox) | ProRes, AV1 (M3+) |
| Output | MLX, NumPy | PyTorch-MPS (DLPack) |
| Platform | macOS / Apple Silicon | Linux+CUDA fallback, x86 (SW decode) |

## Benchmarks

The headline metric is decode+load throughput on Apple Silicon vs. the PyAV/CPU path.
Numbers will be published here with a reproducible harness once v0.1 lands.

| Pipeline | Frames/s (M-series) | Notes |
|---|---|---|
| PyAV / CPU (baseline) | _TBD_ | torchvision default backend |
| PyRoboFrames (VideoToolbox, zero-copy) | _TBD_ | target: multiple× baseline |

## Roadmap

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design and decisions.

- **v0.1** — LeRobotDataset v3.0 → VideoToolbox decode → zero-copy MLX loader + validation + benchmark harness.
- **v0.2** — MCAP ingest, PyTorch-MPS output via DLPack.
- **v0.3** — RLDS / HDF5 ingest, multi-Mac distributed loading.

## Contributing

Contributions welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md). The Rust core lives in
`crates/`, the Python package in `python/`. The most valuable early contributions are around
the MLX zero-copy init path (see [mlx#2855](https://github.com/ml-explore/mlx/issues/2855))
and the benchmark harness.

## Prior art & acknowledgements

PyRoboFrames stands on [LeRobot](https://github.com/huggingface/lerobot),
[MLX](https://github.com/ml-explore/mlx), Apple VideoToolbox, [PyO3](https://pyo3.rs), and the
Rust FFmpeg ecosystem. It deliberately does **not** reinvent robotics middleware — that space
is well served by [Zenoh](https://github.com/eclipse-zenoh/zenoh) and
[dora-rs](https://github.com/dora-rs/dora). It targets the one layer they leave unsolved on
Apple Silicon: the training data feed.

## License

[MIT](./LICENSE) © Georgi Mammen Mullassery
