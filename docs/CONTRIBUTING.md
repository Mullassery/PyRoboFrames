# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

PyRoboFrames is a Rust workspace with two crates providing zero-copy dataloaders for robot learning on Apple Silicon and Linux:

1. **`crates/pyroboframes-core`**:
   - `dataset.rs` — LeRobot dataset format reader (JSONL episode index, PARQUET/HDF5 trajectories)
   - `loaders/` — Dataloader implementations
     - `roboframe_loader.rs` — Video + proprioceptive data with hardware decode
     - `proprioceptive_loader.rs` — State/action only (no video, 10x faster)
     - `temporal_loader.rs` — Windowing for sequence models
   - `codec/` — Hardware video decode abstraction
     - `videotoolbox.rs` — macOS native H.264/HEVC decode (VideoToolbox)
     - `ffmpeg.rs` — Cross-platform fallback (FFmpeg)
     - `nvdec.rs` — NVIDIA hardware decode (future)
   - `output/` — Zero-copy tensor conversions
     - `torch.rs`, `mlx.rs`, `numpy.rs`, `jax.rs`

2. **`crates/pyroboframes-py`**:
   - PyO3 FFI wrapping `pyroboframes-core`
   - Exposes `RoboFrameDataset`, `ProprioceptiveLoader`, `TemporalLoader` to Python

**Key design**: Decode pipeline is multi-worker (`num_workers` parameter). Each worker spawned as separate Rust thread (via crossbeam), hardware-decoded frames stay in GPU/ANE memory until copied to output tensor (zero-copy when supported).

## Build & Test Commands

**Build**:
```bash
cargo build --release
```

**Python wheel**:
```bash
maturin develop          # Dev install with hot reload
maturin build --release  # Build wheel for PyPI
```

**Tests**:
```bash
cargo test --workspace --release
cargo test -p pyroboframes-core
cargo test -p pyroboframes-py
```

**Benchmarks**:
```bash
cargo bench --package pyroboframes-core
```

**Lint**:
```bash
cargo clippy --workspace
cargo fmt --check
```

## Important Implementation Details

- **Python ≥ 3.10 required** — f-string parsing in dataset index relies on Python 3.10+ `ast` module.
- **Hardware decode fallback**: VideoToolbox (macOS) > NVIDIA NVDEC (CUDA) > FFmpeg (all platforms). Automatic fallback if codec unavailable.
- **LeRobot dataset structure**: Expects `episodes/` directory with JSONL index, PARQUET/HDF5 frames. See [LeRobot docs](https://github.com/huggingface/lerobot).
- **Multi-worker safety**: Crossbeam channels separate worker decode threads from loader thread. No GIL issues.
- **Zero-copy constraint**: Tensor output is zero-copy only when source and target memory layouts match (C-contiguous). Non-matching layouts force copy; logs `frame_copy` metric.
- **Temporal windowing**: `window_size=2, stride=1` on 100-frame episode yields 99 windows. Useful for RNN/Transformer policies requiring context.
- **Output device**: `device="mlx"` targets Apple Neural Engine. `device="cuda"` requires CUDA-enabled PyTorch/JAX. CPU fallback always available.
