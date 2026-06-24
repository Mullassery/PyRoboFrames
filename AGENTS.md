# AGENTS.md

Orientation for contributors and AI coding agents working in this repository. (Human-readable
too — it's just a concise map of the project and its conventions.)

## What this project is

PyRoboFrames is a **dataloader for robot-learning training**. It reads robot demonstration
datasets (camera video + joint-state/action streams), decodes the video, and yields batches to
a training loop. The differentiator is **Apple Silicon**: hardware video decode via
VideoToolbox and zero-copy hand-off to **MLX**. It also runs on Linux (FFmpeg, and CUDA/NVDEC
when present). The product is a **Python package**; the engine is **Rust**. Target format is
**LeRobotDataset v3.0**.

See `README.md` for the user view and `ARCHITECTURE.md` for the design.

## Current status (keep this honest)

- **Works:** LeRobotDataset v3.0 readers; a tabular dataloader (state/action → NumPy) with
  shuffle, `drop_last`, seeding, and temporal windows (`delta_timestamps`). macOS + Linux.
- **Stubbed / in progress:** the real video-decode backends (VideoToolbox / FFmpeg / NVDEC) —
  they currently return a "not implemented" error behind cargo features. The decode trait,
  frame cache, frame pool, backend selection, and pipeline integration are done and tested.
- **Not built yet:** zero-copy MLX output, PyTorch/CUDA output, `ds.validate()`.

When you implement something, update `README.md` ("What works today") and `CHANGELOG.md`.

## Repository layout

```
crates/pyroboframes-core/   Rust engine (no Python). All real logic lives here.
  src/
    info.rs        meta/info.json: schema, camera detection, fps, path templates
    dataset.rs     Dataset facade (open, accessors, shard-path resolution)
    episodes.rs    meta/episodes/*.parquet index; locate(global_frame) -> (cam, file, ts)
    data.rs        data/*.parquet tabular reader (float vector features)
    decode.rs      Decoder trait, Frame/FrameBuffer, Backend selection, FrameCache, FramePool
    sampler.rs     buffered/quasi-random shuffle ordering
    window.rs      delta_timestamps -> in-episode frame offsets (snap + clamp)
    loader.rs      TabularLoader (state/action) + decode_frames() integration
    lib.rs         Error/Result, LoaderConfig, ValidationReport, re-exports
crates/pyroboframes-py/     Thin PyO3 binding -> the `pyroboframes._core` extension module
python/pyroboframes/        Python package surface (re-exports the extension)
tests/                      Python end-to-end tests (build a synthetic dataset with pyarrow)
benches/                    (placeholder) decode/load throughput harness
docs/                       COMPARISON.md, IMPLEMENTATION_PLAN.md
```

## Build, test, lint

```bash
# Rust engine (fast inner loop — no Python needed)
cargo test -p pyroboframes-core
cargo clippy --workspace --all-targets -- -D warnings   # must be clean
cargo fmt --all

# With a decode backend feature enabled (stubs today)
cargo test -p pyroboframes-core --features cuda          # or videotoolbox / ffmpeg

# Python: build the extension into the current env, then run pytest
maturin develop
python3 -m pytest -q tests/
```

CI is intentionally not configured (no GitHub Actions). Releases are published manually with
`maturin publish` (a pre-release, `0.1.0a0`, is on PyPI).

## Conventions & invariants

- **Logic in core, not the binding.** `pyroboframes-py` is a thin shell; put real logic in
  `pyroboframes-core` so it's testable with `cargo test`. The binding owns a value type, not a
  borrow — `TabularLoader` holds `Arc<Dataset>` so PyO3 classes can own it.
- **Decode backends are `cfg`/feature-gated** behind the `Decoder` trait. `videotoolbox`,
  `ffmpeg`, `cuda` cargo features select real backends (currently stubs). To add one: implement
  `Decoder` for it under its feature, wire it into `Backend::preferred()`, keep the default
  build (no features) compiling and tests green.
- **Clippy must pass with `-D warnings`** across default and all backend features. The binding
  crate has a crate-level `#![allow(clippy::useless_conversion)]` because PyO3 0.22 macro
  codegen triggers it — don't remove it.
- **LeRobotDataset v3.0 schema assumptions** (in `episodes.rs` / `loader.rs`): episodes parquet
  columns are slash-named (`data/chunk_index`, `videos/<cam>/from_timestamp`, etc.); a data
  shard concatenates its episodes in global-frame order, so a shard's first global row is the
  min `dataset_from_index` of its episodes. If a real dataset disagrees, fix here.
- **Tests are self-contained:** Rust tests synthesize parquet with the `arrow`/`parquet`
  writers; Python tests synthesize a dataset with `pyarrow`. No network, no fixtures on disk.
- **Frames are not faked.** If a real codec can't be built/verified in the environment, leave
  the backend a stub and say so — do not fabricate decoded output.

## Don'ts

- Don't add a custom dataset container format — stay LeRobot-native (adoption).
- Don't move logic into the PyO3 layer.
- Don't claim a capability in docs that isn't implemented and tested.
