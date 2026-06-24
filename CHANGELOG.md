# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding: Rust workspace (`pyroboframes-core`, `pyroboframes-py`), maturin/PyO3
  build, Python package skeleton.
- `ARCHITECTURE.md` and `docs/IMPLEMENTATION_PLAN.md` (design, options investigated, the gap,
  and the spike-first phased build plan).
- **Dataset reader (Phase 1a):** `info.json` parsing, schema/camera detection, shard-path
  resolution, and frame timestamps for LeRobotDataset v3.0 — platform-agnostic, with tests.
- **Episode index + tabular reader (Phase 1b):** read `meta/episodes/*.parquet` into an
  `EpisodeIndex` that resolves a global frame to `(camera, video file, timestamp)` via
  `locate()`; read `data/*.parquet` state/action vectors via `DataShard` (fixed-size / list
  float32). Backed by arrow/parquet, with self-contained parquet round-trip tests (12 total).
- **Decode architecture (Phase 2):** `decode` module with `Decoder` trait (incl. batched
  `decode_batch` seeks), `Frame`/`FrameBuffer`, `Backend` selection, a frame-buffer pool, and a
  decoded-frame `FrameCache` (LRU, the Robo-DM lever). Real VideoToolbox/FFmpeg backends are
  feature-gated stubs pending the Phase 0 spikes. Clippy clean with features on/off.
- **Sampler (Phase 3):** buffered / quasi-random shuffle (DALI/FFCV) — sequential read with a
  bounded shuffle window; deterministic and seedable, per-epoch reproducible.
- **Functional tabular dataloader:** `TabularLoader` (Rust) resolves a global frame to its
  episode + data-shard row and reads float features; **Python `RoboFrameDataset` + `Loader`**
  iterate NumPy batches of `observation.state` / `action` with shuffle, `drop_last`, and seeding.
  End-to-end tested from Python against a generated dataset (25 Rust + 7 Python tests).
- Pre-release version `0.1.0a0` (not yet published).

### Changed
- **Cross-platform target:** macOS *and* Linux are both first-class in v0.1. Decode is two
  `cfg`-gated backends behind one `Decoder` trait — VideoToolbox (macOS) and FFmpeg (Linux,
  VAAPI/NVDEC + software fallback).

### In progress (v0.1)
- Spike A (MLX zero-copy) and Spike B (VideoToolbox→IOSurface) — gate the decode design.
- VideoToolbox / FFmpeg decode backends; zero-copy MLX (macOS) and NumPy/PyTorch (Linux) output.
- Sampling/windowing, prefetch pipeline, validation pass, and the benchmark harness.

[Unreleased]: https://github.com/Mullassery/PyRoboFrames/commits/main
