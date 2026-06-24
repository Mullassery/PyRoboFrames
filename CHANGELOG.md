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

### Changed
- **Cross-platform target:** macOS *and* Linux are both first-class in v0.1. Decode is two
  `cfg`-gated backends behind one `Decoder` trait — VideoToolbox (macOS) and FFmpeg (Linux,
  VAAPI/NVDEC + software fallback).

### In progress (v0.1)
- Per-episode index + tabular parquet reading (Phase 1b) for exact frame location.
- VideoToolbox / FFmpeg decode backends; zero-copy MLX (macOS) and NumPy/PyTorch (Linux) output.
- Sampling/windowing, prefetch pipeline, validation pass, and the benchmark harness.

[Unreleased]: https://github.com/Mullassery/PyRoboFrames/commits/main
