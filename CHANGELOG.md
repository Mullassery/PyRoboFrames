# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`ds.stats()`** — per-feature statistics from `meta/stats.json` (`mean`/`std`/`min`/`max`/
  `count`) for normalization, returned as a dict; `None` when the file is absent. Tolerant parser
  (flattens nested image-channel stats, ignores non-numeric fields).
- **`ds.train_val_split(val_fraction, seed)`** — deterministic split over **episode** indices
  (never by frame, to avoid temporal leakage), returning sorted `(train, val)` episode lists.
- **Loader checkpoint/resume** — `loader.position` (frames consumed this epoch) and
  `loader.seek(position)` to resume an interrupted epoch on a fresh, identically-seeded loader.
- **`loader(episodes=[...])`** — restrict iteration to a set of episode indices; pass one half of
  `ds.train_val_split(...)` for a train- or validation-only loader. Core:
  `TabularLoader::frame_indices_for_episodes`.
- Core: `stats` + `split` modules, shared `rng` (SplitMix64) module; `Dataset::stats` /
  `Dataset::train_val_split`. +13 tests (8 Rust, 5 Python).

## [0.1.2] - 2026-06-25

### Added
- **`output=` on the loader:** `"numpy"` (default), `"mlx"` (`mlx.core.array`), or `"torch"`
  (`torch.from_numpy`, zero-copy from the NumPy buffers). Converts every array in the batch —
  state/action, windows, and camera frames. Verified against torch 2.12 and mlx.

### Notes
- Closes "MLX/PyTorch output" as **functional**. The *true* zero-copy MLX path (decode →
  IOSurface → MLX with no NumPy hop) and native VideoToolbox/NVDEC backends remain in progress
  (zero-copy MLX is gated on upstream mlx#2855).

## [0.1.1] - 2026-06-25

### Added
- **Camera frame decoding (FFmpeg).** Real `FfmpegDecoder` (drives the `ffmpeg` CLI; works on
  macOS + Linux, uses platform hwaccel where the ffmpeg build supports it). `loader(cameras=[...])`
  now yields `[batch, H, W, 3]` `uint8` frame arrays alongside state/action. Requires
  `ffmpeg`/`ffprobe` on `PATH`; built into the wheel via the `ffmpeg` feature.
- **`ds.validate()`** — metadata integrity checks (frame-range contiguity, episode lengths,
  per-camera timestamp bounds, totals vs `info.json`) returning a `ValidationReport`.
- Core APIs: `TabularLoader::frames_for/locate/dataset`, `decode::FfmpegDecoder`,
  `validate` module, `Dataset::validate`.

### Notes
- Still in progress: native VideoToolbox/NVDEC backends and **zero-copy MLX** output (frames are
  NumPy today; convert to MLX/PyTorch in one line).

## [0.1.0] - 2026-06-25

First public PyPI release (`pip install pyroboframes`). Functional tabular dataloader; video
decode in progress. `0.x` — API may change.

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
- **Temporal windowing (Phase 3):** `window` module (`delta_timestamps` → in-episode frame
  offsets, nearest-frame snap with `tolerance_s`, edge-clamp); `TabularLoader::windowed_sample`
  and the Python `loader(delta_timestamps=..., tolerance_s=...)` returning `[batch, steps, dim]`.
- **Decode integration (Phase 2):** `decode_frames` resolves each camera's video shard + timestamp
  and decodes via the `Decoder` + `FrameCache` (tested with a mock decoder; real codecs drop in).
- **Linux + CUDA backend:** `Backend::Cuda` (NVIDIA NVDEC) + `cuda` cargo feature; selected on
  Linux when built `--features cuda`. Stub pending CUDA toolkit integration.
- Released as **0.1.0** (stable scheme, no `--pre` needed). 32 Rust + 8 Python tests; clippy
  clean across default and all backend features (`videotoolbox`, `ffmpeg`, `cuda`).

### Changed
- **Cross-platform target:** macOS *and* Linux are both first-class in v0.1. Decode is two
  `cfg`-gated backends behind one `Decoder` trait — VideoToolbox (macOS) and FFmpeg (Linux,
  VAAPI/NVDEC + software fallback).

### In progress (v0.1)
- Spike A (MLX zero-copy) and Spike B (VideoToolbox→IOSurface) — gate the decode design.
- VideoToolbox / FFmpeg decode backends; zero-copy MLX (macOS) and NumPy/PyTorch (Linux) output.
- Sampling/windowing, prefetch pipeline, validation pass, and the benchmark harness.

[Unreleased]: https://github.com/Mullassery/PyRoboFrames/commits/main
