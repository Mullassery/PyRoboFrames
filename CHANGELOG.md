# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-29

### Added
- **Video codec selection** — `write_lerobot_dataset(..., video_codec="hevc", video_crf=23)` now
  passes `-c:v libx264/libx265/libsvtav1` to FFmpeg; 40–50% storage savings with HEVC/AV1 vs H.264.
  New `encode_video_frames(frames, path, codec, crf)` helper for standalone video encoding.
- **Data validation toolkit** — `DatasetValidator` runs deep checks: missing video files, temporal
  gaps (`TemporalGapChecker`), frame count mismatches (`MissingFrameChecker`), codec decode health
  (`CodecHealthChecker`). Returns a structured `FullValidationReport` with per-issue severity.
  Rust `validate.rs` extended with cross-episode overlap and zero-duration video span detection.
- **Intelligent episode caching** — `EpisodeCache(dataset, max_episodes=4)` pre-decodes entire
  episodes into RAM with LRU eviction at episode granularity and non-blocking `prefetch()`.
  `loader(cache_size=N)` exposes configurable LRU frame cache capacity (previously hardcoded).
  `loader(episode_prefetch=True)` enables episode-boundary background prefetch.
- **Cross-dataset quality scoring** — `DatasetQualityProfile.from_scores()` builds percentile
  distributions (p25/p50/p75/p90) from `EpisodeScorer` output. `CrossDatasetComparator` ranks
  individual episodes against a reference distribution and computes Cohen's d + overlap coefficient.
  `compare_datasets(ds_a, ds_b)` is a convenience wrapper for quick cross-dataset comparison.
- **HDF5 support** — `HDF5Dataset.from_path()` reads ROBOMIMIC, ACT, and custom HDF5 layouts.
  `convert_hdf5(path, out_dir)` auto-detects episode groups (`demo_*`, `episode_*`, `traj_*`)
  and writes LeRobot v3.0 Parquet. Optional dep: `h5py>=3.0`.
- **NetCDF support** — `NetCDFDataset.from_path()` reads scientific simulation and robotics
  datasets. `convert_netcdf(path, out_dir)` auto-detects episode breaks from `done`/`terminal`
  variables or explicit `episode_breaks` array. Optional dep: `xarray` + `netCDF4`.
- **RLDS / Open X-Embodiment support** — `RLDSDataset.from_tfds(name)` and `convert_rlds(name,
  out_dir)` read HuggingFace Open X-Embodiment datasets. Maps `steps[i].observation.*` → feature
  columns, `steps[i].action` → action column. Optional dep: `tensorflow-datasets>=4.9`.
- **Remote dataset streaming** — `RemoteDataset.from_s3(uri)` / `.from_gcs(uri)` download
  LeRobot datasets from cloud storage on demand with background episode prefetch. Optional deps:
  `fsspec`, `s3fs`, `gcsfs`.
- **Ray distributed loader** — `RayDistributedLoader(dataset_path, rank, world_size)` shards
  episodes across Ray workers via round-robin. `shard_episodes(total, world_size, rank)` is a
  standalone helper for custom distributed setups. Optional dep: `ray>=2.0`.
- **Codec benchmark** — `benches/codec_comparison.py` prints a storage × speed table for H.264,
  HEVC, and AV1 with extrapolated 10k-frame estimates.

### Changed
- `write_lerobot_dataset()` gains `video_crf: int = 23` parameter; `video_codec` validation
  now raises `ValueError` (previously silent on unknown values).
- `RoboFrameDataset.loader()` gains `cache_size: Optional[int]` and `episode_prefetch: bool`
  kwargs (fully backwards-compatible; defaults match prior behaviour).
- `Cargo.toml` description updated to reflect production-ready status.

## [0.1.10] - 2026-06-27

### Added
- **LeRobot write-back** — `write_lerobot_dataset(path, features, episode_lengths, fps)` exports a
  tabular `LeRobotDataset v3.0` (info/episodes/data parquet + stats); round-trips through
  `RoboFrameDataset`.
- **Native dataset save** — `RoboticsDataFrame.save(path)` writes the columnar format (Parquet +
  `metadata.json` + `stats.json`), round-tripping via `from_converted`.
- **Hugging Face Hub importer** — `download_lerobot_dataset(repo_id, …)` (optional `huggingface_hub`).
- **Multi-rate resampling / fusion** — `RoboticsDataFrame.resample(period, method="previous"|
  "nearest"|"linear")` fuses multi-rate topics onto one uniform time grid.
- **Curriculum sampling** — `loader(curriculum=True)` orders the epoch easy→hard (shorter episodes
  first).
- **Goal-conditioned sampling** — `loader(goal="final")` adds `<feature>.goal` (the episode's final
  frame) to each sample.
- **Windowed video sync** — `delta_timestamps` now applies to cameras too, yielding
  `[batch, steps, H, W, 3]` temporal frame stacks.
- **Memory-mapped data shards** — `data/*.parquet` shards are mmap-backed (lower resident memory on
  large datasets), values unchanged.
- **Backend parity** — `default_framework(device)` + `to_backend(obj, device)` (unified output
  abstraction) and `transforms.resolve_transform_backend()` (CV-CUDA → Torch → NumPy fallback chain).
- **NVIDIA NVDEC decoder** (`--features cuda`) — `CudaDecoder` drives `ffmpeg -hwaccel cuda`,
  sharing the CLI path with the FFmpeg backend. Compile-/lint-clean; functional verification
  deferred to NVIDIA hardware.

## [0.1.9] - 2026-06-27

### Added
- **protobuf MCAP decoding** — `convert_mcap` now decodes `protobuf` topics dynamically from the
  channel's embedded `FileDescriptorSet` (no codegen), flattening to columns like JSON.
- **ros2msg / CDR decoding** — `cdr` topics decode against the parsed `ros2msg` schema (primitives,
  fixed/bounded/unbounded arrays, strings, nested message types) via a new `core::ros2` XCDR1 reader.
- **ROS 2 bag converter** — `convert_ros2_bag(input, out_dir)` reads a rosbag2 SQLite `.db3` (topics
  + CDR blobs + embedded `message_definitions`) and writes one Parquet table per CDR topic.
- **Automatic dataset metadata** — both converters now emit `metadata.json` (a self-describing
  manifest: per-topic path, row count, `log_time` range, column dtypes) and `stats.json`
  (per-column count/mean/std/min/max, loader-compatible) alongside the Parquet.
- **`RoboticsDataFrame`** — a typed, time-indexed, multi-sensor view over converted output:
  per-topic `TopicFrame` access, `time_range()`, `slice(start, end)`, and `align(reference,
  tolerance=…)` (backward as-of join for time-synchronized multi-sensor fusion). Construct via
  `from_converted` / `from_mcap` / `from_ros2_bag`.

### Fixed
- Declared `numpy` + `pyarrow` as runtime dependencies — a bare-venv `pip install pyroboframes`
  previously crashed on `import pyroboframes` (numpy is imported at package import time).

## [0.1.8] - 2026-06-27

### Added
- **Episode-chunking sampler** — `loader(chunk_size=N)` cuts each episode into contiguous
  N-frame chunks, shuffles the chunks as units, and keeps frames in temporal order inside a chunk.
  Sequence-friendly batches with decode locality; never crosses an episode boundary. Core
  `sampler::chunked_order` + `TabularLoader::episode_runs`.
- **MLX sequence batching + benchmarks** — temporal-window (`delta_timestamps`) + episode-chunked
  `[batch, steps, dim]` sequences now feed `output="mlx"` directly; `benches/throughput.py` gains
  an output-framework (numpy/mlx/torch/jax) comparison and a sequence-batching section.
- **MCAP → columnar (Parquet) converter** (Tier-1 data-platform milestone) — `convert_mcap(input,
  out_dir)` reads an MCAP robotics log and writes one flattened Parquet table per JSON topic
  (dot-path leaf columns + `log_time`); non-JSON topics (protobuf/ros2msg/cdr) are reported as
  skipped. Core `mcap::convert`.

## [0.1.7] - 2026-06-25

### Added
- **JAX output** — `loader(output="jax")` returns `jax.numpy` arrays (alongside numpy/mlx/torch).
- **Real transforms + augmentations** — `transforms.Resize` now defaults to **bilinear**
  (`interpolation="nearest"` still available), plus `RandomHorizontalFlip`, `RandomCrop`,
  `ColorJitter` (all seedable, per-sample).
- **Loader profiling** — `DataLoader(on_batch=…)` callback per batch and `loader.stats`
  (batches / frames / seconds / frames_per_s).

## [0.1.6] - 2026-06-25

### Added
- **Backend / device selection** (`pyroboframes.backend`) — `resolve_device("auto")` picks
  `cuda → mlx/mps (Apple) → cpu` at runtime, honoring the `PYROBOFRAMES_DEVICE` override;
  `available_backends()` for diagnostics.
- **Image transforms** (`pyroboframes.transforms`) — `Compose`, `Resize`, `CenterCrop`,
  `Normalize` over `[N, H, W, C]` camera batches (NumPy; the op surface CV-CUDA/MLX will reuse).
- **`DataLoader` wrapper** — wraps a loader, applies transforms to camera frames, and moves each
  batch to the resolved device/framework (NumPy / Torch[cuda|mps] / MLX) — same loop everywhere.
- **Balanced sampling** — `loader(balanced=True)` draws frames so every episode is sampled
  equally regardless of length (weighted sampling with replacement). Core
  `sampler::weighted_with_replacement`.

## [0.1.5] - 2026-06-25

### Added
- **Off-GIL prefetch pipeline** — `loader(num_workers=N, prefetch=M)` assembles batches on N
  background threads ahead of consumption (token-bounded in-flight count, reorder buffer keeps
  epoch order; the blocking wait releases the GIL). `num_workers=0` (default) stays synchronous.
  New core `pipeline` module (`RustBatch`, `BatchAssembler`, `Prefetcher`).
- **Normalization** — `loader(normalize=["observation.state", ...])` applies `(x - mean) / std`
  from `meta/stats.json` (zero std treated as 1).
- **Episode iteration** — `ds.episodes()` returns per-episode metadata
  (`episode_index`, `length`, `from_index`, `to_index`).
- **Throughput benchmark harness** — `benches/throughput.py`: frames/s for a full epoch across
  `num_workers`, tabular and (if `ffmpeg` present) camera-decode. On synthetic data (Apple
  Silicon) the camera path scales ~2.7× with 4 workers vs synchronous.

## [0.1.4] - 2026-06-25

### Changed
- **Honesty / metadata only (no code changes).** The PyPI summary and the top of the README now
  lead with an **early / experimental** note, and the package description drops the
  "zero-copy, hardware-accelerated" wording — those paths (HW decode, zero-copy MLX, parallel
  prefetch) are still in progress and throughput is not yet benchmarked.

## [0.1.3] - 2026-06-25

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
