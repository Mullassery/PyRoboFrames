# PyRoboFrames — Honest Status

**Current Version:** v2.4.0
**Last Updated:** 2026-08-16
**Status:** Beta. Core LeRobot loading + native macOS video decode are solid and tested;
other formats and distributed/streaming features are real but less battle-tested.

This file exists to say plainly what works, what's rough, and what's aspirational —
`README.md` documents the intended public API; this file is the "have we actually
verified this" companion.

---

## 🟢 Solid (real implementation, real test coverage)

- **LeRobot v3.0 dataset reading** — native Rust core (`crates/pyroboframes-core`).
  Episode indexing, temporal windowing, train/val split, per-feature stats, batch
  loading with worker threads. Extensively covered by `cargo test` (75 unit tests) and
  `pytest` (`tests/test_loader.py`, `test_dataset_loaders.py`, etc.).
- **VideoToolbox hardware decode (macOS/Apple Silicon)** — a real, in-process
  `VTDecompressionSession`: MP4 demux + `CMSampleBuffer` construction in Rust, decoded
  frames come back as a real IOSurface-backed `CVPixelBuffer`. Not a shell-out to the
  `ffmpeg` CLI (a subprocess boundary can only hand back copied bytes; this stays
  in-process). Verified on real Apple Silicon hardware: `crates/pyroboframes-core/src/videotoolbox_native.rs`'s
  test module generates a real H.264 clip via `ffmpeg`, decodes it through the real
  `VTDecompressionSession` path, confirms genuine IOSurface backing, and cross-validates
  hardware-decoded pixels against `ffmpeg`'s software decode of the same bitstream.
  **Known scope limits:** H.264 only (no HEVC parameter-set extraction yet); decode-order
  reordering handles the no-B-frames case and isolated lookups correctly, not a full
  streaming reorder buffer.
- **FFmpeg / NVDEC fallback decode paths** — cross-platform, real (shells out to the
  `ffmpeg` CLI), used when `videotoolbox` isn't available or on Linux.
- **MCAP / ROS2 bag → Parquet conversion** — native Rust, real (not a stub); covered by
  `cargo test` (`mcap::tests`, `rosbag::tests`).
- **HDF5 / NetCDF / RLDS → LeRobot conversion** — real readers using `h5py` / `xarray`+
  `netCDF4` / `tensorflow_datasets` respectively (all optional dependencies; a clear
  `ImportError` is raised if missing, not a silent no-op). Covered by
  `tests/test_hdf5.py`, `test_netcdf.py`, `test_dataset_loaders.py` (RLDS tests
  `importorskip` if `tensorflow_datasets` isn't installed).
- **S3 / GCS dataset access** — real, via `fsspec` + `s3fs`/`gcsfs`. Downloads to a local
  cache directory and reads from there — **this is on-demand local caching, not a true
  zero-copy remote stream.** Construction and error-path tested
  (`tests/test_distributed.py`); a full download round-trip against a real bucket isn't
  exercised by the offline test suite (no test infrastructure for that).
- **Multi-framework array output** — NumPy (default), PyTorch (`device="cpu"/"cuda"/"mps"`),
  MLX, JAX. `pyroboframes.backend`/`transforms`/`unified_outputs` resolve a fallback
  chain (CV-CUDA → MLX → Torch → NumPy) so the same script degrades gracefully across
  hardware.
- **Distributed loading (PyTorch)** — `DistributedSampler`/`DistributedLoader` do real
  episode sharding with no overlap across ranks; unit-tested directly
  (`test_shard_episodes_*`).
- **Filtering / masking / quality scoring / augmentation / versioning** — real,
  dependency-free NumPy/Arrow implementations, each with direct test coverage.

## 🟡 Real but thinner coverage

- **Ray distributed loading (`RayDistributedLoader`)** — real integration code, but only
  its import/construction path is tested in this repo's offline suite (no `ray` cluster
  in CI). If you rely on this, test it against your own cluster before trusting it.
- **Streaming ingestion (MQTT/Kafka)** — real `MQTTStreamer`/`KafkaStreamer` with a
  thread-safe message buffer and time-windowed alignment; not exercised against a live
  broker in the test suite.
- **3D occupancy grid / LiDAR processing** (`occupancy_3d.py`) — real NumPy
  implementations; the surface-normal computation silently falls back to zero-vectors
  without `scikit-learn` installed (a `UserWarning` is emitted — check for it if you
  depend on real normals, don't just check the return shape).
- **GPU-acceleration transforms** (`gpu_acceleration.py`) — real CuPy/MLX/NumPy paths;
  the `scipy`-backed resize/filter paths need `scipy<1.13` under this package's pinned
  `numpy==1.24` (newer `scipy` requires `numpy>=1.26.4` and fails to import).

## 🔴 Known gaps / not done

- **True zero-copy array handoff (DLPack, skipping NumPy entirely)** — decode-to-buffer
  is zero-copy on macOS as of v2.3.0, but `Loader`'s batch path still copies frame bytes
  into one combined `[batch, H, W, 3]` NumPy array to build a batch from independent
  per-frame buffers. Not something zero-copy decode alone removes; still future work.
- **HEVC decode** in the native VideoToolbox path.
- **Only macOS (Apple Silicon) wheels published to PyPI.** No Linux or Windows wheel has
  been published; Linux/Windows users build from the source distribution (requires a
  Rust toolchain + `ffmpeg` at build time). Linux `aarch64` and Windows haven't been
  validated at all.
- **`numpy==1.24` pin friction** — hard-pinned for the compiled extension's ABI and the
  Parquet path, but recent `scipy`/`scikit-learn`/`pandas` releases require
  `numpy>=1.26`. Real, documented friction (see `SECURITY.md`), not silently papered
  over — pin companion packages below their numpy-2-requiring thresholds.
- **No adversarial-input hardening.** Format parsers (Parquet/MP4/HDF5/NetCDF/MCAP)
  assume trusted input; none have been fuzz-tested. `pyroboframes.security.validate_dataset_path()`
  exists but most entry points don't call it automatically — see `SECURITY.md`.

## Fixed this release (v2.4.0)

A round of correctness fixes surfaced during a fresh audit pass — see `CHANGELOG.md` for
full detail. Highlights: `RoboFrameDataset.num_episodes`/`.fps`/`.cameras` were PyO3
*properties*, but several call sites across the public API (`DatasetValidator`,
`EpisodeCache`, `EpisodeFilter`, `MaskedDataFrame`, `DatasetVersion`, every
`distributed.py` entry point, `EpisodeScorer`) called them as *methods*, which crashed
with `TypeError` on first use against a real dataset. `DatasetValidator.validate()`
crashed on every episode (`.dataset.path()` didn't exist — added a real `path` getter).
`transforms.Resize(interpolation="nearest")` crashed on the Torch backend and, once
fixed, was found to also violate its own documented dtype-preservation contract. Also
removed ~700 lines of dead code (`cli_workflow.py`, `server_workflow.py`, two `examples/mcp_*.py`
scripts, an unpackaged top-level `pyroboframes/` directory) that returned hardcoded fake
data and had zero real callers — leftover from an unrelated project template, along with
the README boilerplate from the same source.

---

## Hardware Support

Prebuilt wheels:
- ✅ macOS (Apple Silicon, arm64) — `videotoolbox` feature enabled
- ❌ macOS (Intel) — not published; should build from source with the `ffmpeg` feature
- ❌ Linux (x86_64/aarch64) — not published; build from source (`ffmpeg` feature)
- ❌ Windows — not supported/tested

## Dependencies

Hard-pinned (see `pyproject.toml` for the authoritative list):
```
numpy==1.24
pyarrow==14
```
Optional extras (`mlx`, `dev`) and format-specific optional imports (`h5py`, `xarray`,
`netCDF4`, `tensorflow_datasets`, `fsspec`+`s3fs`/`gcsfs`, `torch`, `jax`, `scipy`,
`scikit-learn`) are not hard dependencies — install what you need.
