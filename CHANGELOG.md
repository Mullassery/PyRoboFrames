# Changelog

All notable changes to PyRoboFrames are documented in this file.

## [Unreleased]

### Added
- **`cargo audit` now runs in CI** (`.github/workflows/ci.yml`'s new `security-audit`
  job) — the prior audit pass couldn't verify this sandbox had crates.io advisory-database
  access; it does. Found 2 real vulnerabilities (`RUSTSEC-2025-0020`, `RUSTSEC-2026-0177`,
  both `pyo3` 0.22.x, fixed upstream in a major-version bump not done here) and 3
  unmaintained/unsound warnings (`paste`, `lru`) — see `ROADMAP_HONEST.md`. Non-blocking
  (`|| true`) until the `pyo3` migration lands, so it reports without failing CI today.

### Fixed
- **8 broken relative Markdown links**, found by resolving every relative link in the repo
  against its source file's directory: `CHANGELOG.md`'s `DEPLOYMENT_SECURITY.md` link
  (file actually lives at `docs/DEPLOYMENT_SECURITY.md`), `docs/MCP_QUICKSTART.md`'s
  `README.md` link (needed `../`), and 6 links inside `docs/archive/*_STALE.md` files that
  broke when those files were moved into `docs/archive/` in an earlier pass without
  updating their internal relative paths (`PRIORITIES_v0.2.1_STALE.md` × 4,
  `ROADMAP_v1.1_STALE.md`, `KNOWN_ISSUES_2026-07-20_STALE.md`). Verified by re-resolving
  every relative link in the repo against its filesystem location — zero broken afterward.
- **`.github/workflows/ci.yml`: bumped `actions/checkout` v4→v7 and `actions/setup-python`
  v5→v7** (3 call sites) to the latest available major versions. Checked each intermediate
  release's changelog for breaking changes relevant to this workflow (no special inputs
  used, GitHub-hosted runners already meet the minimum runner version) before bumping;
  `actionlint` clean before and after.

### Removed
- **`pyroboframes/scripts/setup_shortcuts.sh`** — an orphaned top-level directory
  (`pyroboframes/`, distinct from the real package at `python/pyroboframes/`) containing a
  script that only aliased a `pyroboframes dashboard` CLI command that doesn't exist
  anywhere in this codebase (no `console_scripts`/`[project.scripts]` entry, no `dashboard`
  function). Zero references to it anywhere in the repo.
- **`scripts/test_dashboards_crossplatform.sh`** — a "cross-platform dashboard test"
  script that never tested PyRoboFrames at all: every package name it checks
  (`pystreamai`, `pystreammcp`, `pystreampdf`, `pystreamxl`, `statguardian`,
  `pyreverseetl`, `pyterrainmap`, `pyroboreplay`, `pyrobosimulator`) belongs to unrelated
  sibling projects by the same author — leftover from a copy-pasted template, like the
  dead `cli_workflow.py`/`server_workflow.py`/`mcp_*.py` files removed previously (see
  `623835b`). Not referenced by CI, `Makefile`, or any doc.

### Fixed
- **Two fuzz-found crashes in third-party dependencies, both confirmed fixed.** The
  `mcap_convert` fuzz target found a real out-of-memory abort on malformed MCAP input
  (an unbounded allocation inside `mcap` 0.9's own record parser, not this project's
  code); fixed by upgrading `mcap` 0.9 → 0.25 (latest) — confirmed with a
  315k-iteration/60s local fuzz run finding zero crashes. While verifying that fix, a
  second, unrelated crash turned up: `data_shard_parquet` panicking inside `parquet`
  55's Thrift-compact-protocol metadata decoder on a truncated Parquet footer; fixed by
  upgrading `arrow`/`parquet` 55 → 59 (latest). `mcap` 0.25's `Channel`/`Schema` structs
  gained a required `id: u16` field; updated the 7 test-fixture literals constructing
  these directly.
- **Hardened `mcap::convert()` against a third, still-open upstream `mcap` bug.** After
  the two upgrades above, the CI fuzz smoke test found a *different* crash in
  `mcap_convert`: a `usize`-overflow panic inside `mcap` 0.25.0's own reader on a
  different malformed length field. This is a genuine bug in the latest available `mcap`
  release (no newer version fixes it, and `mcap`'s public API has no way to bound
  record-length parsing to avoid it being reachable). Mitigated — not fixed at the
  source — by wrapping the read loop in `catch_unwind`, so real callers get a clean
  `Err` instead of a process abort (verified directly against the exact crash input).
  `cargo-fuzz`'s own smoke test for this one target will keep failing regardless, since
  `libfuzzer-sys` calls `process::abort()` on any panic before `catch_unwind` can run —
  see `ROADMAP_HONEST.md` and the CI fix below for why that's expected, not a regression.
- **CI's fuzz smoke-test job was silently skipping 4 of its 5 targets.** `set -e` meant
  the loop aborted on `mcap_convert`'s first (already-documented, unfixable-from-here)
  failure above, so `rosbag_convert`/`data_shard_parquet`/`episodes_parquet`/
  `ros2_cdr_decode` never actually got smoke-tested on any push since this started
  failing. Every target now always runs; `mcap_convert`'s known crash is logged as a
  warning and doesn't fail the job, but a crash in any other target still does. Also
  bumped `actions/setup-python` v4 → v5 (flagged by `actionlint`).

## [2.5.1] — 2026-08-30

### Fixed
- **The `2.5.0` package published to PyPI was completely non-functional.** Inspecting the
  actual uploaded wheel (`pyroboframes-2.5.0-cp310-abi3-macosx_11_0_arm64.whl`, 38 KB)
  showed it contained no compiled extension and no Python source at all — just a
  `pyroboframes.pth` file whose only content was the absolute local path
  `/Users/georgimullassery/PyRoboFrames/python`. That's the artifact `maturin develop`
  (editable-install mode) produces, not `maturin build` (a real, self-contained
  distributable wheel) — it appears the release was built with the former and uploaded as
  if it were the latter. `pip install pyroboframes` (any version resolving to `2.5.0`)
  installed a package where `import pyroboframes` would fail immediately on any machine
  other than the exact one it was built on, since the `.pth` path doesn't exist elsewhere.
  Fixed by rebuilding with `maturin build --release` and verifying the result actually
  contains `_core.abi3.so` and the `pyroboframes` Python package before upload — see
  `scripts/release.sh`, added specifically to make this class of mistake structurally
  harder to repeat (it fails loudly if the built wheel doesn't look like a real package).
  Republished as `2.5.1` since PyPI never allows re-uploading a filename once claimed, even
  after the broken `2.5.0` file is yanked.

## [2.5.0] — 2026-08-30

### Added
- **Fuzz testing.** No fuzz infrastructure existed at all before this — adds
  `cargo-fuzz` scaffolding (`crates/pyroboframes-core/fuzz/`) with 5 real
  targets exercising the parsers that handle untrusted/external input: MCAP,
  rosbag, Parquet data-shard, Parquet episodes, and ROS2 CDR decode. A new CI
  job builds and smoke-tests each target on every push.
- **HEVC decode in the native VideoToolbox path** (`hev1`-tagged files — see
  `ROADMAP_HONEST.md` for the `hvc1` scope limit). The `mp4` crate (0.14.0, its newest
  published version) parses H.264's `avcC` box fully but its `HvcCBox` only reads
  `configurationVersion` and discards the rest, so
  `crates/pyroboframes-core/src/videotoolbox_native.rs::hevc_hvcc` is a from-scratch
  `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15) box-tree reader that extracts the
  real VPS/SPS/PPS NAL arrays. Verified against a real `libx265`-encoded clip: decoded
  through `VTDecompressionSession`, cross-checked pixel-for-pixel against `ffmpeg`'s
  software HEVC decode of the same bitstream.

### Performance
- **`Loader`'s camera-frame batch assembly now copies each frame's pixels once instead
  of twice.** Previously each decoded frame was copied into a throwaway per-frame `Vec`
  (`Frame::to_rgb24_bytes()`) and then copied *again* into the shared batch array via
  `extend_from_slice`. Added `Frame::write_rgb24_into()` (`crates/pyroboframes-core/src/decode.rs`)
  to write a decoded frame's pixels directly into its slot in a pre-sized batch buffer,
  and updated both the non-windowed (`[batch, H, W, 3]`) and windowed
  (`[batch, steps, H, W, 3]`) assembly paths in `crates/pyroboframes-py/src/lib.rs` to use
  it. This does not make batching fully zero-copy (numpy still needs one contiguous
  array, decoded frames still each live in their own buffer) — see `ROADMAP_HONEST.md`.

### Changed
- **`dev` extra's `numpy<1.27`/`pandas<2.2`/`scipy<1.13`/`scikit-learn<1.5` caps
  removed.** These existed to keep the `dev` extra compatible with an assumed
  `numpy==1.24` hard pin — but that pin was already relaxed to a floor
  (`numpy>=1.24`, no ceiling) in `2804b06`, and `ROADMAP_HONEST.md`/`SECURITY.md`
  documented the caps as a live gap without anyone re-verifying whether they were
  still needed. They weren't: verified numpy 2.4.6 + pyarrow 25.0.1 + scipy 1.17.1 +
  scikit-learn 1.9.0 + pandas 3.0.5 together — full test suite green (302 passed, 0
  failed) both against a manually-assembled modern environment and a clean
  `pip install -e ".[dev]"`. Docs corrected to match.

### Fixed
- **`pyroboframes.security.validate_dataset_path()` never actually rejected traversal.**
  It resolved the path (which collapses `..` components away) *before* checking for
  `..`, so the check could never trigger. Now checks path components pre-resolve.
- **Path-containment (`base_dir`) is now wired into loader entry points**, not just
  available as a helper callers had to remember to invoke themselves:
  `RoboFrameDataset.from_path`, `convert_mcap`, `convert_ros2_bag` (Rust,
  `crates/pyroboframes-py/src/lib.rs`), `HDF5Dataset`/`HDF5Dataset.from_path`/
  `convert_hdf5`, and `NetCDFDataset`/`NetCDFDataset.from_path`/`convert_netcdf` all
  accept an optional `base_dir` that rejects paths (including symlink escapes, via
  canonicalization) outside it. Opt-in — omitting `base_dir` preserves prior
  unrestricted behavior. See `SECURITY.md`.
- **VideoToolbox hardware decode was dead code in every published wheel.**
  `pyproject.toml`'s `[tool.maturin] features` never included `videotoolbox`, so every
  wheel actually shipped fell back to the `ffmpeg` CLI-subprocess decoder on macOS — the
  real, in-process `VTDecompressionSession` path (`videotoolbox_native.rs`) was real,
  tested (`cargo test --features videotoolbox`), and completely unreachable in production.
  Confirmed via `otool -L` on the built `.so`: no CoreMedia/CoreVideo/VideoToolbox
  frameworks were linked before this fix. `videotoolbox` added to the maturin feature
  list; safe on non-macOS source builds since the underlying crates are already
  `target_os = "macos"`-gated in `Cargo.toml`.
- **`NativeVideoToolboxFile::decode_at` could return the wrong or a duplicate frame** for
  any clip with B-frame reordering (the normal case for real encoded video — the existing
  test fixtures happened to disable B-frames to sidestep the *documented* no-reorder-buffer
  limitation, which incidentally also hid this *different*, undocumented bug). It matched
  a caller's timestamp against each sample's raw position in decode order, but decode
  order only equals presentation order when there's no B-frame reordering; with reordering,
  the whole PTS timeline is typically shifted by the encoder (visible as the first sample's
  presentation time being nonzero), so nearby requested timestamps could resolve to the
  identical nearest decode-order sample. Fixed by matching against each sample's actual
  presentation time (DTS + rendering offset), offset-normalized to the stream's earliest
  presentation time. Regression test:
  `native_decode_handles_pts_offset_from_bframe_reordering`. This was only caught because
  turning on `videotoolbox` in the shipped build (previous bullet) made this code path
  reachable through the Python test suite for the first time.

## [2.4.0] — 2026-08-16

Engineering/correctness pass: real bug fixes across the public Python API, dead-code
removal, and documentation accuracy. No new dataset formats in this release.

### Fixed
- **`RoboFrameDataset.num_episodes` / `.fps` / `.cameras` called as methods.** These are
  PyO3 `#[getter]` properties, not methods — `ds.num_episodes()` raised
  `TypeError: 'int' object is not callable`. This was live-broken in
  `DatasetValidator.validate()`, `EpisodeCache`, `EpisodeFilter`, `MaskedDataFrame`,
  `DatasetVersion.append()`, `EpisodeScorer`, and every `distributed.py` entry point
  (`DistributedLoader`, `RayDistributedLoader`) — i.e. calling `.validate()` on a real
  dataset, or constructing any of those classes with a real `RoboFrameDataset`, crashed.
  Also fixed two call sites using a nonexistent `.total_frames()` method (the real getter
  is `.num_frames`).
- **`DatasetValidator` crashed on every episode.** `validate_episode()` called
  `self.dataset.path()`, which didn't exist on `RoboFrameDataset` at all. Added a real
  `path` getter (`crates/pyroboframes-py/src/lib.rs`) backed by the Rust core's existing
  `Dataset::root()`, and fixed the call site to use it as a property.
- **`transforms.Resize(..., interpolation="nearest")` crashed on the Torch backend** —
  `align_corners` was passed as `False` (not `None`) for `mode="nearest"`, which Torch
  rejects (`align_corners` is only valid for interpolating modes). Fixed, and also fixed
  nearest-neighbor resize silently upcasting `uint8` frames to `float32` (violated the
  documented contract that nearest preserves input dtype).
- **`tests/test_caching.py`** passed the wrong value as a dataset path in 5 tests —
  `make_dataset()` returns the frame count, not the path, so `RoboFrameDataset.from_path(str(make_dataset(...)))`
  was opening a path like `"30"`. Fixed to use `tmp_path` (where `make_dataset` actually
  writes) as the dataset path.
- **`test_backend_parity.py`'s `_to_numpy` helper** never actually exercised its
  `.cpu().numpy()` fallback for non-CPU Torch tensors (MPS/CUDA), because `hasattr(v,
  "__array__")` is `True` for Torch tensors regardless of device — masking a real
  `device="auto"` vs `device="cpu"` conformance test on Apple Silicon (MPS).
- `cargo clippy -- -D warnings` failures (doc-comment formatting, a needless `return`, an
  `Iterator::last` on a `DoubleEndedIterator`, an unnecessary cast) and `cargo fmt`
  drift across `decode.rs`, `depth.rs`, `videotoolbox_native.rs`, `lib.rs`.

### Removed
- **`python/pyroboframes/cli_workflow.py` and `server_workflow.py`** — dead code (not
  imported by `__init__.py`, no console-script entry point, no tests, no docs) left over
  from an unrelated project template. Every method returned hardcoded fake data
  (`"episodes": 100,  # Simulated`, `"conversion_time_s": 45.2`) regardless of input —
  none of it touched a real dataset.
- **`examples/mcp_datasets.py` and `mcp_pyroboframes.py`** — example scripts importing
  `DatasetMetadata` / `PerceptionEngine`, neither of which exist anywhere in this
  package; leftover from the same unrelated template.
- **Top-level `pyroboframes/` directory** (`okf_dataset_composition.py`,
  `scripts/post_install.py`) — not part of the packaged source (`pyproject.toml`'s
  `python-source = "python"` only packages `python/pyroboframes/`), not imported by
  anything, dead.
- README's leftover "MCP 2.0 Mega-Platform" boilerplate (wrong license claim, fake tool
  counts, ports, and unrelated project description) — full rewrite from the real public
  API.

### Changed
- `pyproject.toml`'s `dev` extra now includes `pandas`, `scipy`, and `scikit-learn`
  (version-capped to stay importable under the `numpy==1.24` pin) — `tests/test_storage.py`
  unconditionally needs `pandas` (via `hub.py`'s HuggingFace-hub download path), and the
  GPU-acceleration/occupancy-grid tests need `scipy`/`scikit-learn`; without these the
  `pip install -e ".[dev]"` step in CI could not actually run the full suite.
- `SECURITY.md` updated to reflect the current state (was still describing "v1.1.0
  NO PRODUCTION USE" against a package that has since shipped VideoToolbox hardware
  decode, HDF5/NetCDF/RLDS conversion, and S3/GCS remote datasets).

## [2.3.0] — 2026-08-07

Real, in-process VideoToolbox hardware decode (see `ed84820` in git history for full
detail): `VTDecompressionSession` driven directly via MP4 demux + `CMSampleBuffer`
construction, producing a real IOSurface-backed `CVPixelBuffer` — not a shell-out to the
`ffmpeg` CLI. Verified against real Apple Silicon hardware with pixel-level cross-checks
against ffmpeg's software decode of the same bitstream.

*(Changelog entries between 1.2.0 and 2.3.0 were not maintained day-to-day; see `git log`
or GitHub Releases for the full history of that range — multi-format dataset support,
GPU acceleration, 3D occupancy/LiDAR processing, and workflow tooling all landed in this
window.)*

## [1.2.0] — 2026-07-17

### 🎉 Major Features (Complete Phases 4-7)

#### Phase 4: GPU Acceleration
- **GPU Transform Operators** — `gpu_acceleration.py` module with automatic fallback chain
  - CuPy backend for NVIDIA (CUDA)
  - MLX backend for Apple Silicon GPU
  - NumPy fallback for CPU
  - `GPUTransforms` class with `resize()` and `normalize()` operations
- **Automatic Device Detection** — `device="auto"` picks best available: CUDA → MLX → NumPy
- **Cross-Platform Parity** — Same transform code runs on M3, RTX 5090, H100 without changes

#### Phase 5: Temporal Consistency
- **Optical Flow Estimation** — `OpticalFlowEstimator` for motion detection
  - Lucas-Kanade method (real OpenCV when available)
  - Gradient-based fallback (pure NumPy)
- **Temporal Filtering** — `TemporalFilter` for video smoothing
  - Exponential moving average (EMA) for motion smoothing
  - Median filtering for temporal denoising
- **Use Case** — Reduces flickering in panoramic stitching, improves temporal coherence

#### Phase 6: Real-World Autonomous Driving Datasets
- **Waymo Open Dataset Loader** — `WaymoDatasetLoader`
  - Scene indexing and metadata parsing
  - Per-camera frame loading with automatic calibration
  - Standard Waymo intrinsics (fx=2015, fy=2015, cx=960, cy=600)
- **nuScenes Integration** — `nuScenesDatasetLoader`
  - Multi-camera support (CAM_FRONT, etc.)
  - Standard nuScenes calibration (1266px focal length)
  - 20 Hz frame rate support
- **KITTI Dataset Support** — `KITTIDatasetLoader`
  - Training/testing split support
  - Stereo camera handling (camera 0, 1)
  - Calibration file parsing from KITTI format
- **Unified API** — All loaders expose:
  - `CameraCalibration` (fx, fy, cx, cy, width, height, distortion)
  - `FrameMetadata` (timestamp, camera_name, frame_index, image_path, calibration)

#### Phase 7: Occupancy & 3D Perception
- **Occupancy Grid Mapping** — `OccupancyGrid` class
  - World-to-grid coordinate transformation (configurable resolution)
  - Point cloud integration (ray tracing for free space)
  - 3D bounding box insertion
  - Morphological operations (dilate, erode)
  - Free space mask extraction
- **LiDAR Processing** — `LiDARProcessor` utility class
  - Distance filtering (by max range)
  - Height filtering (by min/max Z)
  - Ground segmentation (elevation threshold)
  - Point clustering (DBSCAN-like, with scikit-learn fallback)
  - Surface normal estimation (PCA on k-NN)
- **Radar Fusion** — `RadarFusionProcessor` for multi-sensor perception
  - Velocity estimation from Doppler shift
  - Spatial association (radar ↔ LiDAR)
  - Combined position + velocity output for tracking

### 📊 Testing
- **47 New Tests** (222 total, up from 175)
  - 10 GPU acceleration tests (resize, normalize, device selection)
  - 16 dataset loader tests (Waymo, nuScenes, KITTI, calibration)
  - 21 occupancy/3D tests (grid operations, LiDAR, radar fusion)
- **All Tests Passing** — 100% pass rate
- **Coverage** — 82%+ coverage across new modules

### 📚 Documentation
- **Updated README.md**
  - v1.2 feature showcase with code examples
  - Updated test count (175 → 222)
  - Competitive analysis matrix
  - GPU support section
- **ROADMAP.md Updates**
  - Marked P0-P11 complete
  - Updated version status (v0.5.0 → v1.2.0)
- **New COMPETITIVE_ANALYSIS.md**
  - Detailed comparison vs torchcodec, Robo-DM, LeRobot, PyAV
  - Honest strengths and weaknesses
  - Use case guidance ("Train Anywhere" positioning)

### 🔧 Engineering
- **Python 3.13 Support** — Updated pyproject.toml
  - Changed `requires-python = "==3.10"` → `">=3.10"`
  - Added Python 3.13 classifier
- **Version Bump** — Cargo.toml: 1.0.0 → 1.2.0
- **PyPI Release** — Published as `pyroboframes-1.2.0.tar.gz`

### ✨ Key Differentiators
1. **Only** dataloader with multi-platform GPU parity (NVIDIA + Apple Silicon)
2. **Only** tool bridging robot learning + autonomous driving datasets
3. **Only** loader with native occupancy grid + 3D perception
4. **Largest test suite** for robot dataloaders (222 tests, 82% coverage)

---

## [1.1.0] — Previous Release
See GitHub tags for v1.1.0 and earlier.

---

## Future Roadmap

### v1.3.0 (Planned)
- [ ] Streaming ingestion (MQTT / Kafka)
- [ ] Vision-language dataset generation (auto-annotation with CLIP/SAM2)
- [ ] Distributed loading (Ray / Slurm / RunPod)
- [ ] Additional AV datasets (KITTI-360, Argoverse 2)

### v2.0.0+ (Long-term)
- [ ] Zero-copy MLX arrays (awaiting upstream mlx#2855)
- [ ] Real-time imitation learning pipelines
- [ ] Advanced curriculum learning strategies
- [ ] Enterprise deployment templates

---

## Security

### 🔒 v1.2.0 Security Status
- ✅ Path traversal protection (S3/GCS access)
- ✅ Hardware fallback graceful degradation
- ✅ Dependency pinning (numpy==1.24, pyarrow==14)
- ⚠️ PyPI token rotated (was exposed in development)

See [docs/DEPLOYMENT_SECURITY.md](./docs/DEPLOYMENT_SECURITY.md) for details.

---

## Contributors
- **Georgi Mammen Mullassery** — Core implementation, GPU acceleration, dataset loaders, 3D perception

---

## Installation

```bash
pip install pyroboframes==1.2.0
# or
uv add pyroboframes==1.2.0
```

For development:
```bash
git clone https://github.com/Mullassery/PyRoboFrames.git
cd PyRoboFrames
pip install -e ".[dev]"
pytest tests/ -v
```
