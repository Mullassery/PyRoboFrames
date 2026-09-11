# PyRoboFrames — Honest Status

**Current Version:** v2.5.1
**Last Updated:** 2026-09-11
**Status:** Beta. Core LeRobot loading + native macOS video decode are solid and tested;
other formats and distributed/streaming features are real but less battle-tested.

This file exists to say plainly what works, what's rough, and what's aspirational —
`README.md` documents the intended public API; this file is the "have we actually
verified this" companion.

---

## 🟢 Solid (real implementation, real test coverage)

- **LeRobot v3.0 dataset reading** — native Rust core (`crates/pyroboframes-core`).
  Episode indexing, temporal windowing, train/val split, per-feature stats, batch
  loading with worker threads. Extensively covered by `cargo test` (324 unit tests) and
  `pytest` (`tests/test_loader.py`, `test_dataset_loaders.py`, etc. — 323 tests total).
- **VideoToolbox hardware decode (macOS/Apple Silicon)** — a real, in-process
  `VTDecompressionSession`: MP4 demux + `CMSampleBuffer` construction in Rust, decoded
  frames come back as a real IOSurface-backed `CVPixelBuffer`. Not a shell-out to the
  `ffmpeg` CLI (a subprocess boundary can only hand back copied bytes; this stays
  in-process). Handles both H.264 (`avc1`) and HEVC tagged `hev1` (see scope limits).
  Verified on real Apple Silicon hardware: `crates/pyroboframes-core/src/videotoolbox_native.rs`'s
  test module generates real H.264 *and* HEVC clips via `ffmpeg`, decodes each through the
  real `VTDecompressionSession` path, confirms genuine IOSurface backing, and cross-validates
  hardware-decoded pixels against `ffmpeg`'s software decode of the same bitstream.
  **This path was dead code in every published build until v2.5.0** — `pyproject.toml`'s
  `[tool.maturin] features` never included `videotoolbox`, so every wheel actually shipped
  fell back to the `ffmpeg` CLI subprocess decoder on macOS despite this file's test
  coverage; `otool -L` on the built `.so` showed no CoreMedia/CoreVideo/VideoToolbox
  frameworks linked. Fixed by adding it to the feature list — verify yourself with
  `otool -L $(python -c "import pyroboframes,os;print(os.path.join(os.path.dirname(pyroboframes.__file__),'_core.abi3.so'))")`.
  **Known scope limits:**
  - HEVC tagged `hvc1` (a common alternative tag — some encoders default to it over `hev1`)
    isn't supported: the `mp4` crate (0.14.0, its newest published version) doesn't
    recognize `hvc1` sample entries at the track-discovery level at all, before any of
    this module's own code runs. Re-mux/re-encode with `-tag:v hev1` as a workaround.
  - No general B-frame reorder buffer: each `decode_at` call decodes samples in decode
    order from the nearest preceding keyframe through the target sample, then returns
    whichever decoded frame's presentation timestamp is closest to the target. Correct for
    the no-B-frames case and isolated single-frame lookups, not a full streaming-playback
    reorder buffer. A *related* bug existed until v2.5.0 and is now fixed: `decode_at`
    matched the caller's timestamp against raw decode-order sample position instead of
    presentation time, so on any clip where the encoder delays the whole PTS timeline (the
    normal case whenever B-frames are used — visible as the first sample's presentation
    time being nonzero) it could silently return the *wrong or duplicate* frame for nearby
    timestamps. This wasn't caught earlier because it was untested (see the dead-code note
    above) and the existing test fixtures explicitly disabled B-frames
    (`-profile:v baseline`/`-bf 0`) to sidestep the reorder-buffer limitation, which also
    hid this timestamp bug. Regression test:
    `native_decode_handles_pts_offset_from_bframe_reordering`.
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
  the `scipy`-backed resize/filter paths work against the package's current
  `numpy>=1.24` floor (no ceiling — see the Dependencies section below; the old
  `scipy<1.13`-under-`numpy==1.24` restriction described here previously was already
  stale as of the caps being removed in v2.5.0).

## 🔴 Known gaps / not done

- **True zero-copy array handoff (DLPack, skipping NumPy entirely)** — decode-to-buffer
  is zero-copy on macOS as of v2.3.0, but `Loader`'s batch path still allocates one
  combined `[batch, H, W, 3]` NumPy array and copies each decoded frame's pixels into it
  — unavoidable as long as the public contract is "one packed NumPy array per batch"
  (numpy needs contiguous memory; the source frames are independent buffers, one per
  decode). What *was* fixable: each frame used to be copied twice — decode buffer → a
  throwaway per-frame `Vec` (`Frame::to_rgb24_bytes()`) → the batch array via
  `extend_from_slice`. `Frame::write_rgb24_into()` (`crates/pyroboframes-core/src/decode.rs`)
  now writes straight from the decode buffer into the frame's slot in the batch array, so
  it's down to the one copy that's structurally required. Skipping that last copy too
  (e.g. decoding straight into the batch array's memory, or a DLPack array-of-buffers
  instead of one packed array) is a further redesign, still future work.
- **HEVC decode for `hvc1`-tagged files** in the native VideoToolbox path (see the scope
  limits under VideoToolbox above — `hev1`-tagged HEVC works).
- **Only macOS (Apple Silicon) wheels published to PyPI.** No Linux or Windows wheel has
  been published; Linux/Windows users build from the source distribution (requires a
  Rust toolchain + `ffmpeg` at build time). Linux `aarch64` and Windows haven't been
  validated at all.
- **CI is currently red on `main`: the `mcap_convert` fuzz target crashes with a
  libFuzzer out-of-memory abort on a malformed MCAP header.** This is a real,
  currently-open bug — an unbounded allocation reachable from adversarial input, not
  flaky CI infra. `Python Tests` and `Rust Build & Test` jobs pass; only `Fuzz Targets
  (build + smoke test)` fails. Not yet fixed as of this writing; check
  `gh run list --repo Mullassery/PyRoboFrames` for current status before relying on
  `convert_mcap` against untrusted MCAP files.
- **Fuzz testing covers MCAP/rosbag/Parquet/ROS2 CDR, not MP4/HDF5/NetCDF.**
  `crates/pyroboframes-core/fuzz/` (added v2.5.0) has 5 targets against
  adversarially malformed bytes: MCAP, rosbag, Parquet data-shard, Parquet
  episodes, and ROS2 CDR decode (the MCAP target's own OOM crash above is exactly
  the kind of bug this suite exists to catch — it caught one). The MP4/HDF5/NetCDF
  parsers still assume trusted input and aren't fuzzed yet. Path
  traversal is handled — `RoboFrameDataset.from_path`, `convert_mcap`,
  `convert_ros2_bag`, `HDF5Dataset`/`convert_hdf5`, and `NetCDFDataset`/`convert_netcdf`
  all accept an opt-in `base_dir` that enforces `pyroboframes.security.validate_dataset_path()`
  containment — see `SECURITY.md`.

## Fixed this release (v2.5.1)

- **The `2.5.0` package published to PyPI was completely non-functional** — a `maturin
  develop` (editable-install) artifact got uploaded instead of a real `maturin build`
  wheel; the file contained no compiled extension and no Python source, just a `.pth`
  pointing at the maintainer's local machine. Every `pip install pyroboframes` resolving
  to `2.5.0` was broken for every user. Fixed and republished as `2.5.1`; see
  `CHANGELOG.md` and `scripts/release.sh` (new — verifies a built wheel actually looks
  like a real package before it's ever uploaded).

## Fixed in v2.5.0

- **VideoToolbox hardware decode shipped as dead code in every prior release** — the
  `videotoolbox` Cargo feature was missing from `pyproject.toml`'s `[tool.maturin]`
  feature list, so every published wheel silently used the `ffmpeg` CLI fallback on macOS
  instead of the real `VTDecompressionSession` path this doc has described as "verified on
  real Apple Silicon hardware" all along (true of the code and its `cargo test` coverage,
  never true of what actually shipped). Fixed; see the VideoToolbox entry above.
- **HEVC decode added** to the native VideoToolbox path (`hev1`-tagged files; see scope
  limits above) — the `mp4` crate's `hvcC` box parser only reads `configurationVersion`
  and discards the VPS/SPS/PPS arrays, so `crates/pyroboframes-core/src/videotoolbox_native.rs::hevc_hvcc`
  is a from-scratch `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15) reader.
- **`decode_at` timestamp bug** (real-world B-frame clips could decode the wrong/duplicate
  frame) — see the VideoToolbox entry above.
- Path traversal, batch-copy perf, and numpy-cap fixes — see `CHANGELOG.md`.

## Fixed in v2.4.0

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

Not hard-pinned — floor only (see `pyproject.toml` for the authoritative list):
```
numpy>=1.24
pyarrow>=14
```
Earlier releases pinned `numpy==1.24`/`pyarrow==14` exactly, believed necessary for the
compiled extension's ABI. That was relaxed to a floor in `2804b06`; verified here against
numpy 2.4.6 + pyarrow 25.0.1 (plus modern scipy/scikit-learn/pandas) — full test suite
green (302 passed, 0 failed). `pip install pyroboframes` resolves a mutually-compatible
pair on its own; the only failure mode is manually pinning an old `pyarrow` (pre-numpy-2
ABI) against a numpy>=2 install yourself.

Optional extras (`mlx`, `dev`) and format-specific optional imports (`h5py`, `xarray`,
`netCDF4`, `tensorflow_datasets`, `fsspec`+`s3fs`/`gcsfs`, `torch`, `jax`, `scipy`,
`scikit-learn`) are not hard dependencies — install what you need.
