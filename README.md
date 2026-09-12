# PyRoboFrames

[![CI](https://github.com/Mullassery/PyRoboFrames/actions/workflows/ci.yml/badge.svg)](https://github.com/Mullassery/PyRoboFrames/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

## Problem

Loading robot-learning datasets for training is usually slower than it needs to be:
Python-side video decode, format-specific one-off loaders per dataset (HDF5 here, ROS2
bags there, RLDS somewhere else), and on Apple Silicon specifically, no path to hardware
video decode without either shelling out to `ffmpeg` per frame or writing your own
VideoToolbox bindings.

## Solution

PyRoboFrames is a Rust-backed ML dataloader for robot learning datasets. Native support
for the [LeRobot](https://github.com/huggingface/lerobot) v3.0 dataset format, with
hardware video decode (real, in-process **VideoToolbox** on Apple Silicon — not an
`ffmpeg` subprocess), conversion from HDF5/NetCDF/RLDS/MCAP/ROS2-bag into that format,
and output as NumPy, PyTorch, JAX, or MLX arrays.

The heavy lifting — dataset reading, video decode, temporal windowing — is a compiled
Rust extension (`pyroboframes._core`, built with PyO3/maturin). The Python package on top
of it is the ergonomic surface: `RoboFrameDataset`, `DataLoader`, format converters, and
device adapters. If you're evaluating this against a bigger project like Hugging Face
`datasets` or `torchcodec`: this is smaller in scope, focused specifically on robot
learning's LeRobot-style episodic data (state/action/video, aligned by frame index), and
its differentiating feature is genuine zero-copy hardware video decode on Apple Silicon.

```bash
pip install pyroboframes
```

> **Platform note:** prebuilt wheels are currently published for **macOS (Apple
> Silicon) only**. Linux/Windows users install from the source distribution, which
> needs a Rust toolchain at build time — see [Installation](#installation) below.

## Use cases

- **Training-loop data loading** — iterate a LeRobot v3.0 dataset in batches
  (state/action tensors + decoded camera frames) directly into NumPy/PyTorch/JAX/MLX.
  This is the primary, best-tested path. See [Quick start](#quick-start) and
  [`examples/humanoid_multimodal_fusion.py`](examples/humanoid_multimodal_fusion.py) /
  [`examples/robotdog_proprioceptive_learning.py`](examples/robotdog_proprioceptive_learning.py)
  for full working scripts.
- **Converting existing recordings into LeRobot format** — HDF5 (ROBOMIMIC/ACT-style),
  NetCDF, RLDS (Open X-Embodiment), or raw MCAP/ROS2 bag recordings, via
  `convert_hdf5()` / `convert_netcdf()` / `RLDSDataset.from_tfds()` / `convert_mcap()` /
  `convert_ros2_bag()`, so a downstream pipeline only has to know one format.
- **Apple Silicon training runs where camera decode is the bottleneck** — the
  in-process VideoToolbox path avoids the subprocess-copy cost of shelling out to
  `ffmpeg` per frame (see [Hardware video decode](#hardware-video-decode)).
- **Not yet a good fit for:** Linux/Windows users who need a prebuilt wheel (source
  build only, requires a Rust toolchain — see [Installation](#installation)); anything
  needing true zero-copy remote dataset streaming (`RemoteDataset` downloads to a local
  cache first, it doesn't stream); or footage that's `hvc1`-tagged HEVC or relies on a
  B-frame reorder buffer (see the honest limitation in
  [Hardware video decode](#hardware-video-decode)).

## Quick start

```python
import pyroboframes as prf

# Open a local LeRobot v3.0 dataset (the directory holding meta/, data/, videos/)
ds = prf.RoboFrameDataset.from_path("/path/to/lerobot_dataset")
print(ds.num_frames, ds.num_episodes, ds.fps, ds.cameras)

# Or pull one from the Hugging Face Hub first
local_path = prf.download_lerobot_dataset("lerobot/aloha_mobile_cabinet")
ds = prf.RoboFrameDataset.from_path(local_path)

# Batched iteration — state/action tensors plus decoded camera frames
loader = ds.loader(
    batch_size=32,
    shuffle=True,
    cameras=["observation.images.top"],  # decodes video on the fly
    output="numpy",                      # or "torch" / "mlx" / "jax"
)
for batch in loader:
    batch["observation.state"]            # [32, state_dim] float32
    batch["action"]                       # [32, action_dim] float32
    batch["observation.images.top"]       # [32, H, W, 3] uint8
```

See [`.github/INSTALL.md`](.github/INSTALL.md) for platform-specific install notes,
[`examples/`](examples/) for full training-loop scripts (humanoid multimodal fusion,
proprioceptive-only quadruped loading), and [`docs/`](docs/) for deeper architecture notes.

## Hardware video decode

Video decode is the part of this project most worth being skeptical of, so here's
what's actually true as of this release:

- **macOS (Apple Silicon), `videotoolbox` build feature:** a real, in-process
  `VTDecompressionSession` — MP4 demuxing and `CMSampleBuffer` construction happen in
  Rust, frames come back as an IOSurface-backed `CVPixelBuffer`, and nothing shells out
  to the `ffmpeg` CLI. This is what makes zero-copy handoff to Apple's ML frameworks
  possible: a subprocess can only hand back decoded *bytes* (a copy by construction); an
  in-process `VTDecompressionSession` hands back a live buffer reference. See
  `crates/pyroboframes-core/src/videotoolbox_native.rs` for the implementation, and its
  test module for hardware-decode tests that cross-validate real decoded pixels against
  `ffmpeg`'s software decode of the same bitstream.
- **Cross-platform fallback, `ffmpeg` build feature:** shells out to the `ffmpeg` CLI
  (with `-hwaccel videotoolbox`/`vaapi` where available). This is what ships in the
  default build config and works everywhere `ffmpeg` is installed, at the cost of a copy
  through the subprocess pipe.
- **Linux + NVIDIA, `cuda` build feature:** NVDEC via `ffmpeg -hwaccel cuda`. Also
  downloads decoded frames to host memory today (not yet a zero-copy CUDA buffer handoff).

The published macOS wheel is built with `--features videotoolbox`; the source
distribution defaults to the portable `ffmpeg` feature so it builds on any platform.

**History:** every wheel published before v2.5.0 silently used the `ffmpeg`-subprocess
fallback on macOS instead of the VideoToolbox path above, even though that path was real
and covered by `cargo test --features videotoolbox` the whole time — the `videotoolbox`
Cargo feature was simply missing from `pyproject.toml`'s `[tool.maturin] features` list,
so it never got linked into a published wheel. Fixed in v2.5.0 (verify yourself with
`otool -L` on your installed `_core*.so`: it should link CoreMedia/CoreVideo/VideoToolbox).

**Honest limitation:** the native VideoToolbox path decodes H.264 and HEVC tagged `hev1`
(not `hvc1` — see `ROADMAP_HONEST.md`), and doesn't implement a full B-frame reorder
buffer — correct for the common no-B-frames case and for isolated single-frame lookups,
not yet a general streaming-playback decoder. Also, `Loader`'s batch path still allocates
one combined `[batch, H, W, 3]` NumPy array and copies each decoded frame's pixels into it
(one copy per frame, down from two as of v2.5.0) — decode-to-CPU-buffer is zero-copy, but
building a single packed batch array from independent per-frame buffers isn't free; a true
zero-copy `mx.array`/DLPack handoff that skips NumPy entirely is still future work.

## Dataset formats

| Format | Status | Notes |
|---|---|---|
| **LeRobot v3.0** | Native, primary | Direct Rust reader; everything else converts *to* this layout. |
| **HDF5** (ROBOMIMIC/ACT-style) | Real, via `h5py` (optional dep) | `HDF5Dataset.from_path()`, `convert_hdf5()`. |
| **NetCDF** | Real, via `xarray`+`netCDF4` (optional deps) | `NetCDFDataset.from_path()`, `convert_netcdf()`. |
| **RLDS** (Open X-Embodiment) | Real, via `tensorflow_datasets` (optional dep) | `RLDSDataset.from_tfds()` / `.from_directory()`. |
| **MCAP / ROS2 bag** | Real, native Rust | `convert_mcap()`, `convert_ros2_bag()` → Parquet. See [What's not working](#whats-not-working--open-issues) — `convert_mcap` currently has an open fuzz-found crash on malformed input. |
| **Cloud object storage** | Real, via `fsspec`+`s3fs`/`gcsfs` (optional deps) | `RemoteDataset.from_s3()`/`from_gcs()` — downloads to a local cache and reads from there; this is *not* a true zero-copy remote stream. |

Each optional-dependency reader raises a clear `ImportError` with an install hint if the
dependency is missing, rather than silently producing empty output. `pyroboframes/_format_registry.py`
adds a unified `load_dataset(path, format=...)` entry point across the above.

## Installation

```bash
pip install pyroboframes
```

This installs a prebuilt wheel on **macOS arm64**. On other platforms `pip` falls back
to the source distribution, which needs a Rust toolchain and (for the default `ffmpeg`
build feature) `ffmpeg`/`ffprobe` on `PATH` at build time:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install pyroboframes
```

Optional extras, installed separately depending on which formats/backends you use:
`h5py` (HDF5), `xarray netCDF4` (NetCDF), `tensorflow_datasets` (RLDS), `fsspec s3fs
gcsfs` (cloud object storage), `mlx` (Apple Silicon array output — also `pip install pyroboframes[mlx]`),
`torch`/`jax` (other array backends), `scipy scikit-learn` (GPU-acceleration transforms
and 3D occupancy-grid morphology — no version ceiling as of v2.5.0; verified working
against numpy 2.4.6 + scipy 1.17 + scikit-learn 1.9).

See [`.github/INSTALL.md`](.github/INSTALL.md) for troubleshooting.

## Development

```bash
git clone https://github.com/Mullassery/PyRoboFrames
cd PyRoboFrames
pip install -e ".[dev]"
python -m maturin develop --release   # or: --release --features videotoolbox (macOS)
pytest tests/ -v
cargo test --workspace
cargo clippy --all-targets -- -D warnings
```

To cut a release, use `scripts/release.sh` rather than `maturin build` directly — it
exists because v2.5.0 was accidentally published as a broken editable-install artifact
(see [What's not working](#whats-not-working--open-issues)), and refuses to proceed if
the built wheel doesn't look like a real, importable package.

## What's working now (verified)

323 Python tests / 324 Rust unit+integration tests as of this revision (`grep -c "def
test_"` / `grep -c "#\[test\]"` — treat the CI badge above as more authoritative than
any number in this file, since these drift). See [`ROADMAP_HONEST.md`](ROADMAP_HONEST.md)
for the full solid-vs-rough breakdown and [`SECURITY.md`](SECURITY.md) for the current
security/compliance posture.

**CI history, because the badge alone doesn't tell you this:** before the pass that
produced v2.5.1, CI had been red on every run for over a week straight, for six
independent, real reasons — meaning neither the Rust nor the Python test suite was
actually being exercised on any recent commit, despite the badge being visible in this
README the whole time:

1. `--all-features` unconditionally enabled `pyroboframes-py`'s `extension-module`
   feature, which only links correctly under maturin's build — a plain `cargo build`
   fails to link regardless of platform. Fixed by testing with `--features ffmpeg`
   instead on the Linux CI runner.
2. `apple-cf`/`videotoolbox` (macOS-only system-framework bindings) were plain
   `[dependencies]` rather than scoped to `[target.'cfg(target_os = "macos")'.dependencies]`,
   so enabling `videotoolbox` on Linux tried to compile them and failed. Fixed.
3. The Python job ran `cd python && pip install -e ".[dev]"`, but `pyproject.toml` lives
   at the repo root — this failed outright on every run. Fixed to install from the root.
4. The next step checked `if [ -d "python/tests" ]` before running pytest, but the real
   suite lives in `./tests` at the repo root, so this silently printed "No Python tests
   found" instead of running anything. Fixed.
5. `numpy==1.24` (pinned in `dev` extras) never published a `cp312` wheel, so the `3.12`
   leg of the test matrix fell back to a source build that failed outright. Relaxed to a
   floor with no ceiling (see Installation above).
6. With (1)–(2) fixed, a real test that shells out to the `ffmpeg` binary started
   running, but `ffmpeg` isn't installed on the `ubuntu-latest` runner by default. Added
   an explicit `apt-get install -y ffmpeg` step.

Fixing all six surfaced a real gap (`huggingface_hub` missing from `dev` extras, added)
and, separately, ~19 real assertion failures once 5 previously-non-compiling Rust
integration-test files were fixed — including a priority-ordering bug that silently
broke `FeedbackLoop::get_learning_report`'s retraining list, a model-id parsing bug that
truncated any id containing an underscore, and an anomaly detector that stopped tracking
temporal jitter after its first detected anomaly. All fixed as of v2.5.1.

**Two incidents worth knowing about**, both already fixed but worth disclosing rather
than burying in the changelog:

- **v2.5.0's published PyPI wheel was completely broken** for every user — it contained
  no compiled extension and no Python source, just a stray `.pth` file pointing at the
  maintainer's local machine (the artifact of `maturin develop`, uploaded as if it were
  a real `maturin build` wheel). `import pyroboframes` failed on any machine other than
  the one it was built on. Fixed and republished as v2.5.1; `scripts/release.sh` (see
  Development above) now makes this class of mistake structurally hard to repeat.
- **VideoToolbox hardware decode — this README's headline feature — was dead code in
  every wheel ever published before v2.5.0**, for the reason described in
  [Hardware video decode](#hardware-video-decode) above. If you installed any version
  before 2.5.0, you were silently getting the `ffmpeg`-subprocess fallback regardless of
  what this README said.

## What's not working / open issues

- **`Fuzz Targets (build + smoke test)` is currently failing on `main`, for `mcap_convert`
  specifically.** A real, currently-open bug in `mcap` 0.25.0 itself (the latest release —
  no newer version fixes it): parsing a malformed record length can overflow an internal
  `usize` addition and panic, inside `mcap`'s own reader, not our code. `mcap::convert()`
  now isolates this behind `catch_unwind`, so **real callers get a clean `Err` instead of
  a crash** (verified directly against the exact crash input). This does *not* make the
  CI smoke test itself pass, though: `cargo-fuzz`'s harness deliberately aborts the
  process on any panic before `catch_unwind` gets a chance to run, specifically so fuzzing
  can find and report bugs like this one — that's the fuzzer working as intended, not a
  regression. `mcap`'s public API has no way to bound record-length parsing to avoid the
  panic being reachable at all (that knob exists only on an internal type this crate
  doesn't expose). Tracked as an upstream issue to file against
  [foxglove/mcap](https://github.com/foxglove/mcap); until fixed there, this is the
  practical ceiling for hardening this path.
- **Only macOS (Apple Silicon) wheels are published to PyPI.** Linux/Windows users must
  build from source (Rust toolchain + `ffmpeg` at build time); Linux `aarch64` and
  Windows haven't been validated at all.
- **HEVC decode only covers `hev1`-tagged files, not `hvc1`**, and there's no B-frame
  reorder buffer — see the honest limitation under
  [Hardware video decode](#hardware-video-decode).
- **`RemoteDataset`'s cloud-storage readers are not true zero-copy streaming** — they
  download to a local cache first.
- **Fuzz testing covers MCAP/rosbag/Parquet/ROS2 CDR, not MP4/HDF5/NetCDF.** Those
  parsers still assume trusted input.
- **CUDA decode (`cuda` build feature) downloads frames to host memory** — not yet a
  zero-copy CUDA buffer handoff.

## Known Issues

- **2026-09-13: three fuzz-found crashes investigated, two fully fixed, one mitigated
  (all three in third-party dependencies, not our code).**
  1. `mcap_convert` was hitting a real libFuzzer out-of-memory abort on malformed MCAP
     headers — an unbounded allocation inside the `mcap` crate's own record parser
     (v0.9). Fixed by upgrading to `mcap` 0.25 (latest): the exact saved crash input no
     longer reproduces, and a 315k-iteration local fuzz run found zero crashes. `mcap`
     0.25's `Channel`/`Schema` structs gained a required `id: u16` field; the 7
     test-fixture literals constructing these directly were updated accordingly.
  2. `data_shard_parquet` was separately panicking inside the `parquet` crate's
     Thrift-compact-protocol metadata decoder on a truncated Parquet footer (v55). Fixed
     by upgrading `arrow`/`parquet` 55→59 (latest): the saved crash input no longer
     reproduces.
  3. After both upgrades, CI's fuzz smoke test found a **third, different** bug in
     `mcap_convert`: a `usize` overflow panic inside `mcap` 0.25.0's own reader on a
     different malformed length field — a genuine bug in the latest release, with no
     newer version available and no public API to bound record-length parsing to avoid
     it. Mitigated (not fixed) by wrapping the read path in `catch_unwind`, so real
     callers get a clean `Err` instead of a process abort (verified against the exact
     crash input) — see [What's not working](#whats-not-working--open-issues) for why
     this specific CI job will keep showing red until the bug is fixed upstream.
  All fixes verified against the full 318-test `cargo test --release --features ffmpeg`
  suite (all passing) in addition to direct fuzz-input replay.
- No open GitHub issues and no real `TODO`/`FIXME`/`XXX` markers in `crates/` or
  `python/` as of this pass (one `XXX` match is a filename placeholder in a doc
  comment, not an actual TODO).
- `python/pyroboframes/vision.py` (`CLIPEmbedder`/`SAM2Segmenter`/`GroundingDINO`) and
  `docs/ROADMAP_V0.5.3_SAM_MODELS.md` were removed 2026-08-24: the module was dead code
  (never imported by `__init__.py`), untested, and `GroundingDINO.detect()` returned
  empty results for every frame instead of calling a model. Foundation-model /
  auto-annotation work for this ecosystem lives in **PyRoboVision** instead — see
  [Cross-repo compatibility](#cross-repo-compatibility) below.
- Package version (`2.5.1`, dynamic from `Cargo.toml`) matches the version currently
  published on PyPI — no drift as of this pass.

## Cross-repo compatibility

This repo is one of several independently-published robotics packages by
the same author (`PyRoboSimulator`, `pyroboreplay`, `PyRoboVision`,
`PyTerrainMap`). Verified by reading every `Cargo.toml`/`pyproject.toml` in
that group: this repo has no Cargo or pip dependency on any of them.
PyRoboVision's README documents an intentionally loose, install-separately
relationship with this repo (feeding PyRoboFrames-loaded data into its
tracker) — this repo has no corresponding code and makes no claim of its
own about that relationship.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

Questions or bug reports: [GitHub Issues](https://github.com/Mullassery/PyRoboFrames/issues).
