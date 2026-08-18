# PyRoboFrames

[![CI](https://github.com/Mullassery/PyRoboFrames/actions/workflows/ci.yml/badge.svg)](https://github.com/Mullassery/PyRoboFrames/actions/workflows/ci.yml)

A Rust-backed ML dataloader for robot learning datasets. Native support for the
[LeRobot](https://github.com/huggingface/lerobot) v3.0 dataset format, with hardware
video decode (real, in-process **VideoToolbox** on Apple Silicon — not an `ffmpeg`
subprocess), conversion from HDF5/NetCDF/RLDS/MCAP/ROS2-bag, and output as NumPy, PyTorch,
JAX, or MLX arrays.

```bash
pip install pyroboframes
```

> **Platform note:** prebuilt wheels are currently published for **macOS (Apple
> Silicon) only**. Linux/Windows users install from the source distribution, which
> needs a Rust toolchain at build time — see [Installation](#installation) below.

## What this actually is

The heavy lifting — dataset reading, video decode, temporal windowing — is a compiled
Rust extension (`pyroboframes._core`, built with PyO3/maturin). The Python package on top
of it is the ergonomic surface: `RoboFrameDataset`, `DataLoader`, format converters, and
device adapters. If you're evaluating this against a bigger project like Hugging Face
`datasets` or `torchcodec`: this is smaller in scope, focused specifically on robot
learning's LeRobot-style episodic data (state/action/video, aligned by frame index), and
its differentiating feature is genuine zero-copy hardware video decode on Apple Silicon.

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

**Honest limitation:** the native VideoToolbox path decodes H.264 only (no HEVC yet),
and doesn't implement a full B-frame reorder buffer — correct for the common no-B-frames
case and for isolated single-frame lookups, not yet a general streaming-playback decoder.
Also, `Loader`'s batch path still copies frame bytes into one combined `[batch, H, W, 3]`
NumPy array — decode-to-CPU-buffer is zero-copy, but building a single batched array from
independent per-frame buffers isn't free; a true zero-copy `mx.array`/DLPack handoff that
skips NumPy entirely is still future work.

## Dataset formats

| Format | Status | Notes |
|---|---|---|
| **LeRobot v3.0** | Native, primary | Direct Rust reader; everything else converts *to* this layout. |
| **HDF5** (ROBOMIMIC/ACT-style) | Real, via `h5py` (optional dep) | `HDF5Dataset.from_path()`, `convert_hdf5()`. |
| **NetCDF** | Real, via `xarray`+`netCDF4` (optional deps) | `NetCDFDataset.from_path()`, `convert_netcdf()`. |
| **RLDS** (Open X-Embodiment) | Real, via `tensorflow_datasets` (optional dep) | `RLDSDataset.from_tfds()` / `.from_directory()`. |
| **MCAP / ROS2 bag** | Real, native Rust | `convert_mcap()`, `convert_ros2_bag()` → Parquet. |
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
and 3D occupancy-grid morphology — pin below `scipy<1.13`/`scikit-learn<1.5` to stay
compatible with this package's `numpy==1.24` pin).

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

## Status

~308 Python tests / ~318 Rust unit+integration tests as of this revision (counted via
`grep -c "def test_"` / `grep -c "#\[test\]"`). See [`ROADMAP_HONEST.md`](ROADMAP_HONEST.md)
for an unvarnished list of what's solid vs. what's still rough, and
[`SECURITY.md`](SECURITY.md) for the current security/compliance posture.

**Don't trust the CI badge above without reading this first.** Before this pass, CI had
been red on every run for over a week straight, for six independent, real reasons -
meaning neither the Rust nor the Python test suite was actually being exercised on any
recent commit, despite the badge being visible in this README the whole time:

1. `--all-features` unconditionally enables `pyroboframes-py`'s `extension-module`
   feature, which that crate's own `Cargo.toml` documents as only safe to combine with
   maturin's build (it needs maturin's dynamic-lookup linker flags) - a plain
   `cargo build`/`cargo test --all-features` fails to link regardless of platform. Fixed
   by building/testing with `--features ffmpeg` instead (the feature actually relevant to
   an ubuntu-latest runner; `videotoolbox` is macOS-only and `cuda` needs a CUDA toolkit
   the runner doesn't have).
2. `apple-cf`/`videotoolbox` (real macOS-only system-framework bindings, needed for the
   VideoToolbox hardware decode path) were plain `[dependencies]` rather than scoped to
   `[target.'cfg(target_os = "macos")'.dependencies]`, so even setting (1) aside, enabling
   the `videotoolbox` feature on Linux tried to compile them and failed. Fixed.
3. The Python job ran `cd python && pip install -e ".[dev]"`, but `pyproject.toml` lives at
   the repo root, not in `python/` - this failed outright on every run, so the Python suite
   never got a chance to run at all. Fixed to install from the root.
4. Even if (3) hadn't failed first, the next step checked `if [ -d "python/tests" ]` before
   running pytest - but the real suite lives in `./tests` at the repo root, so this check
   was always false and silently printed "No Python tests found" instead of running
   anything. Fixed to check/run `tests/` at the root.
5. `numpy==1.24` (pinned in the `dev` extras) never published a `cp312` wheel - it predates
   Python 3.12 - so on the `3.12` leg of the Python test matrix, `pip install -e ".[dev]"`
   fell back to a source build that failed outright (no working
   `setuptools.build_meta`). Relaxed to `numpy>=1.24,<1.27`, which resolves to 1.24.x on
   3.10/3.11 and 1.26.4 (the first release with 3.12 wheels) on 3.12 - verified all of
   numpy/pandas/scipy/scikit-learn still import together cleanly at that combination.
6. With (1)-(2) fixed, `--features ffmpeg` started actually building and running
   `decode::tests::ffmpeg_decoder_decodes_a_real_frame`, a real test that shells out to the
   `ffmpeg` binary - which isn't installed on the `ubuntu-latest` runner by default. Added
   an explicit `apt-get install -y ffmpeg` step.

With all six fixed, running the real suite for the first time surfaced one more real gap:
`tests/test_storage.py` exercises `hub.py`'s optional `huggingface_hub`-based LeRobot-hub
download path, but `huggingface_hub` wasn't listed in the `dev` extras, so a clean
`pip install -e ".[dev]"` couldn't actually run that test. Added it to `dev`. Full result
after all of the above: 273 passed, 10 skipped (environment-gated, e.g. missing
ffprobe/OpenCV), 0 failed.

Separately, this pass also ran the full Rust suite (`cargo test --workspace`) locally for
the first time in a while: 5 of the `crates/pyroboframes-core/tests/*.rs` integration-test
files didn't even compile (private-field access from an external test crate, a
borrow-of-moved-value, and a borrow-checker conflict). Fixing the compile errors surfaced
~19 real assertion failures underneath, roughly split between miscalibrated test fixtures
(values that didn't actually cross the thresholds they were meant to trigger) and genuine
production bugs, the more notable of which: `TriggerPriority`/`DecisionPriority`/`CachePriority`
all derived `Ord` with `Critical` declared first, so a natural `priority >=
SomeVariant::High` comparison silently ranked `Critical` *below* `High`/`Medium`/`Low` -
this broke `FeedbackLoop::get_learning_report`'s `models_to_retrain` field for real (not
just in tests); `EnsembleOrchestrator::get_best_model` parsed a model's id back out of a
`"{id}_{type}"` key via `.split('_').next()`, which silently truncated any model id
containing an underscore; `AnomalyDetector` only recorded a frame's timestamp when no
anomaly was found, so a single detected anomaly permanently broke temporal-jitter tracking
for every later frame; and `DistributedCoordinator::elect_leader`'s
`availability / (latency_ms + 1)` scoring let tiny latency differences dominate over large
availability differences. All of the above (Rust suite and CI itself) are fixed as of this
pass; the CI badge should reflect that starting with the next run on `main`.

## Known Issues

- No open GitHub issues and no real `TODO`/`FIXME`/`XXX` markers in `crates/` or
  `python/` as of this pass (one `XXX` match is a filename placeholder in a doc
  comment, not an actual TODO).
- The Rust/Python test counts in the Status section above had drifted from the
  actual source (previously stated as `223`/`75`); corrected here based on a
  `grep` count. Treat the CI badge as authoritative over any number in prose.
- Package version (`2.4.0`, dynamic from `Cargo.toml`) matches the version
  currently published on PyPI — no drift as of this pass.
- The native VideoToolbox decode path is H.264-only (no HEVC) and doesn't
  implement a full B-frame reorder buffer — see "Hardware video decode" above
  for the exact scope. `RemoteDataset`'s cloud-storage readers download to a
  local cache rather than true zero-copy streaming.

## License

Proprietary — free to use with explicit attribution. See [`LICENSE`](LICENSE).

---

Questions or bug reports: [GitHub Issues](https://github.com/Mullassery/PyRoboFrames/issues).
