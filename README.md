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

~248 Python tests / ~76 Rust unit tests as of this revision (counted via `grep -c "def
test_"` / `grep -c "#\[test\]"`, not a full `pytest`/`cargo test` run — see the CI badge
above for the authoritative, currently-passing count). See
[`ROADMAP_HONEST.md`](ROADMAP_HONEST.md) for an unvarnished list of what's solid vs. what's
still rough, and [`SECURITY.md`](SECURITY.md) for the current security/compliance posture.

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
- This working tree had uncommitted local changes (a v2.0.0-era "MCP 2.0"
  connector module, an OTel/observability setup guide, and a rewritten
  ROADMAP.md reintroducing emoji/aspirational-checklist content) that predate
  and conflict with this repository's own documented cleanup pass (see the
  `git log` entry "Fix live property/method API bugs, remove dead fake code,
  rewrite docs for accuracy (v2.4.0)"). Those changes were intentionally left
  uncommitted rather than merged in — see repo owner's own working tree for
  disposition.

## License

Proprietary — free to use with explicit attribution. See [`LICENSE`](LICENSE).

---

Questions or bug reports: [GitHub Issues](https://github.com/Mullassery/PyRoboFrames/issues).
