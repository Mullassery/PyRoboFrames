# Installation Guide

## Quick Install

```bash
pip install pyroboframes
```

## Requirements

- Python 3.10+
- **PyPI wheel:** macOS (Apple Silicon / arm64) only, currently.
- **Linux x86_64/aarch64 and Windows x86_64:** CI builds and smoke-tests wheels as
  artifacts, but they are not yet published to PyPI. Until then, install from the
  source distribution — needs a Rust toolchain and, for the default `ffmpeg`
  video-decode build feature, `ffmpeg`/`ffprobe` on `PATH` at build time.

## Installation

### macOS (Apple Silicon) — prebuilt wheel
```bash
pip install pyroboframes
```
`pip` picks up the prebuilt wheel automatically; no Rust toolchain needed. This wheel is
built with the `videotoolbox` feature — real in-process hardware video decode.

### Linux / Windows / Intel macOS — build from source
```bash
# Rust toolchain (needed to compile the extension)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# ffmpeg (needed at build time for the default `ffmpeg` decode feature, and at
# runtime for actual video decode)
#   Debian/Ubuntu: sudo apt-get install ffmpeg
#   Fedora:        sudo dnf install ffmpeg
#   Windows:       https://ffmpeg.org/download.html

pip install pyroboframes
```
`pip` falls back to the source distribution automatically when no matching wheel exists
for your platform, and builds it using the toolchain above.

## Troubleshooting

### "No matching distribution found" / build fails from source
Confirm the Rust toolchain and `ffmpeg` are both on `PATH`:
```bash
rustc --version
ffmpeg -version
```
Then retry:
```bash
pip install --force-reinstall --no-cache-dir pyroboframes
```

### Python version issues
Ensure Python 3.10+ (the compiled extension targets the stable `abi3` ABI from 3.10):
```bash
python --version
```

### Missing dependencies (Linux)
```bash
sudo apt-get install python3-dev build-essential
```

### Optional format support
Installing `pyroboframes` alone gets you LeRobot dataset loading and MCAP/ROS2-bag
conversion. Other formats need their own optional dependency:
```bash
pip install h5py                          # HDF5 conversion
pip install xarray netCDF4                 # NetCDF conversion
pip install tensorflow_datasets            # RLDS conversion
pip install fsspec s3fs gcsfs              # S3 / GCS remote datasets
pip install pyroboframes[mlx]              # MLX array output (Apple Silicon)
```

## Next Steps

After installation:
1. See [README.md](../README.md) for a real quick-start example
2. Check [examples/](../examples/) for full training-loop scripts
3. Read [../docs/internal/ROADMAP_HONEST.md](../docs/internal/ROADMAP_HONEST.md) for what's solid vs. still rough
4. Read [docs/](../docs/) for deeper architecture notes
