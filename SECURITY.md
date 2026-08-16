# Security Policy

## Reporting Security Issues

**DO NOT** open public GitHub issues for security vulnerabilities.

If you discover a security vulnerability, please email: **mullassery@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact
- Suggested fix (if you have one)

## Current Status (v2.4.0)

PyRoboFrames is a small, single-maintainer project. It is **not** independently
security-audited and carries **no compliance certifications** (see below) — evaluate it
yourself before using it to handle sensitive data, and don't treat this document as a
substitute for your own review. That said, "beta, no production use" no longer reflects
reality: the core dataset-loading and video-decode path is real, tested Rust/PyO3 code
(223 Python tests + 75 Rust unit tests passing, 0 known failures as of this release), the
headline VideoToolbox hardware-decode path is a genuine in-process `VTDecompressionSession`
integration (not a subprocess shell-out), and it has been used to load real LeRobot
datasets end-to-end. Treat it as: solid for the code paths you've tested against your own
data, not yet hardened against adversarial input.

### What's implemented and tested
- LeRobot v3.0 dataset reading (the native, most-exercised format) — Rust core, `pytest`
  + `cargo test` covered.
- HDF5, NetCDF, and RLDS → LeRobot conversion — real readers via `h5py` / `xarray` /
  `tensorflow_datasets` (each is an optional dependency; the reader raises a clear
  `ImportError` with an install hint if missing, rather than silently no-op'ing).
- MCAP / ROS2 bag → Parquet conversion (Rust core).
- VideoToolbox (macOS), FFmpeg (cross-platform), and NVDEC (Linux/CUDA) video decode
  backends.
- S3/GCS dataset access via `fsspec`/`s3fs`/`gcsfs` — downloads to a local cache directory
  and reads from there; **not** a true zero-copy remote stream (see `ROADMAP_HONEST.md`).

### Known gaps (be aware of these)
- **Input trust boundary:** dataset files (Parquet, MP4, HDF5, NetCDF, MCAP) are assumed
  to come from a source you trust. None of the format parsers have been fuzz-tested
  against adversarially malformed input. Don't point this at untrusted user uploads
  without your own sandboxing.
- **Path handling:** `pyroboframes.security.validate_dataset_path()` rejects `..`
  components and can restrict a path to a base directory, but it is opt-in — most loader
  entry points (`RoboFrameDataset.from_path`, `convert_hdf5`, etc.) do **not** call it
  automatically. Call it yourself if the path comes from an untrusted source.
- **S3/GCS credentials:** `RemoteDataset.from_s3()` / `.from_gcs()` pass through to
  `fsspec`'s credential resolution (AWS profile / IAM role / GCS token). Prefer IAM
  roles over long-lived keys — see `DEPLOYMENT_SECURITY.md`.
- **Dependency pins:** `numpy==1.24` and `pyarrow==14` are hard-pinned (required by the
  compiled extension's ABI and by the LeRobot Parquet path). This is good for supply-chain
  reproducibility but means recent versions of optional companion libraries (`scipy>=1.13`,
  `scikit-learn>=1.5`, `pandas>=2.2`) will fail to import — pin them below those
  thresholds if you install the `dev` extra or use the optional GPU-acceleration /
  occupancy-grid modules. Tracked as a real gap, not silently papered over.
- **Only macOS (Apple Silicon) wheels are currently published to PyPI.** Linux/Windows
  users install from the source distribution, which requires a Rust toolchain and
  `ffmpeg` at build time (see `.github/INSTALL.md`).

## Security Updates

Security patches will be released as minor/patch versions when vulnerabilities are
discovered. There is no formal SLA — this is a single-maintainer project.

## Dependency Security

This project uses:
- Python 3.10+
- Rust 1.78+ (for Rust components; `rust-toolchain.toml` pins the exact version CI builds against)

`numpy` and `pyarrow` are pinned to exact versions (see `pyproject.toml`) rather than
left floating, specifically to avoid unreviewed transitive upgrades landing silently.
Bump them deliberately, not via `pip install -U`.

## Compliance & Certifications

This software is **NOT**:
- SOC 2 certified
- HIPAA compliant
- GDPR compliant
- PCI DSS compliant
- ISO 27001 certified

If your use case requires any of the above, this package alone does not satisfy it —
build your own controls around it.

## Development Security

When contributing:
- Do not commit secrets, credentials, or API keys
- Use environment variables (or IAM roles / GCS tokens) for sensitive configuration —
  never hardcode credentials, see `DEPLOYMENT_SECURITY.md`
- Run `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` before submitting
- Write tests for security-related code (path validation, credential handling)

## Questions?

For security questions (non-vulnerability): open a GitHub Discussion.
