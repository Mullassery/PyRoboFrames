# Security Policy

## Reporting Security Issues

**DO NOT** open public GitHub issues for security vulnerabilities.

If you discover a security vulnerability, please email: **mullassery@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact
- Suggested fix (if you have one)

## Current Status (v2.5.1)

PyRoboFrames is a small, single-maintainer project. It is **not** independently
security-audited and carries **no compliance certifications** (see below) — evaluate it
yourself before using it to handle sensitive data, and don't treat this document as a
substitute for your own review. That said, "beta, no production use" no longer reflects
reality: the core dataset-loading and video-decode path is real, tested Rust/PyO3 code
(323 Python tests + 324 Rust unit/integration tests as of this revision — treat the CI
badge in `README.md` as more authoritative than any number in this file, since these
drift), the headline VideoToolbox hardware-decode path is a genuine in-process
`VTDecompressionSession` integration (not a subprocess shell-out), and it has been used to
load real LeRobot datasets end-to-end. Treat it as: solid for the code paths you've tested
against your own data, partially (not fully) hardened against adversarial input — see the
fuzz-testing gap below.

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
- **Input trust boundary — partial fuzz coverage, not full.** `crates/pyroboframes-core/fuzz/`
  (5 targets: MCAP, rosbag, Parquet data-shard, Parquet episodes, ROS2 CDR decode) has
  actually found and driven fixes for real crashes — see `ROADMAP_HONEST.md`'s "Fixed in
  v2.5.0"/"Technical Debt" sections, including one (an upstream `mcap` 0.25.0 `usize`
  overflow panic) that's mitigated via `catch_unwind` in `mcap::convert()` but not fixed
  upstream. **MP4, HDF5, and NetCDF parsing are not fuzzed at all** and still assume
  trusted input. Don't point any of this at untrusted user uploads without your own
  sandboxing, and treat the MP4/HDF5/NetCDF paths as the least-hardened of the group.
- **Path handling:** `pyroboframes.security.validate_dataset_path()` rejects `..`
  components and can restrict a path to a base directory. `RoboFrameDataset.from_path`,
  `convert_mcap`, `convert_ros2_bag`, `HDF5Dataset`/`HDF5Dataset.from_path`/`convert_hdf5`,
  and `NetCDFDataset`/`NetCDFDataset.from_path`/`convert_netcdf` all now accept a
  `base_dir` keyword that enforces this containment check on their behalf — pass it
  whenever the path comes from an untrusted source. It remains opt-in (omitting
  `base_dir` preserves today's unrestricted behavior) so existing callers aren't broken
  by paths that legitimately live outside any sandbox.
- **Cloud storage credentials:** `RemoteDataset.from_s3()` / `.from_gcs()` pass through
  to `fsspec`'s credential resolution (cloud provider profile / IAM role / storage
  token). Prefer short-lived IAM-style roles over long-lived keys — see
  `DEPLOYMENT_SECURITY.md`.
- **Dependency floors, not hard pins:** `numpy>=1.24` and `pyarrow>=14` (no upper bound)
  — earlier releases hard-pinned `numpy==1.24`/`pyarrow==14`; relaxed to a floor in
  `2804b06` and verified here against numpy 2.4.6 + pyarrow 25.0.1 + modern
  scipy/scikit-learn/pandas (full test suite green). `pip install`'s resolver picks a
  mutually-compatible pair on its own; the only failure mode is manually pinning an old
  `pyarrow` (pre-numpy-2 ABI) against a `numpy>=2` install yourself.
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

`numpy` and `pyarrow` declare a floor (`numpy>=1.24`, `pyarrow>=14`), not an exact pin —
see "Known gaps" above for why the earlier exact pins were relaxed. There is currently no
`cargo audit` (or equivalent) job in CI checking Rust dependencies for known
vulnerabilities — this is a real, unverified gap (this project's sandbox environments
have no network access to crates.io's advisory database to run one), tracked in
`ROADMAP_HONEST.md`'s Technical Debt section rather than claimed as done here.

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
