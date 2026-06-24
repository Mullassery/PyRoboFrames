# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding: Rust workspace (`pyroboframes-core`, `pyroboframes-py`), maturin/PyO3
  build, Python package skeleton, CI (Apple Silicon), and Trusted-Publishing release workflow.
- `ARCHITECTURE.md` documenting the design, the investigation of existing options, and the gap.

### In progress (v0.1)
- LeRobotDataset v3.0 reader (parquet index + tabular + video locator).
- VideoToolbox hardware decode backend with IOSurface zero-copy output.
- Zero-copy MLX dataloader and the validation pass.
- Decode+load throughput benchmark harness vs. the PyAV/CPU baseline.

[Unreleased]: https://github.com/Mullassery/PyRoboFrames/commits/main
