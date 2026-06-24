# Contributing to PyRoboFrames

Thanks for your interest! PyRoboFrames is a Rust engine with a Python API, built with
[PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs).

## Project layout

```
crates/pyroboframes-core/   Rust engine (no Python) — most logic lives here, unit-tested with `cargo test`
crates/pyroboframes-py/     Thin PyO3 binding → the `pyroboframes._core` extension module
python/pyroboframes/        Ergonomic Python API + MLX/NumPy/PyTorch adapters
tests/                      Python integration tests
benches/                    Throughput benchmark harness
```

Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) first — it explains the design and the decisions
behind it.

## Dev setup (Apple Silicon)

```bash
# Rust toolchain via rustup, then:
pip install maturin pytest numpy
maturin develop            # builds the extension and installs it into your venv
pytest -q                  # Python tests
cargo test --workspace     # Rust tests
```

## Before opening a PR

- `cargo fmt --all` and `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` and `pytest -q` pass
- New behavior has tests (prefer Rust-side tests in `pyroboframes-core` where possible)
- Update `CHANGELOG.md` under "Unreleased"

## High-impact areas right now

- **MLX zero-copy init** from IOSurface / CVPixelBuffer — see
  [mlx#2855](https://github.com/ml-explore/mlx/issues/2855); the cleanest path may need an
  upstream MLX contribution.
- **Benchmark harness** — decode+load throughput vs. the PyAV/CPU baseline (the headline metric).
- **VideoToolbox decode backend** and IOSurface buffer-lifetime handling.

## License

By contributing, you agree your contributions are licensed under the [MIT License](./LICENSE).
