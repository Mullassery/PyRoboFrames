//! PyO3 bindings exposing the PyRoboFrames engine as the `pyroboframes._core` extension
//! module. This crate is intentionally thin: all logic lives in `pyroboframes-core`. The
//! ergonomic, user-facing API is layered on top in `python/pyroboframes/`.

use pyo3::prelude::*;

/// `pyroboframes._core` — the compiled extension module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", pyroboframes_core::VERSION)?;
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    Ok(())
}

/// Return the Rust engine version. Smoke-test entry point until the dataset/loader bindings land.
#[pyfunction]
fn engine_version() -> &'static str {
    pyroboframes_core::VERSION
}
