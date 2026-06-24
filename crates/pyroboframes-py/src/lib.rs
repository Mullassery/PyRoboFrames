//! PyO3 bindings exposing the PyRoboFrames engine as the `pyroboframes._core` extension module.
//!
//! v0.1 (functional) surface: open a LeRobotDataset v3.0 and iterate a dataloader that yields
//! batches of tabular features (state, action, …) as NumPy arrays. Video frames / MLX output
//! layer on once a decode backend lands. All logic lives in `pyroboframes-core`.

// PyO3 0.22's `#[pymethods]` codegen emits an identity `.into()` on the return value of
// `#[staticmethod]` / `#[pyo3(signature = ...)]` methods, which clippy flags as
// useless_conversion in code we don't control. Silenced crate-wide as this crate is only the
// thin binding shell (all real logic lives in `pyroboframes-core`).
#![allow(clippy::useless_conversion)]

use std::path::PathBuf;
use std::sync::Arc;

use numpy::ndarray::{Array2, Array3};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pyroboframes_core::dataset::Dataset;
use pyroboframes_core::loader::{Sample, TabularLoader, WindowedSample};
use pyroboframes_core::sampler::Sampler;
use pyroboframes_core::window::WindowSpec;

fn core_err(e: pyroboframes_core::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// An opened LeRobotDataset v3.0.
#[pyclass]
struct RoboFrameDataset {
    dataset: Arc<Dataset>,
}

#[pymethods]
impl RoboFrameDataset {
    /// Open a dataset from a local path (the directory holding `meta/`, `data/`, `videos/`).
    #[staticmethod]
    fn from_path(path: PathBuf) -> PyResult<Self> {
        match Dataset::open(&path) {
            Ok(dataset) => Ok(Self {
                dataset: Arc::new(dataset),
            }),
            Err(e) => Err(core_err(e)),
        }
    }

    #[getter]
    fn num_frames(&self) -> usize {
        self.dataset.num_frames()
    }

    #[getter]
    fn num_episodes(&self) -> usize {
        self.dataset.num_episodes()
    }

    #[getter]
    fn fps(&self) -> f64 {
        self.dataset.fps()
    }

    #[getter]
    fn cameras(&self) -> Vec<String> {
        self.dataset.cameras()
    }

    /// Build a dataloader over the tabular (state/action) features.
    ///
    /// `delta_timestamps` (optional) maps a feature to a list of time offsets in seconds, e.g.
    /// `{"observation.state": [-0.1, 0.0]}`, producing a temporal window per sample; matching is
    /// validated against `tolerance_s`.
    #[pyo3(signature = (batch_size=32, shuffle=true, shuffle_buffer=1024, seed=0, drop_last=false, delta_timestamps=None, tolerance_s=1e-4))]
    #[allow(clippy::too_many_arguments)]
    fn loader(
        &self,
        batch_size: usize,
        shuffle: bool,
        shuffle_buffer: usize,
        seed: u64,
        drop_last: bool,
        delta_timestamps: Option<Bound<'_, PyDict>>,
        tolerance_s: f64,
    ) -> PyResult<Loader> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be >= 1"));
        }
        let window = match delta_timestamps {
            None => None,
            Some(dict) => {
                let mut spec = WindowSpec {
                    tolerance_s,
                    ..Default::default()
                };
                for (k, v) in dict.iter() {
                    let key: String = k.extract()?;
                    let deltas: Vec<f64> = v.extract()?;
                    spec.delta_timestamps.insert(key, deltas);
                }
                Some(spec)
            }
        };
        let inner = match TabularLoader::open(self.dataset.clone()) {
            Ok(inner) => inner,
            Err(e) => return Err(core_err(e)),
        };
        let order = Sampler::new(shuffle, shuffle_buffer, seed).order(inner.total_frames(), 0);
        Ok(Loader {
            inner,
            order,
            cursor: 0,
            batch_size,
            drop_last,
            window,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RoboFrameDataset(episodes={}, frames={}, cameras={:?})",
            self.dataset.num_episodes(),
            self.dataset.num_frames(),
            self.dataset.cameras(),
        )
    }
}

/// Iterable dataloader yielding dict batches of NumPy arrays, one entry per tabular feature
/// (shape `[batch, feature_dim]`) plus an `episode_index` vector.
#[pyclass]
struct Loader {
    inner: TabularLoader,
    order: Vec<usize>,
    cursor: usize,
    batch_size: usize,
    drop_last: bool,
    /// When set, each feature is returned as a `[batch, num_deltas, dim]` temporal window.
    window: Option<WindowSpec>,
}

#[pymethods]
impl Loader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        if self.cursor >= self.order.len() {
            return Ok(None);
        }
        let end = (self.cursor + self.batch_size).min(self.order.len());
        if self.drop_last && end - self.cursor < self.batch_size {
            return Ok(None);
        }
        let indices = self.order[self.cursor..end].to_vec();
        self.cursor = end;

        // Clone the spec so we don't hold a borrow of `self` while calling `&mut self.inner`.
        match self.window.clone() {
            None => {
                let samples = match self.inner.batch(&indices) {
                    Ok(samples) => samples,
                    Err(e) => return Err(core_err(e)),
                };
                Ok(Some(batch_to_dict(py, &samples)?))
            }
            Some(spec) => {
                let mut samples = Vec::with_capacity(indices.len());
                for i in indices {
                    match self.inner.windowed_sample(i, &spec) {
                        Ok(s) => samples.push(s),
                        Err(e) => return Err(core_err(e)),
                    }
                }
                Ok(Some(windowed_batch_to_dict(py, &samples)?))
            }
        }
    }

    /// Number of batches in one epoch.
    fn __len__(&self) -> usize {
        if self.batch_size == 0 {
            return 0;
        }
        if self.drop_last {
            self.order.len() / self.batch_size
        } else {
            self.order.len().div_ceil(self.batch_size)
        }
    }
}

/// Stack a batch of samples into a dict of NumPy arrays.
fn batch_to_dict(py: Python<'_>, samples: &[Sample]) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    if samples.is_empty() {
        return Ok(dict.unbind());
    }
    let n = samples.len();

    // One [n, dim] float32 array per feature.
    for name in samples[0].features.keys() {
        let dim = samples[0].features[name].len();
        let mut data = Vec::with_capacity(n * dim);
        for s in samples {
            let v = s
                .features
                .get(name)
                .ok_or_else(|| PyValueError::new_err(format!("sample missing feature `{name}`")))?;
            if v.len() != dim {
                return Err(PyValueError::new_err(format!(
                    "feature `{name}` has inconsistent dim ({} vs {dim})",
                    v.len()
                )));
            }
            data.extend_from_slice(v);
        }
        let arr = Array2::from_shape_vec((n, dim), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dict.set_item(name, arr.into_pyarray_bound(py))?;
    }

    // Provenance: episode index per row.
    let episodes: Vec<i64> = samples.iter().map(|s| s.episode_index as i64).collect();
    dict.set_item("episode_index", episodes.into_pyarray_bound(py))?;

    Ok(dict.unbind())
}

/// Stack windowed samples into `[batch, num_deltas, dim]` arrays per feature.
fn windowed_batch_to_dict(py: Python<'_>, samples: &[WindowedSample]) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    if samples.is_empty() {
        return Ok(dict.unbind());
    }
    let n = samples.len();

    for name in samples[0].features.keys() {
        let steps = &samples[0].features[name];
        let nd = steps.len();
        let dim = steps.first().map(|v| v.len()).unwrap_or(0);
        let mut data = Vec::with_capacity(n * nd * dim);
        for s in samples {
            let feat = s
                .features
                .get(name)
                .ok_or_else(|| PyValueError::new_err(format!("sample missing feature `{name}`")))?;
            if feat.len() != nd {
                return Err(PyValueError::new_err(format!(
                    "feature `{name}` has inconsistent window length ({} vs {nd})",
                    feat.len()
                )));
            }
            for step in feat {
                if step.len() != dim {
                    return Err(PyValueError::new_err(format!(
                        "feature `{name}` has inconsistent dim ({} vs {dim})",
                        step.len()
                    )));
                }
                data.extend_from_slice(step);
            }
        }
        let arr = Array3::from_shape_vec((n, nd, dim), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dict.set_item(name, arr.into_pyarray_bound(py))?;
    }

    let episodes: Vec<i64> = samples.iter().map(|s| s.episode_index as i64).collect();
    dict.set_item("episode_index", episodes.into_pyarray_bound(py))?;

    Ok(dict.unbind())
}

/// `pyroboframes._core` — the compiled extension module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", pyroboframes_core::VERSION)?;
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    m.add_class::<RoboFrameDataset>()?;
    m.add_class::<Loader>()?;
    Ok(())
}

/// Return the Rust engine version.
#[pyfunction]
fn engine_version() -> &'static str {
    pyroboframes_core::VERSION
}
