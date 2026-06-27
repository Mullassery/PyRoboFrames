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

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use numpy::ndarray::{Array2, Array3, Array4, ArrayD, IxDyn};
use numpy::IntoPyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pyroboframes_core::dataset::Dataset;
use pyroboframes_core::decode::{Decoder, FrameCache};
use pyroboframes_core::depth::PointCloud;
use pyroboframes_core::loader::{Sample, TabularLoader, WindowedSample};
use pyroboframes_core::pipeline::{AssemblerConfig, Prefetcher, RustBatch};
use pyroboframes_core::sampler::{chunked_order, weighted_with_replacement, Sampler};
use pyroboframes_core::window::WindowSpec;

/// Core-typed decoder factory handed to prefetch workers (each builds its own decoder).
#[cfg(feature = "ffmpeg")]
fn core_decoder_factory() -> pyroboframes_core::Result<Box<dyn Decoder + Send>> {
    Ok(Box::new(pyroboframes_core::decode::FfmpegDecoder::default()))
}

#[cfg(not(feature = "ffmpeg"))]
fn core_decoder_factory() -> pyroboframes_core::Result<Box<dyn Decoder + Send>> {
    Err(pyroboframes_core::Error::Decode(
        "frame decoding requires the 'ffmpeg' build feature (and ffmpeg/ffprobe on PATH)".into(),
    ))
}

/// Construct the frame decoder. FFmpeg-CLI based; available when built with the `ffmpeg` feature.
#[cfg(feature = "ffmpeg")]
fn new_frame_decoder() -> PyResult<Box<dyn Decoder + Send>> {
    Ok(Box::new(pyroboframes_core::decode::FfmpegDecoder::default()))
}

#[cfg(not(feature = "ffmpeg"))]
fn new_frame_decoder() -> PyResult<Box<dyn Decoder + Send>> {
    Err(PyRuntimeError::new_err(
        "frame decoding requires the 'ffmpeg' build feature (and ffmpeg/ffprobe on PATH)",
    ))
}

fn core_err(e: pyroboframes_core::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Result of [`RoboFrameDataset.validate`]: integrity `errors` and non-fatal `warnings`.
#[pyclass]
struct ValidationReport {
    #[pyo3(get)]
    errors: Vec<String>,
    #[pyo3(get)]
    warnings: Vec<String>,
}

#[pymethods]
impl ValidationReport {
    /// True when there are no errors (warnings are allowed).
    #[getter]
    fn ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Raise if validation found errors.
    fn raise_if_errors(&self) -> PyResult<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "dataset validation failed: {}",
                self.errors.join("; ")
            )))
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ValidationReport(ok={}, errors={}, warnings={})",
            self.errors.is_empty(),
            self.errors.len(),
            self.warnings.len()
        )
    }
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

    /// Validate dataset metadata integrity (frame-range contiguity, lengths, timestamps, totals).
    fn validate(&self) -> PyResult<ValidationReport> {
        match self.dataset.validate() {
            Ok(r) => Ok(ValidationReport {
                errors: r.errors,
                warnings: r.warnings,
            }),
            Err(e) => Err(core_err(e)),
        }
    }

    /// Per-feature statistics from `meta/stats.json` for normalization, as
    /// `{feature: {"mean", "std", "min", "max", "count"}}`. Returns `None` if the dataset has no
    /// stats file.
    fn stats(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(stats) = self.dataset.stats().map_err(core_err)? else {
            return Ok(None);
        };
        let out = PyDict::new_bound(py);
        for (name, fs) in &stats.features {
            let d = PyDict::new_bound(py);
            d.set_item("mean", fs.mean.clone())?;
            d.set_item("std", fs.std.clone())?;
            d.set_item("min", fs.min.clone())?;
            d.set_item("max", fs.max.clone())?;
            d.set_item("count", fs.count)?;
            out.set_item(name, d)?;
        }
        Ok(Some(out.unbind()))
    }

    /// Deterministic train/validation split over **episode** indices (split by episode, not by
    /// frame, to avoid temporal leakage). Returns `(train_episodes, val_episodes)`, both sorted.
    #[pyo3(signature = (val_fraction=0.1, seed=0))]
    fn train_val_split(&self, val_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
        self.dataset.train_val_split(val_fraction, seed)
    }

    /// Per-episode metadata, ordered by start frame: a list of dicts with `episode_index`,
    /// `length`, and the global frame range `[from_index, to_index)`. Iterate these and pass an
    /// index to `loader(episodes=[i])` to load a single episode.
    fn episodes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let index = self.dataset.episodes().map_err(core_err)?;
        let out = pyo3::types::PyList::empty_bound(py);
        for ep in index.episodes() {
            let d = PyDict::new_bound(py);
            d.set_item("episode_index", ep.episode_index)?;
            d.set_item("length", ep.length)?;
            d.set_item("from_index", ep.from_index)?;
            d.set_item("to_index", ep.to_index)?;
            out.append(d)?;
        }
        Ok(out.unbind().into_any())
    }

    /// Build a dataloader over the tabular (state/action) features.
    ///
    /// `delta_timestamps` (optional) maps a feature to a list of time offsets in seconds, e.g.
    /// `{"observation.state": [-0.1, 0.0]}`, producing a temporal window per sample; matching is
    /// validated against `tolerance_s`.
    ///
    /// `cameras` (optional) names the video streams to decode and include as `[batch, H, W, 3]`
    /// `uint8` arrays (requires an ffmpeg-enabled build with `ffmpeg`/`ffprobe` on `PATH`).
    ///
    /// `output` selects the array type per batch: `"numpy"` (default), `"mlx"`
    /// (`mlx.core.array`), `"torch"` (`torch.from_numpy`, zero-copy from the NumPy buffers), or
    /// `"jax"` (`jax.numpy.asarray`).
    ///
    /// `episodes` (optional) restricts iteration to the given episode indices — pass one half of
    /// `ds.train_val_split(...)` to build a train- or validation-only loader.
    ///
    /// `normalize` (optional) lists features to standardize as `(x - mean) / std` using the
    /// dataset's `meta/stats.json`.
    ///
    /// `num_workers` > 0 runs an **off-GIL prefetch pipeline** (that many background assembler
    /// threads, up to `prefetch` batches in flight); `0` (default) assembles synchronously on the
    /// calling thread. The prefetched loader supports `position` but not `seek`.
    ///
    /// `balanced=True` draws frames so every episode is sampled equally regardless of its length
    /// (weighted sampling with replacement) — useful for length-imbalanced demonstration sets.
    ///
    /// `chunk_size` > 0 switches to **episode-chunking** sampling: the population is cut into
    /// contiguous chunks of that many frames *within* each episode (never crossing a boundary),
    /// the chunks are shuffled as units (when `shuffle`), and frames stay in temporal order inside
    /// a chunk. This keeps decode locality and produces sequence-friendly batches; pair it with a
    /// matching `delta_timestamps` window for MLX/PyTorch sequence models. Ignored when
    /// `balanced=True`.
    ///
    /// `curriculum=True` orders the epoch easy→hard (shorter episodes first), instead of shuffling.
    ///
    /// `goal="final"` makes it **goal-conditioned**: each sample gains `<feature>.goal` columns
    /// holding the final frame of the same episode. (Synchronous tabular path only — not combined
    /// with `delta_timestamps` or `num_workers`.)
    #[pyo3(signature = (batch_size=32, shuffle=true, shuffle_buffer=1024, seed=0, drop_last=false, delta_timestamps=None, tolerance_s=1e-4, cameras=None, output="numpy".to_string(), episodes=None, normalize=None, num_workers=0, prefetch=4, balanced=false, chunk_size=0, curriculum=false, goal=None))]
    #[allow(clippy::too_many_arguments)]
    fn loader(
        &self,
        py: Python<'_>,
        batch_size: usize,
        shuffle: bool,
        shuffle_buffer: usize,
        seed: u64,
        drop_last: bool,
        delta_timestamps: Option<Bound<'_, PyDict>>,
        tolerance_s: f64,
        cameras: Option<Vec<String>>,
        output: String,
        episodes: Option<Vec<usize>>,
        normalize: Option<Vec<String>>,
        num_workers: usize,
        prefetch: usize,
        balanced: bool,
        chunk_size: usize,
        curriculum: bool,
        goal: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be >= 1"));
        }
        if !matches!(output.as_str(), "numpy" | "mlx" | "torch" | "jax") {
            return Err(PyValueError::new_err(format!(
                "output must be 'numpy', 'mlx', 'torch', or 'jax' (got '{output}')"
            )));
        }
        if let Some(g) = &goal {
            if g != "final" {
                return Err(PyValueError::new_err("goal must be 'final' (or None)"));
            }
            if delta_timestamps.is_some() || num_workers >= 1 {
                return Err(PyValueError::new_err(
                    "goal='final' is not supported with delta_timestamps or num_workers>0",
                ));
            }
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
        let cameras = cameras.unwrap_or_default();
        let normalize = normalize.unwrap_or_default();

        let mut inner = TabularLoader::open(self.dataset.clone()).map_err(core_err)?;
        if !normalize.is_empty() {
            inner.enable_normalization(&normalize).map_err(core_err)?;
        }
        // Population of global frame indices to draw from: all frames, or just the chosen episodes.
        let base: Vec<usize> = match &episodes {
            Some(eps) => inner.frame_indices_for_episodes(eps),
            None => (0..inner.total_frames()).collect(),
        };
        // Build the per-epoch order over positions in `base`, then map to global frame indices.
        // `balanced` weights each frame by 1/episode_len so episodes are drawn equally (with
        // replacement); `chunk_size` shuffles contiguous within-episode chunks (sequence-friendly,
        // decode-local); otherwise the sampler permutes the frames (optionally shuffled).
        let positions: Vec<usize> = if balanced {
            let weights: Vec<f64> = base
                .iter()
                .map(|&g| {
                    let len = inner.locate(g).map(|l| l.episode_len).unwrap_or(1).max(1);
                    1.0 / len as f64
                })
                .collect();
            weighted_with_replacement(&weights, base.len(), seed)
        } else if chunk_size >= 1 {
            let runs = inner.episode_runs(episodes.as_deref());
            chunked_order(&runs, chunk_size, shuffle, seed)
        } else if curriculum {
            // Easy→hard: stable-sort positions by (episode_len, episode_index), frames in order.
            let mut keyed: Vec<((usize, usize), usize)> = (0..base.len())
                .map(|i| (inner.curriculum_key(base[i]), i))
                .collect();
            keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            keyed.into_iter().map(|(_, i)| i).collect()
        } else {
            Sampler::new(shuffle, shuffle_buffer, seed).order(base.len(), 0)
        };
        let order: Vec<usize> = positions.into_iter().map(|i| base[i]).collect();

        // Prefetched (off-GIL) path: background workers assemble ahead of consumption.
        if num_workers >= 1 {
            let cfg = AssemblerConfig {
                dataset: self.dataset.clone(),
                features: None,
                normalize,
                window,
                cameras: cameras.clone(),
                batch_size,
                decoder_factory: if cameras.is_empty() {
                    None
                } else {
                    Some(core_decoder_factory)
                },
            };
            let prefetcher =
                Prefetcher::start(cfg, order, batch_size, drop_last, num_workers, prefetch)
                    .map_err(core_err)?;
            let loader = PrefetchLoader {
                prefetcher,
                output,
                consumed_rows: 0,
            };
            return Ok(Py::new(py, loader)?.into_any());
        }

        // Synchronous path (default).
        let (frame_decoder, frame_cache) = if cameras.is_empty() {
            (None, FrameCache::new(1))
        } else {
            let cap = (batch_size * cameras.len() * 8).max(256);
            (Some(new_frame_decoder()?), FrameCache::new(cap))
        };

        let loader = Loader {
            inner,
            order,
            cursor: 0,
            batch_size,
            drop_last,
            window,
            cameras,
            frame_decoder,
            frame_cache,
            output,
            goal: goal.is_some(),
        };
        Ok(Py::new(py, loader)?.into_any())
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
    /// Camera streams to decode into frame arrays (empty = tabular only).
    cameras: Vec<String>,
    frame_decoder: Option<Box<dyn Decoder + Send>>,
    frame_cache: FrameCache,
    /// "numpy" | "mlx" | "torch"
    output: String,
    /// Goal-conditioned: add `<feature>.goal` (the episode's final frame) to each sample.
    goal: bool,
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

        // Tabular features (clone the spec so we don't hold a borrow of `self` across `&mut`).
        let batch: Py<PyDict> = match self.window.clone() {
            None => {
                let samples = match self.inner.batch(&indices) {
                    Ok(samples) => samples,
                    Err(e) => return Err(core_err(e)),
                };
                batch_to_dict(py, &samples)?
            }
            Some(spec) => {
                let mut samples = Vec::with_capacity(indices.len());
                for &i in &indices {
                    match self.inner.windowed_sample(i, &spec) {
                        Ok(s) => samples.push(s),
                        Err(e) => return Err(core_err(e)),
                    }
                }
                windowed_batch_to_dict(py, &samples)?
            }
        };

        // Goal-conditioning: append the episode's final-frame features as `<name>.goal`.
        if self.goal {
            let goal_indices: Vec<usize> = indices
                .iter()
                .map(|&g| self.inner.goal_index(g))
                .collect::<pyroboframes_core::Result<_>>()
                .map_err(core_err)?;
            let goal_samples = self.inner.batch(&goal_indices).map_err(core_err)?;
            let goal_dict = batch_to_dict(py, &goal_samples)?;
            let dst = batch.bind(py);
            for (k, v) in goal_dict.bind(py).iter() {
                let key: String = k.extract()?;
                if key == "episode_index" {
                    continue;
                }
                dst.set_item(format!("{key}.goal"), v)?;
            }
        }

        // Camera frames. Without a window: [batch, H, W, 3] per camera. With a window: a temporal
        // stack [batch, steps, H, W, 3], the per-camera deltas resolved like the tabular window.
        if !self.cameras.is_empty() {
            let cameras = self.cameras.clone();
            let n = indices.len();
            let dict = batch.bind(py);
            match self.window.clone() {
                None => {
                    let decoder = self.frame_decoder.as_deref_mut().expect("decoder set");
                    // camera -> (width, height, concatenated RGB bytes)
                    let mut acc: BTreeMap<String, (u32, u32, Vec<u8>)> = BTreeMap::new();
                    for &i in &indices {
                        let frames = self
                            .inner
                            .frames_for(i, &cameras, decoder, &mut self.frame_cache)
                            .map_err(core_err)?;
                        for (cam, frame) in frames {
                            let entry = acc
                                .entry(cam)
                                .or_insert((frame.width, frame.height, Vec::new()));
                            if entry.0 != frame.width || entry.1 != frame.height {
                                return Err(PyValueError::new_err(
                                    "frames in a batch have inconsistent dimensions",
                                ));
                            }
                            entry.2.extend_from_slice(frame.pixels.as_bytes());
                        }
                    }
                    for (cam, (w, h, data)) in acc {
                        let arr = Array4::from_shape_vec((n, h as usize, w as usize, 3), data)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                        dict.set_item(cam, arr.into_pyarray_bound(py))?;
                    }
                }
                Some(spec) => {
                    let decoder = self.frame_decoder.as_deref_mut().expect("decoder set");
                    // camera -> (steps, width, height, concatenated RGB bytes)
                    let mut acc: BTreeMap<String, (usize, u32, u32, Vec<u8>)> = BTreeMap::new();
                    for &i in &indices {
                        let cam_frames = self
                            .inner
                            .windowed_frames_for(i, &cameras, &spec, decoder, &mut self.frame_cache)
                            .map_err(core_err)?;
                        for (cam, frames) in cam_frames {
                            let steps = frames.len();
                            let (w, h) = frames
                                .first()
                                .map(|f| (f.width, f.height))
                                .unwrap_or((0, 0));
                            let entry = acc.entry(cam).or_insert((steps, w, h, Vec::new()));
                            if entry.0 != steps || entry.1 != w || entry.2 != h {
                                return Err(PyValueError::new_err(
                                    "windowed frames have inconsistent steps or dimensions",
                                ));
                            }
                            for f in &frames {
                                entry.3.extend_from_slice(f.pixels.as_bytes());
                            }
                        }
                    }
                    for (cam, (steps, w, h, data)) in acc {
                        let shape = IxDyn(&[n, steps, h as usize, w as usize, 3]);
                        let arr = ArrayD::from_shape_vec(shape, data)
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                        dict.set_item(cam, arr.into_pyarray_bound(py))?;
                    }
                }
            }
        }

        // Convert every array to the requested framework (NumPy is the native form).
        if self.output != "numpy" {
            convert_batch(py, batch.bind(py), &self.output)?;
        }

        Ok(Some(batch))
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

    /// Frames consumed so far this epoch — save this to checkpoint a run.
    #[getter]
    fn position(&self) -> usize {
        self.cursor
    }

    /// Resume an interrupted epoch: rebuild the loader with the same `seed`/`shuffle`, then
    /// `seek(position)` to skip the frames already consumed (clamped to the epoch length).
    fn seek(&mut self, position: usize) {
        self.cursor = position.min(self.order.len());
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

/// Convert every value in `dict` from a NumPy array to the requested framework's array type
/// in place. `mlx` -> `mlx.core.array` (copy into unified memory); `torch` -> `torch.from_numpy`
/// (zero-copy view of the NumPy buffer); `jax` -> `jax.numpy.asarray`.
fn convert_batch(py: Python<'_>, dict: &Bound<'_, PyDict>, output: &str) -> PyResult<()> {
    let (module_name, func_name) = match output {
        "mlx" => ("mlx.core", "array"),
        "torch" => ("torch", "from_numpy"),
        "jax" => ("jax.numpy", "asarray"),
        other => return Err(PyValueError::new_err(format!("unknown output '{other}'"))),
    };
    let func = py
        .import_bound(module_name)
        .map_err(|e| {
            PyRuntimeError::new_err(format!("output='{output}' requires {module_name}: {e}"))
        })?
        .getattr(func_name)?;

    // Collect keys first to avoid mutating the dict while iterating it.
    let keys: Vec<Py<PyAny>> = dict.keys().iter().map(|k| k.unbind()).collect();
    for key in keys {
        let key = key.bind(py);
        if let Some(val) = dict.get_item(key)? {
            dict.set_item(key, func.call1((val,))?)?;
        }
    }
    Ok(())
}

/// Convert a Python-free [`RustBatch`] (raw buffers + shapes) into a dict of NumPy arrays.
fn rustbatch_to_dict(py: Python<'_>, batch: RustBatch) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    for (name, (data, shape)) in batch.features {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dict.set_item(name, arr.into_pyarray_bound(py))?;
    }
    for (cam, (data, shape)) in batch.frames {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        dict.set_item(cam, arr.into_pyarray_bound(py))?;
    }
    dict.set_item("episode_index", batch.episode_index.into_pyarray_bound(py))?;
    Ok(dict.unbind())
}

/// Iterable dataloader backed by the off-GIL prefetch pipeline (`num_workers > 0`). Yields the
/// same batch dicts as [`Loader`], but background threads assemble them ahead of consumption and
/// the blocking wait releases the GIL.
#[pyclass]
struct PrefetchLoader {
    prefetcher: Prefetcher,
    output: String,
    consumed_rows: usize,
}

#[pymethods]
impl PrefetchLoader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        // Wait for the next assembled batch with the GIL released.
        let batch = py
            .allow_threads(|| self.prefetcher.next_batch())
            .map_err(core_err)?;
        let Some(batch) = batch else {
            return Ok(None);
        };
        self.consumed_rows += batch.episode_index.len();
        let dict = rustbatch_to_dict(py, batch)?;
        if self.output != "numpy" {
            convert_batch(py, dict.bind(py), &self.output)?;
        }
        Ok(Some(dict))
    }

    /// Number of batches in one epoch.
    fn __len__(&self) -> usize {
        self.prefetcher.num_batches()
    }

    /// Rows consumed so far this epoch (for progress/checkpointing).
    #[getter]
    fn position(&self) -> usize {
        self.consumed_rows
    }
}

/// Build the `{"topics": [...], "skipped": [...]}` dict shared by the converters.
fn conversion_report_to_dict(
    py: Python<'_>,
    report: &pyroboframes_core::mcap::ConversionReport,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new_bound(py);
    let topics = pyo3::types::PyList::empty_bound(py);
    for t in &report.topics {
        let d = PyDict::new_bound(py);
        d.set_item("topic", &t.topic)?;
        d.set_item("messages", t.messages)?;
        d.set_item("columns", t.columns)?;
        d.set_item("path", t.path.to_string_lossy())?;
        topics.append(d)?;
    }
    out.set_item("topics", topics)?;
    out.set_item("skipped", report.skipped_topics.clone())?;
    Ok(out.unbind())
}

/// Convert an MCAP robotics log into one Parquet table per topic under `out_dir` (created if
/// absent). Returns `{"topics": [{topic, messages, columns, path}], "skipped": [topic, …]}`.
/// Decodes `json` and `protobuf` (via the embedded descriptor set) and `cdr`/`ros2msg` topics;
/// any other encoding is listed in `skipped`.
#[pyfunction]
fn convert_mcap(py: Python<'_>, input: PathBuf, out_dir: PathBuf) -> PyResult<Py<PyDict>> {
    let report = pyroboframes_core::mcap::convert(&input, &out_dir).map_err(core_err)?;
    conversion_report_to_dict(py, &report)
}

/// Convert a ROS 2 bag (`rosbag2` SQLite `.db3`) into one Parquet table per CDR topic under
/// `out_dir`. Same return shape as [`convert_mcap`]; topics without an embedded `ros2msg`
/// definition or not CDR-serialized are listed in `skipped`.
#[pyfunction]
fn convert_ros2_bag(py: Python<'_>, input: PathBuf, out_dir: PathBuf) -> PyResult<Py<PyDict>> {
    let report = pyroboframes_core::rosbag::convert(&input, &out_dir).map_err(core_err)?;
    conversion_report_to_dict(py, &report)
}

/// `pyroboframes._core` — the compiled extension module.
/// Python wrapper for a point cloud (depth camera data).
#[pyclass]
struct PointCloudPy {
    inner: PointCloud,
}

#[pymethods]
impl PointCloudPy {
    /// Load a point cloud from file (.xyz, .ply, .pcd, .npy).
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let cloud = PointCloud::load(std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: cloud })
    }

    /// Number of points in the cloud.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the point cloud is empty.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get point positions as a NumPy array [N, 3].
    fn points(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut data: Vec<f32> = Vec::with_capacity(self.inner.points.len() * 3);
        for p in &self.inner.points {
            data.extend_from_slice(&p[..]);
        }
        let arr = Array2::from_shape_vec((self.inner.points.len(), 3), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py).unbind().into())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("PointCloud(points={}, colors={}, normals={})",
            self.inner.len(),
            self.inner.colors.is_some(),
            self.inner.normals.is_some()
        )
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", pyroboframes_core::VERSION)?;
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    m.add_function(wrap_pyfunction!(convert_mcap, m)?)?;
    m.add_function(wrap_pyfunction!(convert_ros2_bag, m)?)?;
    m.add_class::<RoboFrameDataset>()?;
    m.add_class::<Loader>()?;
    m.add_class::<PrefetchLoader>()?;
    m.add_class::<ValidationReport>()?;
    m.add_class::<PointCloudPy>()?;
    Ok(())
}

/// Return the Rust engine version.
#[pyfunction]
fn engine_version() -> &'static str {
    pyroboframes_core::VERSION
}
