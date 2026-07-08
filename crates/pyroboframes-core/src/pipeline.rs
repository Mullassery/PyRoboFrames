//! Off-GIL prefetch pipeline.
//!
//! Background worker threads assemble batches *ahead* of consumption so the training loop never
//! waits on parquet reads + video decode. Each worker owns its own [`BatchAssembler`] (no shared
//! mutable state); a token semaphore bounds how many batches are in flight (backpressure), and a
//! small reorder buffer restores epoch order regardless of which worker finishes first.
//!
//! Assembly is Python-free — workers produce [`RustBatch`] (plain Rust buffers + shapes); the
//! caller turns those into NumPy/MLX/Torch arrays under the GIL. This is what lets the heavy work
//! run with the GIL released.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

use crate::dataset::Dataset;
use crate::decode::{Decoder, FrameCache};
use crate::loader::TabularLoader;
use crate::window::WindowSpec;
use crate::{Error, Result};

/// Builds a frame decoder. Provided by the binding layer (the FFmpeg decoder lives behind a
/// cargo feature there), so the core stays decoder-agnostic.
pub type DecoderFactory = fn() -> Result<Box<dyn Decoder + Send>>;

/// A fully-assembled, Python-free batch: row-major buffers plus their shapes.
#[derive(Default)]
pub struct RustBatch {
    /// feature -> (data, shape). Shape is `[n, dim]` or, when windowed, `[n, steps, dim]`.
    pub features: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    /// One episode index per row.
    pub episode_index: Vec<i64>,
    /// camera -> (RGB bytes, `[n, h, w, 3]`).
    pub frames: BTreeMap<String, (Vec<u8>, [usize; 4])>,
}

/// Everything a worker needs to (re)build an assembler from scratch.
#[derive(Clone)]
pub struct AssemblerConfig {
    pub dataset: Arc<Dataset>,
    pub features: Option<Vec<String>>,
    pub normalize: Vec<String>,
    pub window: Option<WindowSpec>,
    pub cameras: Vec<String>,
    pub batch_size: usize,
    pub decoder_factory: Option<DecoderFactory>,
    /// Override LRU frame cache capacity (frames). None = auto: batch_size × num_cameras × 8.
    pub cache_size: Option<usize>,
    /// Pre-fetch the next episode's opening frames when an episode boundary is detected.
    pub episode_prefetch: bool,
}

/// Assembles one batch of indices into a [`RustBatch`]. Not `Sync` (holds a decoder + cache), but
/// `Send`, so each worker thread owns one.
pub struct BatchAssembler {
    loader: TabularLoader,
    window: Option<WindowSpec>,
    cameras: Vec<String>,
    decoder: Option<Box<dyn Decoder + Send>>,
    cache: FrameCache,
}

impl BatchAssembler {
    pub fn build(cfg: &AssemblerConfig) -> Result<Self> {
        let mut loader = TabularLoader::with_features(cfg.dataset.clone(), cfg.features.clone())?;
        if !cfg.normalize.is_empty() {
            loader.enable_normalization(&cfg.normalize)?;
        }
        let (decoder, cache) = if cfg.cameras.is_empty() {
            (None, FrameCache::new(1))
        } else {
            let factory = cfg.decoder_factory.ok_or_else(|| {
                Error::Decode("camera decode requested but no decoder is available".into())
            })?;
            let cap = cfg
                .cache_size
                .unwrap_or_else(|| (cfg.batch_size * cfg.cameras.len() * 8).max(256));
            (Some(factory()?), FrameCache::new(cap))
        };
        Ok(Self {
            loader,
            window: cfg.window.clone(),
            cameras: cfg.cameras.clone(),
            decoder,
            cache,
        })
    }

    /// Assemble the tabular features (+ frames, if cameras are configured) for `indices`.
    pub fn assemble(&mut self, indices: &[usize]) -> Result<RustBatch> {
        let mut batch = RustBatch::default();

        match self.window.clone() {
            None => {
                let samples = self.loader.batch(indices)?;
                batch.episode_index = samples.iter().map(|s| s.episode_index as i64).collect();
                if let Some(first) = samples.first() {
                    for name in first.features.keys() {
                        let dim = first.features[name].len();
                        let mut data = Vec::with_capacity(samples.len() * dim);
                        for s in &samples {
                            data.extend_from_slice(&s.features[name]);
                        }
                        batch
                            .features
                            .insert(name.clone(), (data, vec![samples.len(), dim]));
                    }
                }
            }
            Some(spec) => {
                let mut samples = Vec::with_capacity(indices.len());
                for &i in indices {
                    samples.push(self.loader.windowed_sample(i, &spec)?);
                }
                batch.episode_index = samples.iter().map(|s| s.episode_index as i64).collect();
                if let Some(first) = samples.first() {
                    for name in first.features.keys() {
                        let steps = &first.features[name];
                        let nd = steps.len();
                        let dim = steps.first().map(|v| v.len()).unwrap_or(0);
                        let mut data = Vec::with_capacity(samples.len() * nd * dim);
                        for s in &samples {
                            for step in &s.features[name] {
                                data.extend_from_slice(step);
                            }
                        }
                        batch
                            .features
                            .insert(name.clone(), (data, vec![samples.len(), nd, dim]));
                    }
                }
            }
        }

        if !self.cameras.is_empty() {
            let cameras = self.cameras.clone();
            let decoder = self
                .decoder
                .as_deref_mut()
                .expect("decoder present with cameras");
            // camera -> (width, height, concatenated RGB bytes)
            let mut acc: BTreeMap<String, (u32, u32, Vec<u8>)> = BTreeMap::new();
            for &i in indices {
                let frames = self
                    .loader
                    .frames_for(i, &cameras, decoder, &mut self.cache)?;
                for (cam, frame) in frames {
                    let entry = acc
                        .entry(cam)
                        .or_insert((frame.width, frame.height, Vec::new()));
                    if entry.0 != frame.width || entry.1 != frame.height {
                        return Err(Error::Decode(
                            "frames in a batch have inconsistent dimensions".into(),
                        ));
                    }
                    entry.2.extend_from_slice(frame.pixels.as_bytes());
                }
            }
            let n = indices.len();
            for (cam, (w, h, data)) in acc {
                batch
                    .frames
                    .insert(cam, (data, [n, h as usize, w as usize, 3]));
            }
        }

        Ok(batch)
    }
}

/// Split an epoch order into batch jobs `(seq, indices)`.
fn make_jobs(order: &[usize], batch_size: usize, drop_last: bool) -> Vec<(usize, Vec<usize>)> {
    let mut jobs = Vec::new();
    let mut i = 0;
    let mut seq = 0;
    while i < order.len() {
        let end = (i + batch_size).min(order.len());
        if drop_last && end - i < batch_size {
            break;
        }
        jobs.push((seq, order[i..end].to_vec()));
        seq += 1;
        i = end;
    }
    jobs
}

/// A running prefetch pipeline. Pulls assembled batches in epoch order via [`next_batch`].
pub struct Prefetcher {
    result_rx: Receiver<(usize, Result<RustBatch>)>,
    /// Released (one token) each time a batch is consumed, letting the feeder dispatch one more.
    token_tx: Sender<()>,
    buffer: BTreeMap<usize, Result<RustBatch>>,
    next_seq: usize,
    num_batches: usize,
    _workers: Vec<JoinHandle<()>>,
    _feeder: JoinHandle<()>,
}

impl Prefetcher {
    /// Spawn `num_workers` assembler threads over `order`. At most `prefetch` batches are kept in
    /// flight (memory backpressure).
    pub fn start(
        cfg: AssemblerConfig,
        order: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        prefetch: usize,
    ) -> Result<Self> {
        let jobs = make_jobs(&order, batch_size, drop_last);
        let num_batches = jobs.len();
        let num_workers = num_workers.max(1);
        let prefetch = prefetch.max(num_workers); // never starve the workers

        // Token semaphore: `prefetch` permits; one consumed per dispatched job, returned per
        // consumed batch. Bounds in-flight batches and gives the feeder backpressure.
        let (token_tx, token_rx) = bounded::<()>(prefetch);
        for _ in 0..prefetch {
            token_tx.send(()).expect("prime tokens");
        }

        let (job_tx, job_rx) = unbounded::<(usize, Vec<usize>)>();
        let (result_tx, result_rx) = unbounded::<(usize, Result<RustBatch>)>();

        // Feeder: dispatch jobs in order, blocking until a token is free.
        let feeder = thread::spawn(move || {
            for job in jobs {
                if token_rx.recv().is_err() {
                    break; // consumer gone
                }
                if job_tx.send(job).is_err() {
                    break; // workers gone
                }
            }
            // dropping job_tx closes the queue -> workers finish
        });

        // Workers: each builds its own assembler, then drains jobs.
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let job_rx = job_rx.clone();
            let result_tx = result_tx.clone();
            let cfg = cfg.clone();
            workers.push(thread::spawn(move || {
                let mut assembler = match BatchAssembler::build(&cfg) {
                    Ok(a) => a,
                    Err(e) => {
                        // Surface the build failure on whatever jobs we can claim, then stop.
                        let msg = e.to_string();
                        for (seq, _) in job_rx.iter() {
                            if result_tx
                                .send((seq, Err(Error::Dataset(msg.clone()))))
                                .is_err()
                            {
                                break;
                            }
                        }
                        return;
                    }
                };
                for (seq, indices) in job_rx.iter() {
                    let r = assembler.assemble(&indices);
                    if result_tx.send((seq, r)).is_err() {
                        break; // consumer gone
                    }
                }
            }));
        }

        Ok(Self {
            result_rx,
            token_tx,
            buffer: BTreeMap::new(),
            next_seq: 0,
            num_batches,
            _workers: workers,
            _feeder: feeder,
        })
    }

    /// Number of batches in the epoch.
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// The next batch in epoch order, or `Ok(None)` when the epoch is exhausted.
    pub fn next_batch(&mut self) -> Result<Option<RustBatch>> {
        if self.next_seq >= self.num_batches {
            return Ok(None);
        }
        loop {
            if let Some(slot) = self.buffer.remove(&self.next_seq) {
                self.next_seq += 1;
                let _ = self.token_tx.send(()); // return a permit
                return slot.map(Some);
            }
            match self.result_rx.recv() {
                Ok((seq, res)) => {
                    self.buffer.insert(seq, res);
                }
                Err(_) => {
                    // All senders gone but we still expected batches (worker panicked).
                    return Ok(None);
                }
            }
        }
    }
}
