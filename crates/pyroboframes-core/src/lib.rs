//! PyRoboFrames core engine.
//!
//! Pure-Rust, Python-free engine that reads robot-learning datasets, hardware-decodes their
//! video, and assembles time-synced training windows in zero-copy shared-memory buffers.
//! The Python bindings live in the `pyroboframes-py` crate; this crate is independently
//! unit-testable. See `ARCHITECTURE.md` for the full design.

use thiserror::Error;

pub mod data;
pub mod dataset;
pub mod decode;
pub mod episodes;
pub mod info;

pub use decode::{Decoder, Frame};

/// Errors surfaced by the engine.
#[derive(Debug, Error)]
pub enum Error {
    #[error("dataset error: {0}")]
    Dataset(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("validation failed with {0} error(s)")]
    Validation(usize),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// How a dataset's samples should be drawn and assembled into batches.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    pub batch_size: usize,
    pub cameras: Vec<String>,
    /// Frames of temporal context per sample.
    pub window: usize,
    pub shuffle: bool,
    pub num_workers: usize,
    /// Bounded prefetch depth (backpressure).
    pub prefetch: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            cameras: Vec::new(),
            window: 1,
            shuffle: true,
            num_workers: 4,
            prefetch: 4,
        }
    }
}

/// Outcome of a dataset validation pass.
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationReport {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn raise_if_errors(&self) -> Result<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Validation(self.errors.len()))
        }
    }
}

/// Version string for the engine, surfaced to Python as `pyroboframes.__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_distinguishes_errors_from_warnings() {
        let mut r = ValidationReport::default();
        r.warnings.push("non-fatal".into());
        assert!(r.is_ok());
        r.errors.push("missing frame".into());
        assert!(!r.is_ok());
        assert!(r.raise_if_errors().is_err());
    }

    #[test]
    fn loader_config_defaults_are_sane() {
        let c = LoaderConfig::default();
        assert!(c.batch_size > 0 && c.num_workers > 0 && c.prefetch > 0);
    }
}
