//! Dataset integrity validation (metadata-level; no video decode required).
//!
//! Checks that the episode index is internally consistent — contiguous global frame ranges,
//! per-episode lengths, sane per-camera timestamp bounds, and totals matching `info.json`.
//! Catches the corruption classes that silently break training before a run starts.

use crate::dataset::Dataset;
use crate::{Result, ValidationReport};

/// Validate a dataset's metadata. Returns a report; `Err` only on I/O failures reading the index.
pub fn validate(dataset: &Dataset) -> Result<ValidationReport> {
    let mut report = ValidationReport::default();
    let info = dataset.info();
    let index = dataset.episodes()?;
    let episodes = index.episodes(); // sorted by from_index

    if episodes.is_empty() {
        report.errors.push("dataset has no episodes".into());
        return Ok(report);
    }
    if info.camera_keys().is_empty() {
        report
            .warnings
            .push("no camera (video) features found in schema".into());
    }

    let mut expected_from = 0usize;
    for ep in episodes {
        let id = ep.episode_index;

        if ep.from_index != expected_from {
            report.errors.push(format!(
                "episode {id}: starts at global frame {} but expected {expected_from} \
                 (gap or overlap in the index)",
                ep.from_index
            ));
        }
        if ep.to_index < ep.from_index {
            report
                .errors
                .push(format!("episode {id}: to_index < from_index"));
        }
        let span = ep.to_index.saturating_sub(ep.from_index);
        if span != ep.length {
            report.warnings.push(format!(
                "episode {id}: recorded length {} != frame span {span}",
                ep.length
            ));
        }
        for (cam, v) in &ep.videos {
            if v.to_timestamp < v.from_timestamp {
                report.errors.push(format!(
                    "episode {id} camera `{cam}`: to_timestamp ({}) < from_timestamp ({})",
                    v.to_timestamp, v.from_timestamp
                ));
            }
        }
        expected_from = ep.to_index;
    }

    let total = episodes.last().map(|e| e.to_index).unwrap_or(0);
    if total != info.total_frames {
        report.warnings.push(format!(
            "episode index spans {total} frames but info.json says total_frames = {}",
            info.total_frames
        ));
    }

    Ok(report)
}
