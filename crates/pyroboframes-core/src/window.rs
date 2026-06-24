//! Temporal windowing — LeRobot-style `delta_timestamps`.
//!
//! For a query frame, a feature can request several time offsets (seconds) relative to it, e.g.
//! `[-0.1, 0.0]` for one step of history plus the current frame. This module turns those deltas
//! into concrete in-episode frame offsets, snapping to the nearest frame (validated against a
//! tolerance) and clamping at episode boundaries (edge-repeat padding).

use std::collections::BTreeMap;

use crate::{Error, Result};

/// A temporal-context request: per feature, time offsets in seconds. A feature with no entry
/// (or an empty list) yields just the current frame (`[0.0]`).
#[derive(Debug, Clone, Default)]
pub struct WindowSpec {
    pub delta_timestamps: BTreeMap<String, Vec<f64>>,
    /// Max allowed gap (seconds) between a requested time and the nearest actual frame.
    pub tolerance_s: f64,
}

impl WindowSpec {
    /// Offsets requested for `feature`, defaulting to the current frame.
    pub fn deltas_for(&self, feature: &str) -> &[f64] {
        match self.delta_timestamps.get(feature) {
            Some(v) if !v.is_empty() => v,
            _ => &[0.0],
        }
    }
}

/// Resolve time deltas to 0-based in-episode frame offsets, clamped to `[0, episode_len)`.
/// Errors if a delta does not align to a frame within `tolerance_s`.
pub fn resolve_offsets(
    deltas: &[f64],
    frame_in_episode: usize,
    episode_len: usize,
    fps: f64,
    tolerance_s: f64,
) -> Result<Vec<usize>> {
    if episode_len == 0 {
        return Err(Error::Dataset("empty episode".into()));
    }
    let mut out = Vec::with_capacity(deltas.len());
    for &d in deltas {
        let target = frame_in_episode as f64 + d * fps;
        let nearest = target.round();
        let residual_s = (target - nearest).abs() / fps;
        if residual_s > tolerance_s {
            return Err(Error::Dataset(format!(
                "delta {d}s does not align to a frame within tolerance {tolerance_s}s \
                 (off by {residual_s:.6}s)"
            )));
        }
        let clamped = nearest.clamp(0.0, (episode_len - 1) as f64) as usize;
        out.push(clamped);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_frame_by_default() {
        let spec = WindowSpec::default();
        assert_eq!(spec.deltas_for("anything"), &[0.0]);
    }

    #[test]
    fn resolves_history_window() {
        // frame 10 of a 50-frame episode at 30 fps, deltas [-0.1, 0.0] -> [7, 10]
        let offs = resolve_offsets(&[-0.1, 0.0], 10, 50, 30.0, 1e-3).unwrap();
        assert_eq!(offs, vec![7, 10]);
    }

    #[test]
    fn clamps_at_episode_start_and_end() {
        // frame 0: history clamps to 0 (edge-repeat)
        assert_eq!(
            resolve_offsets(&[-0.1, 0.0], 0, 50, 30.0, 1e-3).unwrap(),
            vec![0, 0]
        );
        // last frame: future clamps to last index
        assert_eq!(
            resolve_offsets(&[0.0, 0.1], 49, 50, 30.0, 1e-3).unwrap(),
            vec![49, 49]
        );
    }

    #[test]
    fn rejects_misaligned_delta_under_tight_tolerance() {
        // 0.05s at 30 fps = 1.5 frames -> 0.5-frame (16.7 ms) residual; tolerance 1 ms -> error
        assert!(resolve_offsets(&[0.05], 10, 50, 30.0, 1e-3).is_err());
        // generous tolerance accepts it (snaps to nearest)
        assert!(resolve_offsets(&[0.05], 10, 50, 30.0, 0.02).is_ok());
    }
}
