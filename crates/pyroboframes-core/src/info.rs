//! Parsing of a LeRobotDataset v3.0 `meta/info.json`.
//!
//! `info.json` describes the dataset's frame rate, feature schema (including which features
//! are camera video streams), totals, chunking, and the path templates used to locate the
//! parquet/mp4 shards. This module is platform-agnostic.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::{Error, Result};

/// A single feature in the dataset schema (e.g. a camera stream, the state vector, the action).
#[derive(Debug, Clone, Deserialize)]
pub struct Feature {
    /// `"video"`, `"image"`, `"float32"`, `"int64"`, ...
    pub dtype: String,
    /// Shape of one element (e.g. `[H, W, C]` for a frame, `[D]` for state).
    #[serde(default)]
    pub shape: Vec<usize>,
    /// Optional per-dimension names (e.g. joint names).
    #[serde(default)]
    pub names: Option<serde_json::Value>,
}

impl Feature {
    /// Whether this feature is a camera video stream stored as mp4.
    pub fn is_video(&self) -> bool {
        matches!(self.dtype.as_str(), "video" | "image")
    }
}

/// Parsed `meta/info.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Info {
    #[serde(default)]
    pub codebase_version: String,
    #[serde(default)]
    pub robot_type: Option<String>,
    pub fps: f64,
    pub total_episodes: usize,
    pub total_frames: usize,
    /// Number of episodes grouped per chunk directory.
    #[serde(default = "default_chunks_size")]
    pub chunks_size: usize,
    /// Template like `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`.
    pub data_path: String,
    /// Template like `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`.
    pub video_path: String,
    /// Feature schema keyed by feature name (e.g. `observation.images.top`).
    pub features: BTreeMap<String, Feature>,
}

fn default_chunks_size() -> usize {
    1000
}

impl Info {
    /// Load and parse `<root>/meta/info.json`.
    pub fn load(root: &Path) -> Result<Self> {
        let path = root.join("meta").join("info.json");
        let bytes = std::fs::read(&path)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| Error::Dataset(format!("parsing {}: {e}", path.display())))
    }

    /// Names of the camera (video) features, in schema order.
    pub fn camera_keys(&self) -> Vec<String> {
        self.features
            .iter()
            .filter(|(_, f)| f.is_video())
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Resolve a data (parquet) shard path relative to the dataset root.
    pub fn data_file_path(&self, chunk_index: usize, file_index: usize) -> String {
        render_path(&self.data_path, None, chunk_index, file_index)
    }

    /// Resolve a video (mp4) shard path for a camera, relative to the dataset root.
    pub fn video_file_path(
        &self,
        camera_key: &str,
        chunk_index: usize,
        file_index: usize,
    ) -> String {
        render_path(&self.video_path, Some(camera_key), chunk_index, file_index)
    }
}

/// Fill a LeRobot path template. Supports `{video_key}`, `{chunk_index:03d}`,
/// `{file_index:03d}` (and the same keys without a width specifier).
fn render_path(
    template: &str,
    video_key: Option<&str>,
    chunk_index: usize,
    file_index: usize,
) -> String {
    let mut out = template.to_string();
    if let Some(vk) = video_key {
        out = out.replace("{video_key}", vk);
    }
    out = replace_indexed(&out, "chunk_index", chunk_index);
    out = replace_indexed(&out, "file_index", file_index);
    out
}

/// Replace `{name:0Nd}` (zero-padded) and bare `{name}` occurrences with `value`.
fn replace_indexed(s: &str, name: &str, value: usize) -> String {
    let mut out = s.to_string();
    // Zero-padded forms like {chunk_index:03d}
    while let Some(start) = out.find(&format!("{{{name}:")) {
        let Some(rel_end) = out[start..].find('}') else {
            break;
        };
        let end = start + rel_end + 1;
        let spec = &out[start..end]; // e.g. {chunk_index:03d}
        let width = parse_zero_pad_width(spec).unwrap_or(0);
        let replacement = format!("{value:0width$}");
        out.replace_range(start..end, &replacement);
    }
    // Bare form {chunk_index}
    out.replace(&format!("{{{name}}}"), &value.to_string())
}

/// Extract N from a `{name:0Nd}` spec.
fn parse_zero_pad_width(spec: &str) -> Option<usize> {
    let colon = spec.find(':')?;
    let inner = &spec[colon + 1..spec.len() - 1]; // strip ':' .. '}'
    let digits: String = inner
        .trim_start_matches('0')
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    // For "03d": trim_start_matches('0') -> "3d"; take digits -> "3"
    digits.parse().ok().or(Some(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_zero_padded_paths() {
        let info = sample_info();
        assert_eq!(info.data_file_path(0, 7), "data/chunk-000/file-007.parquet");
        assert_eq!(
            info.video_file_path("observation.images.top", 12, 3),
            "videos/observation.images.top/chunk-012/file-003.mp4"
        );
    }

    #[test]
    fn identifies_camera_features() {
        let info = sample_info();
        let cams = info.camera_keys();
        assert_eq!(cams, vec!["observation.images.top".to_string()]);
    }

    fn sample_info() -> Info {
        serde_json::from_str(SAMPLE_JSON).unwrap()
    }

    const SAMPLE_JSON: &str = r#"{
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 100,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [14]}
        }
    }"#;
}
