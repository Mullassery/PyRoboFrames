//! Optional dataset statistics (`meta/stats.json`) — per-feature `mean`/`std`/`min`/`max`,
//! used to normalize states/actions before training.
//!
//! The file is optional (and its exact layout varies across LeRobot versions), so loading is
//! tolerant: a missing file yields `Ok(None)`, and numeric arrays are flattened to a flat
//! `Vec<f64>` (so a `[3,1,1]` image-channel stat reads back as 3 values) — anything non-numeric
//! is skipped rather than erroring.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Map, Value};

use crate::{Error, Result};

/// Per-feature summary statistics. Each vector is empty if that field was absent.
#[derive(Debug, Clone, Default)]
pub struct FeatureStats {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    /// Number of samples the stats were computed over, if recorded.
    pub count: Option<u64>,
}

/// All per-feature statistics from `meta/stats.json`.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub features: BTreeMap<String, FeatureStats>,
}

impl Stats {
    /// Load `<root>/meta/stats.json` if present. Returns `Ok(None)` when the file is absent.
    pub fn load(root: &Path) -> Result<Option<Stats>> {
        let path = root.join("meta").join("stats.json");
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Dataset(format!("parsing {}: {e}", path.display())))?;

        // Some exports nest the per-feature map under a top-level "stats" key.
        let obj = value
            .get("stats")
            .and_then(Value::as_object)
            .or_else(|| value.as_object())
            .ok_or_else(|| Error::Dataset(format!("{}: expected a JSON object", path.display())))?;

        let mut features = BTreeMap::new();
        for (name, fs) in obj {
            if let Some(map) = fs.as_object() {
                features.insert(name.clone(), parse_feature(map));
            }
        }
        Ok(Some(Stats { features }))
    }

    /// Stats for one feature (e.g. `observation.state`), if present.
    pub fn get(&self, feature: &str) -> Option<&FeatureStats> {
        self.features.get(feature)
    }
}

fn parse_feature(map: &Map<String, Value>) -> FeatureStats {
    FeatureStats {
        mean: flat_field(map, "mean"),
        std: flat_field(map, "std"),
        min: flat_field(map, "min"),
        max: flat_field(map, "max"),
        count: map.get("count").and_then(first_u64),
    }
}

/// Collect a field's numeric values, flattening nested arrays; empty if missing/non-numeric.
fn flat_field(map: &Map<String, Value>, key: &str) -> Vec<f64> {
    let mut out = Vec::new();
    if let Some(v) = map.get(key) {
        flatten(v, &mut out);
    }
    out
}

fn flatten(v: &Value, out: &mut Vec<f64>) {
    match v {
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.push(f);
            }
        }
        Value::Array(a) => a.iter().for_each(|x| flatten(x, out)),
        _ => {}
    }
}

/// First integer found in a value (handles both `5` and `[5]`).
fn first_u64(v: &Value) -> Option<u64> {
    match v {
        Value::Number(n) => n.as_u64(),
        Value::Array(a) => a.iter().find_map(first_u64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_stats(json: &str) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("meta")).unwrap();
        fs::write(tmp.path().join("meta").join("stats.json"), json).unwrap();
        tmp
    }

    #[test]
    fn absent_file_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(Stats::load(tmp.path()).unwrap().is_none());
    }

    #[test]
    fn parses_flat_feature_stats() {
        let tmp = write_stats(
            r#"{
                "observation.state": {
                    "mean": [1.0, 2.0], "std": [0.5, 0.5],
                    "min": [0.0, 1.0], "max": [2.0, 3.0], "count": 100
                },
                "action": {"mean": [0.0], "std": [1.0], "min": [-1.0], "max": [1.0]}
            }"#,
        );
        let stats = Stats::load(tmp.path()).unwrap().unwrap();
        let s = stats.get("observation.state").unwrap();
        assert_eq!(s.mean, vec![1.0, 2.0]);
        assert_eq!(s.max, vec![2.0, 3.0]);
        assert_eq!(s.count, Some(100));
        assert_eq!(stats.get("action").unwrap().std, vec![1.0]);
        assert!(stats.get("missing").is_none());
    }

    #[test]
    fn flattens_nested_and_tolerates_extras() {
        // image-style nested stats + a "stats" wrapper + a non-numeric extra field
        let tmp = write_stats(
            r#"{"stats": {
                "observation.images.top": {
                    "mean": [[[0.1]], [[0.2]], [[0.3]]],
                    "note": "ignored"
                }
            }}"#,
        );
        let stats = Stats::load(tmp.path()).unwrap().unwrap();
        let s = stats.get("observation.images.top").unwrap();
        assert_eq!(s.mean, vec![0.1, 0.2, 0.3]);
        assert!(s.std.is_empty());
    }
}
