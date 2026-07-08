//! LeRobotDataset v3.0 reader (platform-agnostic).
//!
//! Phase 1a (this cut): open a dataset directory, parse `meta/info.json`, expose the schema
//! (cameras, fps, totals), and resolve shard paths. Phase 1b adds reading the per-episode
//! index (`meta/episodes/*.parquet`) and the frame-by-frame tabular shards (`data/*.parquet`)
//! to provide exact `(camera, file, timestamp)` frame location.

use std::path::{Path, PathBuf};

use crate::data::DataShard;
use crate::episodes::EpisodeIndex;
use crate::info::Info;
use crate::{Error, Result};

// SECURITY: Prevent DOS via excessively large video files
const MAX_VIDEO_FILE_SIZE: u64 = 2 * 1024 * 1024 * 1024; // 2GB per video file

/// An opened LeRobotDataset v3.0.
#[derive(Debug)]
pub struct Dataset {
    root: PathBuf,
    info: Info,
}

impl Dataset {
    /// Open a dataset rooted at `root` (the directory containing `meta/`, `data/`, `videos/`).
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        if !root.join("meta").join("info.json").is_file() {
            return Err(Error::Dataset(format!(
                "{} does not look like a LeRobotDataset v3.0 (missing meta/info.json)",
                root.display()
            )));
        }
        let info = Info::load(&root)?;
        Ok(Self { root, info })
    }

    pub fn info(&self) -> &Info {
        &self.info
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn fps(&self) -> f64 {
        self.info.fps
    }

    pub fn num_episodes(&self) -> usize {
        self.info.total_episodes
    }

    pub fn num_frames(&self) -> usize {
        self.info.total_frames
    }

    /// Camera (video) feature keys in schema order.
    pub fn cameras(&self) -> Vec<String> {
        self.info.camera_keys()
    }

    /// Absolute path to a data (parquet) shard.
    pub fn data_file(&self, chunk_index: usize, file_index: usize) -> PathBuf {
        self.root
            .join(self.info.data_file_path(chunk_index, file_index))
    }

    /// Absolute path to a camera's video (mp4) shard. Validates file size to prevent DOS attacks.
    pub fn video_file(
        &self,
        camera_key: &str,
        chunk_index: usize,
        file_index: usize,
    ) -> Result<PathBuf> {
        let path = self.root.join(
            self.info
                .video_file_path(camera_key, chunk_index, file_index),
        );

        if let Ok(metadata) = std::fs::metadata(&path) {
            if metadata.len() > MAX_VIDEO_FILE_SIZE {
                return Err(Error::Dataset(format!(
                    "Video file {} exceeds size limit: {} > {}",
                    path.display(),
                    metadata.len(),
                    MAX_VIDEO_FILE_SIZE
                )));
            }
        }

        Ok(path)
    }

    /// Timestamp (seconds) of frame `frame_in_episode` within an episode, from the frame rate.
    pub fn frame_timestamp(&self, frame_in_episode: usize) -> f64 {
        frame_in_episode as f64 / self.info.fps
    }

    /// Load the per-episode index (`meta/episodes/*.parquet`), enabling global frame -> decode
    /// location resolution. Reads from disk on each call.
    pub fn episodes(&self) -> Result<EpisodeIndex> {
        EpisodeIndex::load(&self.root, &self.info)
    }

    /// Open a tabular data shard (`data/chunk-XXX/file-YYY.parquet`) holding the state/action
    /// rows for the episodes assigned to it.
    pub fn data_shard(&self, chunk_index: usize, file_index: usize) -> Result<DataShard> {
        DataShard::open(&self.data_file(chunk_index, file_index))
    }

    /// Validate dataset metadata integrity (contiguous frame ranges, lengths, timestamps,
    /// totals). See [`crate::validate`].
    pub fn validate(&self) -> Result<crate::ValidationReport> {
        crate::validate::validate(self)
    }

    /// Per-feature statistics from `meta/stats.json` (mean/std/min/max for normalization).
    /// Returns `Ok(None)` when the dataset has no stats file.
    pub fn stats(&self) -> Result<Option<crate::stats::Stats>> {
        crate::stats::Stats::load(&self.root)
    }

    /// Deterministic train/validation split over episode indices (split by episode, not frame,
    /// to avoid temporal leakage). See [`crate::split`].
    pub fn train_val_split(&self, val_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
        crate::split::split_episodes(self.num_episodes(), val_fraction, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_fixture(dir: &Path) {
        fs::create_dir_all(dir.join("meta")).unwrap();
        fs::write(dir.join("meta").join("info.json"), FIXTURE_INFO).unwrap();
    }

    #[test]
    fn opens_and_reads_schema() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path());

        let ds = Dataset::open(tmp.path()).unwrap();
        assert_eq!(ds.fps(), 30.0);
        assert_eq!(ds.num_episodes(), 2);
        assert_eq!(ds.num_frames(), 100);
        assert_eq!(ds.cameras(), vec!["observation.images.top".to_string()]);
    }

    #[test]
    fn resolves_absolute_shard_paths() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path());
        let ds = Dataset::open(tmp.path()).unwrap();

        assert_eq!(
            ds.data_file(0, 0),
            tmp.path().join("data/chunk-000/file-000.parquet")
        );
        assert_eq!(
            ds.video_file("observation.images.top", 0, 0),
            tmp.path()
                .join("videos/observation.images.top/chunk-000/file-000.mp4")
        );
    }

    #[test]
    fn frame_timestamps_follow_fps() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path());
        let ds = Dataset::open(tmp.path()).unwrap();
        assert_eq!(ds.frame_timestamp(0), 0.0);
        assert_eq!(ds.frame_timestamp(30), 1.0);
    }

    #[test]
    fn rejects_non_dataset_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(Dataset::open(tmp.path()).is_err());
    }

    const FIXTURE_INFO: &str = r#"{
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
