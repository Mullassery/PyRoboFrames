//! The per-episode index (`meta/episodes/*.parquet`) of a LeRobotDataset v3.0.
//!
//! Each episode record says how long the episode is, where its global frame range falls, which
//! `data/` shard holds its tabular rows, and — per camera — which `videos/` shard holds its
//! frames and at what timestamp offset within that (multi-episode) mp4. This is what lets us
//! turn a global frame index into a concrete `(camera, video file, timestamp)` to seek + decode.
//!
//! Assumed v3.0 column names (slash-delimited, matching LeRobot's metadata-in-parquet layout):
//! `episode_index`, `length`, `dataset_from_index`, `dataset_to_index`, `data/chunk_index`,
//! `data/file_index`, and per camera `videos/<key>/{chunk_index,file_index,from_timestamp,
//! to_timestamp}`.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::info::Info;
use crate::{Error, Result};

/// Where a camera's frames for an episode live, and the episode's time offset in that shard.
#[derive(Debug, Clone)]
pub struct VideoLocation {
    pub chunk_index: usize,
    pub file_index: usize,
    pub from_timestamp: f64,
    pub to_timestamp: f64,
}

/// One episode's metadata.
#[derive(Debug, Clone)]
pub struct EpisodeRecord {
    pub episode_index: usize,
    pub length: usize,
    /// Global frame range `[from_index, to_index)`.
    pub from_index: usize,
    pub to_index: usize,
    pub data_chunk_index: usize,
    pub data_file_index: usize,
    /// Per-camera video location.
    pub videos: BTreeMap<String, VideoLocation>,
}

/// A fully-resolved location for a single global frame.
#[derive(Debug, Clone)]
pub struct FrameLocation {
    pub global_index: usize,
    pub episode_index: usize,
    pub frame_in_episode: usize,
    pub data_chunk_index: usize,
    pub data_file_index: usize,
    /// camera key -> (chunk_index, file_index, timestamp_seconds within that mp4 shard).
    pub videos: BTreeMap<String, (usize, usize, f64)>,
}

/// The episode index for a dataset, sorted by global start frame.
#[derive(Debug, Default)]
pub struct EpisodeIndex {
    episodes: Vec<EpisodeRecord>,
    fps: f64,
}

impl EpisodeIndex {
    /// Load and parse every `meta/episodes/**/*.parquet` shard.
    pub fn load(root: &Path, info: &Info) -> Result<Self> {
        let dir = root.join("meta").join("episodes");
        let mut files = Vec::new();
        collect_parquet(&dir, &mut files)?;
        files.sort();

        let cameras = info.camera_keys();
        let mut episodes = Vec::new();
        for f in &files {
            read_episode_file(f, &cameras, &mut episodes)?;
        }
        episodes.sort_by_key(|e| e.from_index);
        Ok(Self {
            episodes,
            fps: info.fps,
        })
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    pub fn episodes(&self) -> &[EpisodeRecord] {
        &self.episodes
    }

    /// Total frames across all episodes (the global frame count).
    pub fn total_frames(&self) -> usize {
        self.episodes.last().map(|e| e.to_index).unwrap_or(0)
    }

    /// Resolve a global frame index to a concrete decode location, or `None` if out of range.
    pub fn locate(&self, global_index: usize) -> Option<FrameLocation> {
        let idx = self
            .episodes
            .binary_search_by(|e| {
                use std::cmp::Ordering::*;
                if global_index < e.from_index {
                    Greater
                } else if global_index >= e.to_index {
                    Less
                } else {
                    Equal
                }
            })
            .ok()?;
        let ep = &self.episodes[idx];
        let frame_in_episode = global_index - ep.from_index;
        let videos = ep
            .videos
            .iter()
            .map(|(cam, v)| {
                let ts = v.from_timestamp + frame_in_episode as f64 / self.fps;
                (cam.clone(), (v.chunk_index, v.file_index, ts))
            })
            .collect();
        Some(FrameLocation {
            global_index,
            episode_index: ep.episode_index,
            frame_in_episode,
            data_chunk_index: ep.data_chunk_index,
            data_file_index: ep.data_file_index,
            videos,
        })
    }
}

/// Recursively collect `*.parquet` files under `dir`.
fn collect_parquet(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.exists() {
        return Err(Error::Dataset(format!(
            "missing episode index directory {}",
            dir.display()
        )));
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_parquet(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "parquet") {
            out.push(path);
        }
    }
    Ok(())
}

fn read_batches(path: &Path) -> Result<Vec<RecordBatch>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::Dataset(format!("{}: {e}", path.display())))?
        .build()
        .map_err(|e| Error::Dataset(format!("{}: {e}", path.display())))?;
    reader
        .map(|b| b.map_err(|e| Error::Dataset(e.to_string())))
        .collect()
}

fn int_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| Error::Dataset(format!("episodes parquet missing column `{name}`")))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| Error::Dataset(format!("episodes column `{name}` is not Int64")))
}

fn f64_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    batch
        .column_by_name(name)
        .ok_or_else(|| Error::Dataset(format!("episodes parquet missing column `{name}`")))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| Error::Dataset(format!("episodes column `{name}` is not Float64")))
}

fn read_episode_file(path: &Path, cameras: &[String], out: &mut Vec<EpisodeRecord>) -> Result<()> {
    for batch in read_batches(path)? {
        let episode_index = int_col(&batch, "episode_index")?;
        let length = int_col(&batch, "length")?;
        let from_index = int_col(&batch, "dataset_from_index")?;
        let to_index = int_col(&batch, "dataset_to_index")?;
        let data_chunk = int_col(&batch, "data/chunk_index")?;
        let data_file = int_col(&batch, "data/file_index")?;

        // Resolve per-camera column arrays once per batch.
        type CamCols<'a> = (
            String,
            &'a Int64Array,
            &'a Int64Array,
            &'a Float64Array,
            &'a Float64Array,
        );
        let cam_cols: Vec<CamCols> = cameras
            .iter()
            .map(|cam| {
                Ok((
                    cam.clone(),
                    int_col(&batch, &format!("videos/{cam}/chunk_index"))?,
                    int_col(&batch, &format!("videos/{cam}/file_index"))?,
                    f64_col(&batch, &format!("videos/{cam}/from_timestamp"))?,
                    f64_col(&batch, &format!("videos/{cam}/to_timestamp"))?,
                ))
            })
            .collect::<Result<_>>()?;

        for row in 0..batch.num_rows() {
            let mut videos = BTreeMap::new();
            for (cam, chunk, file, from_ts, to_ts) in &cam_cols {
                videos.insert(
                    cam.clone(),
                    VideoLocation {
                        chunk_index: chunk.value(row) as usize,
                        file_index: file.value(row) as usize,
                        from_timestamp: from_ts.value(row),
                        to_timestamp: to_ts.value(row),
                    },
                );
            }
            out.push(EpisodeRecord {
                episode_index: episode_index.value(row) as usize,
                length: length.value(row) as usize,
                from_index: from_index.value(row) as usize,
                to_index: to_index.value(row) as usize,
                data_chunk_index: data_chunk.value(row) as usize,
                data_file_index: data_file.value(row) as usize,
                videos,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    const CAM: &str = "observation.images.top";

    fn write_fixture(root: &Path) {
        // info.json (fps 30, one camera)
        std::fs::create_dir_all(root.join("meta")).unwrap();
        std::fs::write(
            root.join("meta/info.json"),
            format!(
                r#"{{"codebase_version":"v3.0","fps":30,"total_episodes":2,"total_frames":100,
                "chunks_size":1000,
                "data_path":"data/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.parquet",
                "video_path":"videos/{{video_key}}/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.mp4",
                "features":{{"{CAM}":{{"dtype":"video","shape":[480,640,3]}},
                "observation.state":{{"dtype":"float32","shape":[14]}},
                "action":{{"dtype":"float32","shape":[14]}}}}}}"#
            ),
        )
        .unwrap();

        // episodes parquet: two episodes, each length 50, global ranges [0,50) and [50,100).
        let dir = root.join("meta/episodes/chunk-000");
        std::fs::create_dir_all(&dir).unwrap();
        let f = |n: &str, t: DataType| Field::new(n, t, false);
        let schema = Arc::new(Schema::new(vec![
            f("episode_index", DataType::Int64),
            f("length", DataType::Int64),
            f("dataset_from_index", DataType::Int64),
            f("dataset_to_index", DataType::Int64),
            f("data/chunk_index", DataType::Int64),
            f("data/file_index", DataType::Int64),
            f(&format!("videos/{CAM}/chunk_index"), DataType::Int64),
            f(&format!("videos/{CAM}/file_index"), DataType::Int64),
            f(&format!("videos/{CAM}/from_timestamp"), DataType::Float64),
            f(&format!("videos/{CAM}/to_timestamp"), DataType::Float64),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![0i64, 1])),
                Arc::new(Int64Array::from(vec![50i64, 50])),
                Arc::new(Int64Array::from(vec![0i64, 50])),
                Arc::new(Int64Array::from(vec![50i64, 100])),
                Arc::new(Int64Array::from(vec![0i64, 0])),
                Arc::new(Int64Array::from(vec![0i64, 0])),
                Arc::new(Int64Array::from(vec![0i64, 0])),
                Arc::new(Int64Array::from(vec![0i64, 0])),
                Arc::new(Float64Array::from(vec![0.0f64, 50.0 / 30.0])),
                Arc::new(Float64Array::from(vec![50.0 / 30.0, 100.0 / 30.0])),
            ],
        )
        .unwrap();
        let file = File::create(dir.join("file-000.parquet")).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    #[test]
    fn loads_and_locates_frames() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path());
        let info = Info::load(tmp.path()).unwrap();
        let idx = EpisodeIndex::load(tmp.path(), &info).unwrap();

        assert_eq!(idx.len(), 2);
        assert_eq!(idx.total_frames(), 100);

        // Frame 60 is the 11th frame (index 10) of episode 1.
        let loc = idx.locate(60).unwrap();
        assert_eq!(loc.episode_index, 1);
        assert_eq!(loc.frame_in_episode, 10);
        let (chunk, file, ts) = &loc.videos[CAM];
        assert_eq!((*chunk, *file), (0, 0));
        // timestamp = episode-1 offset (50/30) + 10 frames at 30 fps.
        assert!((ts - (50.0 / 30.0 + 10.0 / 30.0)).abs() < 1e-9);
    }

    #[test]
    fn out_of_range_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        write_fixture(tmp.path());
        let info = Info::load(tmp.path()).unwrap();
        let idx = EpisodeIndex::load(tmp.path(), &info).unwrap();
        assert!(idx.locate(100).is_none());
        assert!(idx.locate(1000).is_none());
    }
}
