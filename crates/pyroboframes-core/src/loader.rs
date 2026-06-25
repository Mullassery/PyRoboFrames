//! The core sample/batch builder for tabular features (state, action, …).
//!
//! Given a dataset, this resolves a global frame index to its episode and data-shard row (via
//! the [`EpisodeIndex`]) and reads the requested float features from the [`DataShard`]. It is
//! the functional, video-free half of the dataloader: it works today on any LeRobotDataset
//! v3.0. Video frames layer on top once a decode backend is available.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::data::DataShard;
use crate::dataset::Dataset;
use crate::decode::{Decoder, Frame, FrameCache};
use crate::episodes::{EpisodeIndex, FrameLocation};
use crate::window::{resolve_offsets, WindowSpec};
use crate::{Error, Result};

/// One frame's tabular features plus its provenance.
#[derive(Debug, Clone)]
pub struct Sample {
    pub global_index: usize,
    pub episode_index: usize,
    pub frame_in_episode: usize,
    /// Feature name -> values (e.g. `observation.state` -> joint positions).
    pub features: BTreeMap<String, Vec<f32>>,
}

/// A sample with temporal context: each feature carries one value vector per requested delta
/// (shape `[num_deltas][feature_dim]`), assembled per [`WindowSpec`].
#[derive(Debug, Clone)]
pub struct WindowedSample {
    pub global_index: usize,
    pub episode_index: usize,
    pub frame_in_episode: usize,
    pub features: BTreeMap<String, Vec<Vec<f32>>>,
}

/// Builds [`Sample`]s for tabular (non-video) float features. Holds opened data shards so
/// repeated access to the same shard doesn't reopen the parquet file.
pub struct TabularLoader {
    dataset: Arc<Dataset>,
    index: EpisodeIndex,
    fps: f64,
    /// (data_chunk, data_file) -> global frame index of the shard's first row.
    shard_start: HashMap<(usize, usize), usize>,
    features: Vec<String>,
    open: HashMap<(usize, usize), DataShard>,
    /// Feature -> (mean, std) for optional `(x - mean) / std` normalization. Empty = off.
    norm: HashMap<String, (Vec<f32>, Vec<f32>)>,
}

impl TabularLoader {
    /// Open over a dataset, loading its episode index. Features default to every non-video
    /// `float*` feature (e.g. `observation.state`, `action`).
    pub fn open(dataset: Arc<Dataset>) -> Result<Self> {
        Self::with_features(dataset, None)
    }

    /// Like [`open`], but with an explicit feature list.
    pub fn with_features(dataset: Arc<Dataset>, features: Option<Vec<String>>) -> Result<Self> {
        let index = dataset.episodes()?;

        // A data shard concatenates its episodes in global-frame order; the shard's first global
        // row is the smallest `from_index` among the episodes assigned to it.
        let mut shard_start: HashMap<(usize, usize), usize> = HashMap::new();
        for ep in index.episodes() {
            let key = (ep.data_chunk_index, ep.data_file_index);
            let entry = shard_start.entry(key).or_insert(usize::MAX);
            *entry = (*entry).min(ep.from_index);
        }

        let features = features.unwrap_or_else(|| {
            dataset
                .info()
                .features
                .iter()
                .filter(|(_, f)| !f.is_video() && f.dtype.starts_with("float"))
                .map(|(k, _)| k.clone())
                .collect()
        });

        let fps = dataset.fps();
        Ok(Self {
            dataset,
            index,
            fps,
            shard_start,
            features,
            open: HashMap::new(),
            norm: HashMap::new(),
        })
    }

    /// Enable `(x - mean) / std` normalization for the given features, using `meta/stats.json`.
    /// Errors if the dataset has no stats or a requested feature lacks (valid) stats. A zero std
    /// is treated as 1 (leaves that component mean-centered) to avoid division by zero.
    pub fn enable_normalization(&mut self, features: &[String]) -> Result<()> {
        let stats = self.dataset.stats()?.ok_or_else(|| {
            Error::Dataset("normalization requested but dataset has no meta/stats.json".into())
        })?;
        for name in features {
            let fs = stats.get(name).ok_or_else(|| {
                Error::Dataset(format!("no stats for feature `{name}` (needed for normalization)"))
            })?;
            if fs.mean.is_empty() || fs.mean.len() != fs.std.len() {
                return Err(Error::Dataset(format!(
                    "stats for `{name}` have missing or mismatched mean/std"
                )));
            }
            let mean = fs.mean.iter().map(|&x| x as f32).collect();
            let std = fs.std.iter().map(|&x| x as f32).collect();
            self.norm.insert(name.clone(), (mean, std));
        }
        Ok(())
    }

    /// Apply normalization in place if `name` is configured (no-op otherwise).
    fn apply_norm(&self, name: &str, v: &mut [f32]) {
        if let Some((mean, std)) = self.norm.get(name) {
            for (i, x) in v.iter_mut().enumerate() {
                let m = mean.get(i).copied().unwrap_or(0.0);
                let s = std.get(i).copied().unwrap_or(1.0);
                let s = if s == 0.0 { 1.0 } else { s };
                *x = (*x - m) / s;
            }
        }
    }

    pub fn total_frames(&self) -> usize {
        self.index.total_frames()
    }

    pub fn features(&self) -> &[String] {
        &self.features
    }

    /// Global frame indices belonging to the given episode indices, ascending. Lets a loader
    /// iterate just a train (or val) split — see [`Dataset::train_val_split`](crate::dataset::Dataset::train_val_split).
    /// Unknown episode indices are ignored.
    pub fn frame_indices_for_episodes(&self, episodes: &[usize]) -> Vec<usize> {
        let wanted: HashSet<usize> = episodes.iter().copied().collect();
        let mut out = Vec::new();
        for ep in self.index.episodes() {
            if wanted.contains(&ep.episode_index) {
                out.extend(ep.from_index..ep.to_index);
            }
        }
        out.sort_unstable();
        out
    }

    /// Ensure a data shard is open (cached) so subsequent reads don't reopen the file.
    fn ensure_open(&mut self, key: (usize, usize)) -> Result<()> {
        if !self.open.contains_key(&key) {
            let shard = self.dataset.data_shard(key.0, key.1)?;
            self.open.insert(key, shard);
        }
        Ok(())
    }

    /// Build the sample (current frame only) for a global frame index.
    pub fn sample(&mut self, global_index: usize) -> Result<Sample> {
        let loc = self
            .index
            .locate(global_index)
            .ok_or_else(|| Error::Dataset(format!("frame {global_index} out of range")))?;
        let key = (loc.data_chunk_index, loc.data_file_index);
        let start = *self
            .shard_start
            .get(&key)
            .ok_or_else(|| Error::Dataset(format!("no data shard for frame {global_index}")))?;
        let row = global_index - start;

        self.ensure_open(key)?;
        let shard = &self.open[&key];
        let mut features = BTreeMap::new();
        for name in &self.features {
            let mut v = shard.feature_f32(name, row)?;
            self.apply_norm(name, &mut v);
            features.insert(name.clone(), v);
        }
        Ok(Sample {
            global_index,
            episode_index: loc.episode_index,
            frame_in_episode: loc.frame_in_episode,
            features,
        })
    }

    /// Build a windowed sample: each feature gets one value vector per delta in `spec`
    /// (the current frame if a feature has no entry). All offsets stay within the episode.
    pub fn windowed_sample(
        &mut self,
        global_index: usize,
        spec: &WindowSpec,
    ) -> Result<WindowedSample> {
        let loc = self
            .index
            .locate(global_index)
            .ok_or_else(|| Error::Dataset(format!("frame {global_index} out of range")))?;
        let key = (loc.data_chunk_index, loc.data_file_index);
        let start = *self
            .shard_start
            .get(&key)
            .ok_or_else(|| Error::Dataset(format!("no data shard for frame {global_index}")))?;
        let episode_from_global = loc.global_index - loc.frame_in_episode;

        self.ensure_open(key)?;
        let shard = &self.open[&key];
        let mut features = BTreeMap::new();
        for name in &self.features {
            let offsets = resolve_offsets(
                spec.deltas_for(name),
                loc.frame_in_episode,
                loc.episode_len,
                self.fps,
                spec.tolerance_s,
            )?;
            let mut steps = Vec::with_capacity(offsets.len());
            for off in offsets {
                let row = (episode_from_global + off) - start;
                let mut v = shard.feature_f32(name, row)?;
                self.apply_norm(name, &mut v);
                steps.push(v);
            }
            features.insert(name.clone(), steps);
        }
        Ok(WindowedSample {
            global_index,
            episode_index: loc.episode_index,
            frame_in_episode: loc.frame_in_episode,
            features,
        })
    }

    /// Build samples for a batch of global frame indices.
    pub fn batch(&mut self, indices: &[usize]) -> Result<Vec<Sample>> {
        indices.iter().map(|&i| self.sample(i)).collect()
    }

    /// The underlying dataset (for video-path resolution).
    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    /// Resolve a global frame to its decode location.
    pub fn locate(&self, global_index: usize) -> Option<FrameLocation> {
        self.index.locate(global_index)
    }

    /// Decode the requested cameras' frames for a global frame index, via `decoder` + `cache`.
    pub fn frames_for(
        &self,
        global_index: usize,
        cameras: &[String],
        decoder: &mut dyn Decoder,
        cache: &mut FrameCache,
    ) -> Result<Vec<(String, Frame)>> {
        let loc = self
            .index
            .locate(global_index)
            .ok_or_else(|| Error::Dataset(format!("frame {global_index} out of range")))?;
        let mut out = Vec::with_capacity(cameras.len());
        for cam in cameras {
            let &(chunk, file, ts) = loc.videos.get(cam).ok_or_else(|| {
                Error::Dataset(format!("camera `{cam}` not found for frame {global_index}"))
            })?;
            let path = self.dataset.video_file(cam, chunk, file);
            out.push((cam.clone(), cache.get_or_decode(decoder, cam, &path, ts)?));
        }
        Ok(out)
    }
}

/// Decode each camera's frame for a resolved frame location, going through `cache` (so repeated
/// frames aren't re-decoded). This is the video half of the pipeline: it resolves the per-camera
/// video shard path + timestamp and hands them to the `Decoder`. Works with any backend —
/// VideoToolbox / FFmpeg once implemented, or a test decoder today.
pub fn decode_frames(
    dataset: &Dataset,
    loc: &FrameLocation,
    decoder: &mut dyn Decoder,
    cache: &mut FrameCache,
) -> Result<BTreeMap<String, Frame>> {
    let mut frames = BTreeMap::new();
    for (camera, &(chunk, file, timestamp)) in &loc.videos {
        let path = dataset.video_file(camera, chunk, file);
        let frame = cache.get_or_decode(decoder, camera, &path, timestamp)?;
        frames.insert(camera.clone(), frame);
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, FixedSizeListBuilder, Float32Builder, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::fs::{self, File};
    use std::path::Path;
    use std::sync::Arc;

    const CAM: &str = "observation.images.top";

    /// Write a complete two-episode dataset (info + episodes index + one data shard of 100 rows
    /// where `observation.state[row] = [row, 0, 0]` and `action[row] = [row, row, row]`).
    fn write_dataset(root: &Path) {
        fs::create_dir_all(root.join("meta")).unwrap();
        fs::write(
            root.join("meta/info.json"),
            format!(
                r#"{{"codebase_version":"v3.0","fps":30,"total_episodes":2,"total_frames":100,
                "chunks_size":1000,
                "data_path":"data/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.parquet",
                "video_path":"videos/{{video_key}}/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.mp4",
                "features":{{"{CAM}":{{"dtype":"video","shape":[480,640,3]}},
                "observation.state":{{"dtype":"float32","shape":[3]}},
                "action":{{"dtype":"float32","shape":[3]}}}}}}"#
            ),
        )
        .unwrap();

        write_episodes(root);
        write_data(root);
    }

    fn write_episodes(root: &Path) {
        let dir = root.join("meta/episodes/chunk-000");
        fs::create_dir_all(&dir).unwrap();
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
        let mut w = ArrowWriter::try_new(
            File::create(dir.join("file-000.parquet")).unwrap(),
            schema,
            None,
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    fn write_data(root: &Path) {
        let dir = root.join("data/chunk-000");
        fs::create_dir_all(&dir).unwrap();

        let mut state = FixedSizeListBuilder::new(Float32Builder::new(), 3);
        let mut action = FixedSizeListBuilder::new(Float32Builder::new(), 3);
        for i in 0..100i64 {
            let v = i as f32;
            state.values().append_value(v);
            state.values().append_value(0.0);
            state.values().append_value(0.0);
            state.append(true);
            for _ in 0..3 {
                action.values().append_value(v);
            }
            action.append(true);
        }
        let state = state.finish();
        let action = action.finish();
        let schema = Arc::new(Schema::new(vec![
            Field::new("observation.state", state.data_type().clone(), false),
            Field::new("action", action.data_type().clone(), false),
        ]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(state), Arc::new(action)]).unwrap();
        let mut w = ArrowWriter::try_new(
            File::create(dir.join("file-000.parquet")).unwrap(),
            schema,
            None,
        )
        .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    #[test]
    fn builds_samples_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        write_dataset(tmp.path());
        let ds = Dataset::open(tmp.path()).unwrap();
        let mut loader = TabularLoader::open(Arc::new(ds)).unwrap();

        assert_eq!(loader.total_frames(), 100);
        // default features = non-video floats, sorted by BTreeMap key
        assert_eq!(
            loader.features(),
            &["action".to_string(), "observation.state".to_string()]
        );

        let s = loader.sample(60).unwrap();
        assert_eq!(s.episode_index, 1);
        assert_eq!(s.frame_in_episode, 10);
        assert_eq!(s.features["observation.state"], vec![60.0, 0.0, 0.0]);
        assert_eq!(s.features["action"], vec![60.0, 60.0, 60.0]);
    }

    #[test]
    fn batches_multiple_frames() {
        let tmp = tempfile::tempdir().unwrap();
        write_dataset(tmp.path());
        let ds = Dataset::open(tmp.path()).unwrap();
        let mut loader = TabularLoader::open(Arc::new(ds)).unwrap();

        let batch = loader.batch(&[0, 49, 50, 99]).unwrap();
        assert_eq!(batch.len(), 4);
        assert_eq!(batch[0].features["observation.state"], vec![0.0, 0.0, 0.0]);
        assert_eq!(batch[1].episode_index, 0); // frame 49 is last of episode 0
        assert_eq!(batch[2].episode_index, 1); // frame 50 is first of episode 1
        assert_eq!(batch[3].features["action"], vec![99.0, 99.0, 99.0]);
        assert!(loader.sample(100).is_err()); // out of range
    }

    #[test]
    fn frame_indices_for_episodes_selects_subset() {
        let tmp = tempfile::tempdir().unwrap();
        write_dataset(tmp.path());
        let ds = Dataset::open(tmp.path()).unwrap();
        let loader = TabularLoader::open(Arc::new(ds)).unwrap();

        // Episode 1 owns global frames [50, 100).
        assert_eq!(
            loader.frame_indices_for_episodes(&[1]),
            (50..100).collect::<Vec<_>>()
        );
        // Both episodes -> all frames; unknown indices ignored.
        assert_eq!(
            loader.frame_indices_for_episodes(&[0, 1, 99]),
            (0..100).collect::<Vec<_>>()
        );
        assert!(loader.frame_indices_for_episodes(&[]).is_empty());
    }

    #[test]
    fn normalization_applies_mean_std() {
        let tmp = tempfile::tempdir().unwrap();
        write_dataset(tmp.path());
        fs::write(
            tmp.path().join("meta/stats.json"),
            r#"{"observation.state":{"mean":[10.0,0.0,0.0],"std":[2.0,1.0,1.0]},
                "action":{"mean":[0.0,0.0,0.0],"std":[1.0,1.0,1.0]}}"#,
        )
        .unwrap();
        let ds = Dataset::open(tmp.path()).unwrap();
        let mut loader = TabularLoader::open(Arc::new(ds)).unwrap();
        loader
            .enable_normalization(&["observation.state".to_string()])
            .unwrap();

        // frame 60: state [60,0,0] -> ((60-10)/2, 0, 0) = [25, 0, 0]; action untouched.
        let s = loader.sample(60).unwrap();
        assert_eq!(s.features["observation.state"], vec![25.0, 0.0, 0.0]);
        assert_eq!(s.features["action"], vec![60.0, 60.0, 60.0]);
    }

    #[test]
    fn normalization_requires_stats() {
        let tmp = tempfile::tempdir().unwrap();
        write_dataset(tmp.path()); // no stats.json written
        let ds = Dataset::open(tmp.path()).unwrap();
        let mut loader = TabularLoader::open(Arc::new(ds)).unwrap();
        assert!(loader
            .enable_normalization(&["observation.state".to_string()])
            .is_err());
    }

    use crate::decode::{FrameBuffer, FrameCache};

    struct MockDecoder;
    impl Decoder for MockDecoder {
        fn decode(&mut self, camera: &str, _file: &Path, timestamp: f64) -> Result<Frame> {
            Ok(Frame {
                width: 1,
                height: 1,
                camera: camera.to_string(),
                timestamp,
                pixels: FrameBuffer::Owned {
                    data: Arc::new(vec![0, 0, 0]),
                    channels: 3,
                },
            })
        }
    }

    fn window_spec(deltas: Vec<f64>) -> WindowSpec {
        let mut spec = WindowSpec {
            tolerance_s: 1e-3,
            ..Default::default()
        };
        spec.delta_timestamps
            .insert("observation.state".into(), deltas);
        spec
    }

    #[test]
    fn windowed_sample_gathers_temporal_context() {
        let tmp = tempfile::tempdir().unwrap();
        write_dataset(tmp.path());
        let ds = Dataset::open(tmp.path()).unwrap();
        let mut loader = TabularLoader::open(Arc::new(ds)).unwrap();

        // frame 10 of episode 0, deltas [-0.1, 0.0] -> history at frame 7 and current frame 10
        let w = loader
            .windowed_sample(10, &window_spec(vec![-0.1, 0.0]))
            .unwrap();
        let state = &w.features["observation.state"];
        assert_eq!(state.len(), 2);
        assert_eq!(state[0], vec![7.0, 0.0, 0.0]);
        assert_eq!(state[1], vec![10.0, 0.0, 0.0]);
        // `action` has no delta entry -> current frame only
        assert_eq!(w.features["action"], vec![vec![10.0, 10.0, 10.0]]);
    }

    #[test]
    fn windowed_sample_clamps_at_episode_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        write_dataset(tmp.path());
        let ds = Dataset::open(tmp.path()).unwrap();
        let mut loader = TabularLoader::open(Arc::new(ds)).unwrap();

        // global 50 = first frame of episode 1; history clamps to that frame (edge repeat)
        let w = loader
            .windowed_sample(50, &window_spec(vec![-0.1, 0.0]))
            .unwrap();
        let state = &w.features["observation.state"];
        assert_eq!(state[0], vec![50.0, 0.0, 0.0]);
        assert_eq!(state[1], vec![50.0, 0.0, 0.0]);
    }

    #[test]
    fn decode_frames_resolves_each_camera() {
        let tmp = tempfile::tempdir().unwrap();
        write_dataset(tmp.path());
        let ds = Dataset::open(tmp.path()).unwrap();
        let idx = ds.episodes().unwrap();
        let loc = idx.locate(60).unwrap();

        let mut decoder = MockDecoder;
        let mut cache = FrameCache::new(8);
        let frames = decode_frames(&ds, &loc, &mut decoder, &mut cache).unwrap();

        assert_eq!(frames.len(), 1);
        let f = &frames[CAM];
        assert_eq!(f.camera, CAM);
        // timestamp = episode-1 offset (50/30) + 10 frames at 30 fps
        assert!((f.timestamp - (50.0 / 30.0 + 10.0 / 30.0)).abs() < 1e-9);
    }
}
