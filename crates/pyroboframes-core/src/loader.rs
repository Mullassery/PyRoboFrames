//! The core sample/batch builder for tabular features (state, action, …).
//!
//! Given a dataset, this resolves a global frame index to its episode and data-shard row (via
//! the [`EpisodeIndex`]) and reads the requested float features from the [`DataShard`]. It is
//! the functional, video-free half of the dataloader: it works today on any LeRobotDataset
//! v3.0. Video frames layer on top once a decode backend is available.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::data::DataShard;
use crate::dataset::Dataset;
use crate::episodes::EpisodeIndex;
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

/// Builds [`Sample`]s for tabular (non-video) float features. Holds opened data shards so
/// repeated access to the same shard doesn't reopen the parquet file.
pub struct TabularLoader {
    dataset: Arc<Dataset>,
    index: EpisodeIndex,
    /// (data_chunk, data_file) -> global frame index of the shard's first row.
    shard_start: HashMap<(usize, usize), usize>,
    features: Vec<String>,
    open: HashMap<(usize, usize), DataShard>,
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

        Ok(Self {
            dataset,
            index,
            shard_start,
            features,
            open: HashMap::new(),
        })
    }

    pub fn total_frames(&self) -> usize {
        self.index.total_frames()
    }

    pub fn features(&self) -> &[String] {
        &self.features
    }

    /// Build the sample for a global frame index.
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

        if !self.open.contains_key(&key) {
            let shard = self.dataset.data_shard(key.0, key.1)?;
            self.open.insert(key, shard);
        }
        let shard = &self.open[&key];

        let mut features = BTreeMap::new();
        for name in &self.features {
            features.insert(name.clone(), shard.feature_f32(name, row)?);
        }
        Ok(Sample {
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
}
