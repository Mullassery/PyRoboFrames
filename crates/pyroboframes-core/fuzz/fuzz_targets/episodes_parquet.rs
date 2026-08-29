//! Fuzzes `pyroboframes_core::episodes::EpisodeIndex::load`, which reads
//! every `meta/episodes/**/*.parquet` shard of a LeRobotDataset v3.0 and
//! builds the per-episode index used to resolve a global frame to its data
//! shard row and per-camera video location. The `info.json` is held fixed
//! (one camera, matching the schema `EpisodeIndex::load` expects) so the
//! fuzzer's bytes land on the actual untrusted input: the episodes Parquet
//! file itself -- corrupted footers, wrong column types (int_col/f64_col
//! downcasts), missing columns, and boundary row values.
#![no_main]

use libfuzzer_sys::fuzz_target;
use pyroboframes_core::episodes::EpisodeIndex;
use pyroboframes_core::info::Info;

const INFO_JSON: &str = r#"{
    "codebase_version": "v3.0",
    "fps": 30,
    "total_episodes": 1,
    "total_frames": 1,
    "chunks_size": 1000,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.state": {"dtype": "float32", "shape": [14]}
    }
}"#;

fuzz_target!(|data: &[u8]| {
    let Ok(tmp) = tempfile::tempdir() else {
        return;
    };
    let root = tmp.path();
    let meta = root.join("meta");
    if std::fs::create_dir_all(&meta).is_err() {
        return;
    }
    if std::fs::write(meta.join("info.json"), INFO_JSON).is_err() {
        return;
    }
    let ep_dir = meta.join("episodes").join("chunk-000");
    if std::fs::create_dir_all(&ep_dir).is_err() {
        return;
    }
    if std::fs::write(ep_dir.join("file-000.parquet"), data).is_err() {
        return;
    }

    let Ok(info) = Info::load(root) else {
        return;
    };
    if let Ok(idx) = EpisodeIndex::load(root, &info) {
        let total = idx.total_frames();
        let _ = idx.locate(0);
        let _ = idx.locate(total);
        let _ = idx.locate(usize::MAX);
    }
});
