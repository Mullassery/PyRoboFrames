//! Fuzzes `pyroboframes_core::rosbag::convert`, the rosbag2 SQLite (`.db3`)
//! -> Parquet entry point: it opens the input as a read-only SQLite database,
//! reads `topics` / `message_definitions` / `messages`, parses each topic's
//! `ros2msg` schema, and CDR-decodes message blobs. Feeding it arbitrary
//! bytes exercises SQLite's own file-format parsing (corrupted page headers,
//! truncated files, garbage b-tree structures) as well as our schema/CDR
//! decoding once a syntactically-valid-but-adversarial database is found.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(tmp) = tempfile::tempdir() else {
        return;
    };
    let input = tmp.path().join("fuzz.db3");
    if std::fs::write(&input, data).is_err() {
        return;
    }
    let out = tmp.path().join("out");

    let _ = pyroboframes_core::rosbag::convert(&input, &out);
});
