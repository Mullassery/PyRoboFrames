//! Fuzzes `pyroboframes_core::data::DataShard::open` and `feature_f32`, the
//! entry point that memory-maps and parses a `data/*.parquet` tabular shard
//! (the non-video per-frame features of a LeRobotDataset v3.0). Malformed
//! Parquet footers/schemas/page data should fail to open cleanly; once open,
//! `feature_f32` exercises the column-lookup + FixedSizeList/List downcast +
//! row-locate logic against whatever columns/rows the fuzzer produced,
//! including out-of-range rows and unknown column names.
#![no_main]

use libfuzzer_sys::fuzz_target;
use pyroboframes_core::data::DataShard;

fuzz_target!(|data: &[u8]| {
    let Ok(tmp) = tempfile::tempdir() else {
        return;
    };
    let path = tmp.path().join("fuzz.parquet");
    if std::fs::write(&path, data).is_err() {
        return;
    }

    if let Ok(shard) = DataShard::open(&path) {
        let rows = shard.num_rows();
        // Plausible real column names plus one bogus name, at row 0, the last
        // row, and a deliberately out-of-range row.
        for col in ["observation.state", "action", "x", ""] {
            let _ = shard.feature_f32(col, 0);
            if rows > 0 {
                let _ = shard.feature_f32(col, rows - 1);
            }
        }
        let _ = shard.feature_f32("observation.state", usize::MAX);
    }
});
