//! Fuzzes `pyroboframes_core::mcap::convert`, the real MCAP -> Parquet entry
//! point: it reads a `.mcap` file from disk, parses it via `mcap::MessageStream`
//! (chunk/record framing, CRC, index), then decodes each channel's payloads
//! (json / protobuf-via-embedded-descriptor / cdr-ros2msg) and writes the
//! result out as Arrow/Parquet. Truncated files, corrupted chunk/record
//! headers, bogus channel/schema metadata, and malformed message payloads are
//! exactly the kind of input a real `.mcap` fuzzer would throw at this.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(tmp) = tempfile::tempdir() else {
        return;
    };
    let input = tmp.path().join("fuzz.mcap");
    if std::fs::write(&input, data).is_err() {
        return;
    }
    let out = tmp.path().join("out");

    // Errors are expected for malformed/truncated input -- convert() is
    // designed to reject bad files with `Err`, not panic. We only care that
    // it never crashes, hangs, or reads out of bounds.
    let _ = pyroboframes_core::mcap::convert(&input, &out);
});
