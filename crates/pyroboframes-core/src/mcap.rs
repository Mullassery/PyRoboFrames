//! MCAP → columnar (Parquet) conversion.
//!
//! MCAP is the de-facto container format for robotics logs (ROS 2 bags, Foxglove, many
//! teleoperation stacks). This module reads an `.mcap` file and writes **one Parquet table per
//! topic** whose messages are JSON-encoded, flattening each message into scalar columns (dot-path
//! names, e.g. `pose.position.0`) alongside a `log_time` nanosecond column. Non-JSON topics
//! (protobuf / ros2msg / cdr) are reported as skipped — decoding those needs their schema and is a
//! follow-up.
//!
//! This is the first step of the Tier-1 "data platform" identity: turn raw robot logs into the
//! columnar tables the rest of PyRoboFrames — and any Parquet-native tooling — can consume.
//!
//! ```no_run
//! use std::path::Path;
//! let report = pyroboframes_core::mcap::convert(Path::new("run.mcap"), Path::new("out/")).unwrap();
//! for t in &report.topics {
//!     println!("{}: {} messages, {} columns -> {}", t.topic, t.messages, t.columns, t.path.display());
//! }
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, Int64Array, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde_json::Value;

use crate::{Error, Result};

/// One converted topic in a [`ConversionReport`].
#[derive(Debug, Clone)]
pub struct TopicConversion {
    pub topic: String,
    /// Number of JSON messages written as rows.
    pub messages: usize,
    /// Number of data columns (flattened leaf fields, excluding `log_time`).
    pub columns: usize,
    /// Path of the written Parquet file.
    pub path: PathBuf,
}

/// Result of [`convert`]: the topics written, and the topics skipped (non-JSON encodings).
#[derive(Debug, Clone, Default)]
pub struct ConversionReport {
    pub topics: Vec<TopicConversion>,
    pub skipped_topics: Vec<String>,
}

/// A scalar leaf flattened from a JSON message.
#[derive(Clone)]
enum Leaf {
    Num(f64),
    Bool(bool),
    Str(String),
}

/// The Parquet column type inferred for a leaf path after scanning every message of a topic.
#[derive(Clone, Copy, PartialEq)]
enum ColKind {
    Num,
    Bool,
    Str,
}

/// Accumulated rows for one topic before it's written out.
struct TopicAccum {
    log_times: Vec<i64>,
    rows: Vec<BTreeMap<String, Leaf>>,
}

/// Convert an MCAP file at `input` into one Parquet table per JSON topic under `out_dir`
/// (created if absent). Returns a [`ConversionReport`]; non-JSON topics are listed in
/// `skipped_topics`.
pub fn convert(input: &Path, out_dir: &Path) -> Result<ConversionReport> {
    let bytes = fs::read(input)?;
    let stream = ::mcap::MessageStream::new(&bytes)
        .map_err(|e| Error::Conversion(format!("opening MCAP `{}`: {e}", input.display())))?;

    let mut topics: BTreeMap<String, TopicAccum> = BTreeMap::new();
    let mut skipped: BTreeSet<String> = BTreeSet::new();

    for message in stream {
        let message = message.map_err(|e| Error::Conversion(format!("reading message: {e}")))?;
        let topic = message.channel.topic.clone();

        if !message.channel.message_encoding.eq_ignore_ascii_case("json") {
            skipped.insert(topic);
            continue;
        }
        // Tolerate the odd malformed payload rather than failing the whole file.
        let Ok(value) = serde_json::from_slice::<Value>(&message.data) else {
            continue;
        };
        let mut leaves = BTreeMap::new();
        flatten("", &value, &mut leaves);

        let acc = topics.entry(topic).or_insert_with(|| TopicAccum {
            log_times: Vec::new(),
            rows: Vec::new(),
        });
        acc.log_times.push(message.log_time as i64);
        acc.rows.push(leaves);
    }

    fs::create_dir_all(out_dir)?;
    let mut converted = Vec::new();
    for (topic, acc) in &topics {
        if acc.rows.is_empty() {
            continue;
        }
        let path = out_dir.join(format!("{}.parquet", sanitize(topic)));
        let columns = write_topic(&path, acc)?;
        converted.push(TopicConversion {
            topic: topic.clone(),
            messages: acc.rows.len(),
            columns,
            path,
        });
    }

    Ok(ConversionReport {
        topics: converted,
        skipped_topics: skipped.into_iter().collect(),
    })
}

/// Recursively flatten a JSON value into `prefix`-keyed scalar leaves. Objects extend the path by
/// `.key`; arrays by `.index` (so a state vector becomes `state.0`, `state.1`, …). Nulls are
/// dropped (absent → null column cell).
fn flatten(prefix: &str, value: &Value, out: &mut BTreeMap<String, Leaf>) {
    let key = |seg: String| {
        if prefix.is_empty() {
            seg
        } else {
            format!("{prefix}.{seg}")
        }
    };
    match value {
        Value::Null => {}
        Value::Bool(b) => {
            out.insert(prefix.to_string(), Leaf::Bool(*b));
        }
        Value::Number(n) => {
            out.insert(prefix.to_string(), Leaf::Num(n.as_f64().unwrap_or(f64::NAN)));
        }
        Value::String(s) => {
            out.insert(prefix.to_string(), Leaf::Str(s.clone()));
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                flatten(&key(i.to_string()), item, out);
            }
        }
        Value::Object(map) => {
            for (k, item) in map {
                flatten(&key(k.clone()), item, out);
            }
        }
    }
}

/// Write one topic's accumulated rows to a Parquet file. Returns the number of data columns.
fn write_topic(path: &Path, acc: &TopicAccum) -> Result<usize> {
    // Union of all leaf paths seen across the topic's messages, sorted for a stable schema.
    let mut paths: BTreeSet<String> = BTreeSet::new();
    for row in &acc.rows {
        paths.extend(row.keys().cloned());
    }
    let paths: Vec<String> = paths.into_iter().collect();

    let mut fields = vec![Field::new("log_time", DataType::Int64, false)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(acc.log_times.clone()))];

    for p in &paths {
        let kind = infer_kind(&acc.rows, p);
        let (field_type, array) = match kind {
            ColKind::Num => {
                let mut b = Float64Builder::with_capacity(acc.rows.len());
                for row in &acc.rows {
                    match row.get(p) {
                        Some(Leaf::Num(v)) => b.append_value(*v),
                        _ => b.append_null(),
                    }
                }
                (DataType::Float64, Arc::new(b.finish()) as ArrayRef)
            }
            ColKind::Bool => {
                let mut b = BooleanBuilder::with_capacity(acc.rows.len());
                for row in &acc.rows {
                    match row.get(p) {
                        Some(Leaf::Bool(v)) => b.append_value(*v),
                        _ => b.append_null(),
                    }
                }
                (DataType::Boolean, Arc::new(b.finish()) as ArrayRef)
            }
            ColKind::Str => {
                let mut b = StringBuilder::new();
                for row in &acc.rows {
                    match row.get(p) {
                        Some(leaf) => b.append_value(leaf_to_string(leaf)),
                        None => b.append_null(),
                    }
                }
                (DataType::Utf8, Arc::new(b.finish()) as ArrayRef)
            }
        };
        fields.push(Field::new(p, field_type, true));
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| Error::Conversion(format!("building record batch: {e}")))?;

    let file =
        fs::File::create(path).map_err(|e| Error::Conversion(format!("creating Parquet: {e}")))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|e| Error::Conversion(format!("opening Parquet writer: {e}")))?;
    writer
        .write(&batch)
        .map_err(|e| Error::Conversion(format!("writing Parquet: {e}")))?;
    writer
        .close()
        .map_err(|e| Error::Conversion(format!("finalizing Parquet: {e}")))?;

    Ok(paths.len())
}

/// Decide a column's type from the leaf types present at `path`. Uniform numeric → `Num`, uniform
/// boolean → `Bool`; anything mixed or string-bearing falls back to `Str`.
fn infer_kind(rows: &[BTreeMap<String, Leaf>], path: &str) -> ColKind {
    let (mut num, mut boolean, mut string) = (false, false, false);
    for row in rows {
        match row.get(path) {
            Some(Leaf::Num(_)) => num = true,
            Some(Leaf::Bool(_)) => boolean = true,
            Some(Leaf::Str(_)) => string = true,
            None => {}
        }
    }
    if string || (num && boolean) {
        ColKind::Str
    } else if boolean {
        ColKind::Bool
    } else {
        ColKind::Num
    }
}

fn leaf_to_string(leaf: &Leaf) -> String {
    match leaf {
        Leaf::Num(v) => v.to_string(),
        Leaf::Bool(b) => b.to_string(),
        Leaf::Str(s) => s.clone(),
    }
}

/// Turn a topic name into a filesystem-safe stem: non-alphanumerics become `_`, leading/trailing
/// `_` trimmed (so `/joint_states` → `joint_states`). Empty results fall back to `topic`.
fn sanitize(topic: &str) -> String {
    let mapped: String = topic
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let trimmed = mapped.trim_matches('_');
    if trimmed.is_empty() {
        "topic".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::borrow::Cow;
    use std::collections::BTreeMap as Map;
    use std::sync::Arc;

    /// Write a small MCAP with two JSON messages on `/state` and one non-JSON message on `/raw`.
    fn write_mcap(path: &Path) {
        let file = fs::File::create(path).unwrap();
        let mut writer = ::mcap::Writer::new(file).unwrap();

        let json_chan = Arc::new(::mcap::Channel {
            topic: "/state".into(),
            schema: None,
            message_encoding: "json".into(),
            metadata: Map::new(),
        });
        let payloads = [
            br#"{"observation":{"state":[1.0,2.0]},"gripper":true}"#.to_vec(),
            br#"{"observation":{"state":[3.0,4.0]},"gripper":false}"#.to_vec(),
        ];
        for (i, payload) in payloads.iter().enumerate() {
            let t = (i as u64 + 1) * 1_000;
            writer
                .write(&::mcap::Message {
                    channel: json_chan.clone(),
                    sequence: i as u32,
                    log_time: t,
                    publish_time: t,
                    data: Cow::Owned(payload.clone()),
                })
                .unwrap();
        }

        // A non-JSON topic that must be reported as skipped.
        let raw_chan = Arc::new(::mcap::Channel {
            topic: "/raw".into(),
            schema: None,
            message_encoding: "protobuf".into(),
            metadata: Map::new(),
        });
        writer
            .write(&::mcap::Message {
                channel: raw_chan,
                sequence: 0,
                log_time: 5_000,
                publish_time: 5_000,
                data: Cow::Owned(vec![0xDE, 0xAD]),
            })
            .unwrap();

        writer.finish().unwrap();
    }

    #[test]
    fn converts_json_topics_and_skips_others() {
        let tmp = tempfile::tempdir().unwrap();
        let mcap_path = tmp.path().join("run.mcap");
        write_mcap(&mcap_path);
        let out = tmp.path().join("out");

        let report = convert(&mcap_path, &out).unwrap();

        // One JSON topic converted; the protobuf topic skipped.
        assert_eq!(report.topics.len(), 1);
        assert_eq!(report.skipped_topics, vec!["/raw".to_string()]);
        let t = &report.topics[0];
        assert_eq!(t.topic, "/state");
        assert_eq!(t.messages, 2);
        // Columns: observation.state.0, observation.state.1, gripper
        assert_eq!(t.columns, 3);
        assert!(t.path.ends_with("state.parquet"));
        assert!(t.path.exists());

        // Read the Parquet back and check the flattened values.
        let file = fs::File::open(&t.path).unwrap();
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);

        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(names.contains(&"log_time"));
        assert!(names.contains(&"observation.state.0"));
        assert!(names.contains(&"gripper"));

        use arrow::array::{BooleanArray, Float64Array, Int64Array};
        let col = |n: &str| schema.index_of(n).unwrap();
        let log_time = batch
            .column(col("log_time"))
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(log_time.values(), &[1_000, 2_000]);

        let s0 = batch
            .column(col("observation.state.0"))
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(s0.values(), &[1.0, 3.0]);

        let gripper = batch
            .column(col("gripper"))
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(gripper.value(0) && !gripper.value(1));
    }

    #[test]
    fn missing_fields_become_nulls() {
        // Two messages where the second omits `extra` -> nullable column with one null.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("run.mcap");
        {
            let file = fs::File::create(&path).unwrap();
            let mut writer = ::mcap::Writer::new(file).unwrap();
            let chan = Arc::new(::mcap::Channel {
                topic: "/t".into(),
                schema: None,
                message_encoding: "json".into(),
                metadata: Map::new(),
            });
            for (i, p) in [r#"{"a":1,"extra":9}"#, r#"{"a":2}"#].iter().enumerate() {
                writer
                    .write(&::mcap::Message {
                        channel: chan.clone(),
                        sequence: i as u32,
                        log_time: i as u64,
                        publish_time: i as u64,
                        data: Cow::Owned(p.as_bytes().to_vec()),
                    })
                    .unwrap();
            }
            writer.finish().unwrap();
        }

        let out = tmp.path().join("out");
        let report = convert(&path, &out).unwrap();
        let t = &report.topics[0];
        assert_eq!(t.columns, 2); // a, extra

        let file = fs::File::open(&t.path).unwrap();
        let batch = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let extra = batch
            .column(batch.schema().index_of("extra").unwrap())
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert!(!extra.is_null(0) && extra.is_null(1));
    }
}
