//! MCAP → columnar (Parquet) conversion.
//!
//! MCAP is the de-facto container format for robotics logs (ROS 2 bags, Foxglove, many
//! teleoperation stacks). This module reads an `.mcap` file and writes **one Parquet table per
//! topic**, flattening each message into scalar columns (dot-path names, e.g. `pose.position.0`)
//! alongside a `log_time` nanosecond column. Supported message encodings:
//! - **`json`** — parsed directly.
//! - **`protobuf`** — decoded dynamically from the channel's embedded `FileDescriptorSet`
//!   (no generated code needed), via [`prost_reflect`].
//!
//! Other encodings (`cdr`/`ros2msg`, …) are reported as skipped until their decoder lands.
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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, Int64Array, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor, SerializeOptions};
use serde_json::Value;

use crate::{Error, Result};

/// How a topic's messages are decoded into flattened leaves, resolved once per topic from its
/// channel's `message_encoding` (+ schema).
enum TopicDecoder {
    /// `json` message encoding — parse the payload as JSON.
    Json,
    /// `protobuf` — decode dynamically against the message type from the embedded descriptor set.
    Protobuf(MessageDescriptor),
    /// `cdr` with a `ros2msg` schema — decode dynamically against the parsed message definition.
    Ros2(crate::ros2::Ros2Schema),
    /// Encoding we can't decode yet — the topic is reported as skipped.
    Unsupported,
}

impl TopicDecoder {
    /// Resolve the decoder for a channel from its message encoding and schema.
    fn build(channel: &::mcap::Channel) -> Self {
        match channel.message_encoding.to_ascii_lowercase().as_str() {
            "json" => TopicDecoder::Json,
            "protobuf" => Self::build_protobuf(channel).unwrap_or(TopicDecoder::Unsupported),
            "cdr" => Self::build_ros2(channel).unwrap_or(TopicDecoder::Unsupported),
            _ => TopicDecoder::Unsupported,
        }
    }

    /// Build a ROS 2 decoder from a `cdr` channel whose schema is the `ros2msg` `.msg` text.
    fn build_ros2(channel: &::mcap::Channel) -> Option<Self> {
        let schema = channel.schema.as_ref()?;
        if !schema.encoding.eq_ignore_ascii_case("ros2msg") {
            return None;
        }
        let text = std::str::from_utf8(&schema.data).ok()?;
        let parsed = crate::ros2::Ros2Schema::parse(text).ok()?;
        Some(TopicDecoder::Ros2(parsed))
    }

    /// Build a protobuf decoder from the channel's schema: the schema `data` is a serialized
    /// `FileDescriptorSet` and `name` is the fully-qualified message type. `None` if either is
    /// missing or doesn't parse (→ the topic is treated as unsupported).
    fn build_protobuf(channel: &::mcap::Channel) -> Option<Self> {
        let schema = channel.schema.as_ref()?;
        if !schema.encoding.eq_ignore_ascii_case("protobuf") {
            return None;
        }
        let pool = DescriptorPool::decode(schema.data.as_ref()).ok()?;
        let desc = pool.get_message_by_name(&schema.name)?;
        Some(TopicDecoder::Protobuf(desc))
    }

    /// Decode one message payload into flattened scalar leaves. `None` means this individual
    /// payload couldn't be decoded (it's dropped); `Unsupported` decoders never reach here.
    fn decode(&self, data: &[u8]) -> Option<BTreeMap<String, Leaf>> {
        let value: Value = match self {
            TopicDecoder::Json => serde_json::from_slice(data).ok()?,
            TopicDecoder::Protobuf(desc) => {
                let msg = DynamicMessage::decode(desc.clone(), data).ok()?;
                // proto field names (snake_case), 64-bit ints as numbers (not strings), and keep
                // default-valued fields so every message of a topic yields the same columns.
                let opts = SerializeOptions::new()
                    .stringify_64_bit_integers(false)
                    .use_proto_field_name(true)
                    .skip_default_fields(false);
                msg.serialize_with_options(serde_json::value::Serializer, &opts)
                    .ok()?
            }
            TopicDecoder::Ros2(schema) => crate::ros2::decode_cdr(schema, data).ok()?,
            TopicDecoder::Unsupported => return None,
        };
        let mut leaves = BTreeMap::new();
        flatten("", &value, &mut leaves);
        Some(leaves)
    }
}

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

/// Result of a conversion: the topics written, and the topics skipped (undecodable encodings).
#[derive(Debug, Clone, Default)]
pub struct ConversionReport {
    pub topics: Vec<TopicConversion>,
    pub skipped_topics: Vec<String>,
}

/// A scalar leaf flattened from a decoded message. Shared with [`crate::rosbag`].
#[derive(Clone)]
pub(crate) enum Leaf {
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

/// Accumulated rows for one topic before it's written out. Shared with [`crate::rosbag`].
#[derive(Default)]
pub(crate) struct TopicAccum {
    pub(crate) log_times: Vec<i64>,
    pub(crate) rows: Vec<BTreeMap<String, Leaf>>,
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
    // One decoder per topic, resolved from the first message's channel.
    let mut decoders: HashMap<String, TopicDecoder> = HashMap::new();

    for message in stream {
        let message = message.map_err(|e| Error::Conversion(format!("reading message: {e}")))?;
        let topic = message.channel.topic.clone();

        let decoder = decoders
            .entry(topic.clone())
            .or_insert_with(|| TopicDecoder::build(&message.channel));
        if matches!(decoder, TopicDecoder::Unsupported) {
            skipped.insert(topic);
            continue;
        }
        // Tolerate the odd malformed payload rather than failing the whole file.
        let Some(leaves) = decoder.decode(&message.data) else {
            continue;
        };

        let acc = topics.entry(topic).or_insert_with(|| TopicAccum {
            log_times: Vec::new(),
            rows: Vec::new(),
        });
        acc.log_times.push(message.log_time as i64);
        acc.rows.push(leaves);
    }

    Ok(ConversionReport {
        topics: write_all(out_dir, &topics)?,
        skipped_topics: skipped.into_iter().collect(),
    })
}

/// Write each accumulated topic to `out_dir/<sanitized topic>.parquet` (created if absent),
/// returning the per-topic [`TopicConversion`]s. Empty topics are skipped. Shared by the MCAP and
/// ROS 2 bag converters.
pub(crate) fn write_all(
    out_dir: &Path,
    topics: &BTreeMap<String, TopicAccum>,
) -> Result<Vec<TopicConversion>> {
    fs::create_dir_all(out_dir)?;
    let mut converted = Vec::new();
    // Accumulated metadata + stats for the dataset-level manifests.
    let mut topic_meta: Vec<Value> = Vec::new();
    let mut stats = serde_json::Map::new();

    for (topic, acc) in topics {
        if acc.rows.is_empty() {
            continue;
        }
        let path = out_dir.join(format!("{}.parquet", sanitize(topic)));
        let infos = write_topic(&path, acc)?;

        // Column dtypes + per-topic numeric stats.
        let mut columns = serde_json::Map::new();
        let mut topic_stats = serde_json::Map::new();
        for info in &infos {
            columns.insert(info.name.clone(), Value::from(info.dtype));
            if let Some(s) = &info.stats {
                topic_stats.insert(
                    info.name.clone(),
                    serde_json::json!({
                        "count": s.count,
                        "mean": s.mean,
                        "std": s.std,
                        "min": s.min,
                        "max": s.max,
                    }),
                );
            }
        }
        if !topic_stats.is_empty() {
            stats.insert(topic.clone(), Value::Object(topic_stats));
        }
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        topic_meta.push(serde_json::json!({
            "topic": topic,
            "path": file_name,
            "num_rows": acc.rows.len(),
            "log_time_start": acc.log_times.iter().min().copied(),
            "log_time_end": acc.log_times.iter().max().copied(),
            "columns": Value::Object(columns),
        }));

        converted.push(TopicConversion {
            topic: topic.clone(),
            messages: acc.rows.len(),
            columns: infos.len(),
            path,
        });
    }

    // Dataset-level manifests: a self-describing metadata.json + a loader-compatible stats.json.
    let metadata = serde_json::json!({
        "format": "pyroboframes-columnar",
        "version": 1,
        "topics": topic_meta,
    });
    write_json(&out_dir.join("metadata.json"), &metadata)?;
    write_json(&out_dir.join("stats.json"), &Value::Object(stats))?;

    Ok(converted)
}

/// Write a JSON value to `path`, pretty-printed.
fn write_json(path: &Path, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| Error::Conversion(format!("serializing {}: {e}", path.display())))?;
    fs::write(path, text)
        .map_err(|e| Error::Conversion(format!("writing {}: {e}", path.display())))?;
    Ok(())
}

/// Recursively flatten a JSON value into `prefix`-keyed scalar leaves. Objects extend the path by
/// `.key`; arrays by `.index` (so a state vector becomes `state.0`, `state.1`, …). Nulls are
/// dropped (absent → null column cell). Shared with [`crate::rosbag`].
pub(crate) fn flatten(prefix: &str, value: &Value, out: &mut BTreeMap<String, Leaf>) {
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
            out.insert(
                prefix.to_string(),
                Leaf::Num(n.as_f64().unwrap_or(f64::NAN)),
            );
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

/// Per-column summary returned by [`write_topic`] for metadata/stats generation.
pub(crate) struct ColInfo {
    pub(crate) name: String,
    /// Parquet dtype: `"float64"`, `"bool"`, or `"string"`.
    pub(crate) dtype: &'static str,
    /// Numeric statistics (numeric columns only).
    pub(crate) stats: Option<ColStats>,
}

/// Per-feature numeric statistics, mirroring the `meta/stats.json` shape the loader reads for
/// normalization.
pub(crate) struct ColStats {
    pub(crate) count: usize,
    pub(crate) mean: f64,
    pub(crate) std: f64,
    pub(crate) min: f64,
    pub(crate) max: f64,
}

/// Compute count / mean / population-std / min / max over the finite values present at `path`.
fn column_stats(rows: &[BTreeMap<String, Leaf>], path: &str) -> ColStats {
    let mut values: Vec<f64> = Vec::new();
    for row in rows {
        if let Some(Leaf::Num(v)) = row.get(path) {
            if v.is_finite() {
                values.push(*v);
            }
        }
    }
    let count = values.len();
    if count == 0 {
        return ColStats {
            count: 0,
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }
    let mean = values.iter().sum::<f64>() / count as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    ColStats {
        count,
        mean,
        std: var.sqrt(),
        min,
        max,
    }
}

/// Write one topic's accumulated rows to a Parquet file. Returns the per-column info (name, dtype,
/// numeric stats) used to generate dataset metadata.
fn write_topic(path: &Path, acc: &TopicAccum) -> Result<Vec<ColInfo>> {
    // Union of all leaf paths seen across the topic's messages, sorted for a stable schema.
    let mut paths: BTreeSet<String> = BTreeSet::new();
    for row in &acc.rows {
        paths.extend(row.keys().cloned());
    }
    let paths: Vec<String> = paths.into_iter().collect();

    let mut fields = vec![Field::new("log_time", DataType::Int64, false)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(acc.log_times.clone()))];
    let mut infos: Vec<ColInfo> = Vec::with_capacity(paths.len());

    for p in &paths {
        let kind = infer_kind(&acc.rows, p);
        let dtype = match kind {
            ColKind::Num => "float64",
            ColKind::Bool => "bool",
            ColKind::Str => "string",
        };
        infos.push(ColInfo {
            name: p.clone(),
            dtype,
            stats: matches!(kind, ColKind::Num).then(|| column_stats(&acc.rows, p)),
        });
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

    Ok(infos)
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

    /// Build a `FileDescriptorSet` for `demo.State { repeated double position = 1; bool gripper = 2;
    /// int64 stamp = 3; }` — the schema a protobuf MCAP channel embeds.
    fn proto_descriptor_set() -> Vec<u8> {
        use prost::Message as _;
        use prost_types::field_descriptor_proto::{Label, Type};
        use prost_types::{
            DescriptorProto, FieldDescriptorProto, FileDescriptorProto, FileDescriptorSet,
        };
        let field = |name: &str, num: i32, ty: Type, label: Label| FieldDescriptorProto {
            name: Some(name.to_string()),
            number: Some(num),
            label: Some(label as i32),
            r#type: Some(ty as i32),
            ..Default::default()
        };
        let msg = DescriptorProto {
            name: Some("State".to_string()),
            field: vec![
                field("position", 1, Type::Double, Label::Repeated),
                field("gripper", 2, Type::Bool, Label::Optional),
                field("stamp", 3, Type::Int64, Label::Optional),
            ],
            ..Default::default()
        };
        let file = FileDescriptorProto {
            name: Some("state.proto".to_string()),
            package: Some("demo".to_string()),
            message_type: vec![msg],
            syntax: Some("proto3".to_string()),
            ..Default::default()
        };
        FileDescriptorSet { file: vec![file] }.encode_to_vec()
    }

    #[test]
    fn writes_dataset_metadata_and_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let mcap_path = tmp.path().join("run.mcap");
        write_mcap(&mcap_path);
        let out = tmp.path().join("out");
        convert(&mcap_path, &out).unwrap();

        // metadata.json: self-describing manifest.
        let meta: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(meta["format"], "pyroboframes-columnar");
        let topic = &meta["topics"][0];
        assert_eq!(topic["topic"], "/state");
        assert_eq!(topic["path"], "state.parquet");
        assert_eq!(topic["num_rows"], 2);
        assert_eq!(topic["log_time_start"], 1000);
        assert_eq!(topic["log_time_end"], 2000);
        assert_eq!(topic["columns"]["gripper"], "bool");
        assert_eq!(topic["columns"]["observation.state.0"], "float64");

        // stats.json: loader-compatible mean/std/min/max/count (numeric columns only).
        let stats: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("stats.json")).unwrap()).unwrap();
        let s0 = &stats["/state"]["observation.state.0"];
        assert_eq!(s0["count"], 2);
        assert_eq!(s0["mean"], 2.0); // (1 + 3) / 2
        assert_eq!(s0["min"], 1.0);
        assert_eq!(s0["max"], 3.0);
        // bool column has no stats entry.
        assert!(stats["/state"].get("gripper").is_none());
    }

    #[test]
    fn converts_protobuf_topics_via_embedded_descriptor() {
        use prost::Message as _;
        use prost_reflect::Value as PValue;

        let fds = proto_descriptor_set();
        let pool = DescriptorPool::decode(fds.as_ref()).unwrap();
        let desc = pool.get_message_by_name("demo.State").unwrap();

        // Encode two protobuf messages.
        let encode = |pos: [f64; 2], grip: bool, stamp: i64| {
            let mut m = DynamicMessage::new(desc.clone());
            m.set_field_by_name(
                "position",
                PValue::List(vec![PValue::F64(pos[0]), PValue::F64(pos[1])]),
            );
            m.set_field_by_name("gripper", PValue::Bool(grip));
            m.set_field_by_name("stamp", PValue::I64(stamp));
            m.encode_to_vec()
        };

        let tmp = tempfile::tempdir().unwrap();
        let mcap_path = tmp.path().join("run.mcap");
        {
            let file = fs::File::create(&mcap_path).unwrap();
            let mut writer = ::mcap::Writer::new(file).unwrap();
            let schema = Arc::new(::mcap::Schema {
                name: "demo.State".into(),
                encoding: "protobuf".into(),
                data: Cow::Owned(fds.clone()),
            });
            let chan = Arc::new(::mcap::Channel {
                topic: "/state".into(),
                schema: Some(schema),
                message_encoding: "protobuf".into(),
                metadata: Map::new(),
            });
            for (i, payload) in [encode([1.0, 2.0], true, 10), encode([3.0, 4.0], false, 20)]
                .into_iter()
                .enumerate()
            {
                let t = (i as u64 + 1) * 1000;
                writer
                    .write(&::mcap::Message {
                        channel: chan.clone(),
                        sequence: i as u32,
                        log_time: t,
                        publish_time: t,
                        data: Cow::Owned(payload),
                    })
                    .unwrap();
            }
            writer.finish().unwrap();
        }

        let out = tmp.path().join("out");
        let report = convert(&mcap_path, &out).unwrap();
        assert_eq!(report.skipped_topics.len(), 0);
        let t = &report.topics[0];
        assert_eq!(t.topic, "/state");
        assert_eq!(t.messages, 2);

        let file = fs::File::open(&t.path).unwrap();
        let batch = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let schema = batch.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(names.contains(&"position.0"));
        assert!(names.contains(&"gripper"));
        assert!(names.contains(&"stamp")); // int64 stayed numeric (not stringified)

        use arrow::array::{BooleanArray, Float64Array};
        let col = |n: &str| schema.index_of(n).unwrap();
        let p1 = batch
            .column(col("position.1"))
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(p1.values(), &[2.0, 4.0]);
        let stamp = batch
            .column(col("stamp"))
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(stamp.values(), &[10.0, 20.0]);
        let grip = batch
            .column(col("gripper"))
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(grip.value(0) && !grip.value(1));
    }

    #[test]
    fn converts_cdr_ros2msg_topics() {
        let tmp = tempfile::tempdir().unwrap();
        let mcap_path = tmp.path().join("run.mcap");
        {
            let file = fs::File::create(&mcap_path).unwrap();
            let mut writer = ::mcap::Writer::new(file).unwrap();
            let schema = Arc::new(::mcap::Schema {
                name: "demo/Reading".into(),
                encoding: "ros2msg".into(),
                data: Cow::Owned(b"float64 value\nint32 seq\n".to_vec()),
            });
            let chan = Arc::new(::mcap::Channel {
                topic: "/reading".into(),
                schema: Some(schema),
                message_encoding: "cdr".into(),
                metadata: Map::new(),
            });
            // CDR LE: header, value=2.5 (f64 @ offset 8), seq=9 (i32 @ offset 16).
            let payload: Vec<u8> = vec![
                0x00, 0x01, 0x00, 0x00, // header
                0x00, 0x00, 0x00, 0x00, // pad to align f64
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x40, // 2.5
                0x09, 0x00, 0x00, 0x00, // 9
            ];
            writer
                .write(&::mcap::Message {
                    channel: chan,
                    sequence: 0,
                    log_time: 100,
                    publish_time: 100,
                    data: Cow::Owned(payload),
                })
                .unwrap();
            writer.finish().unwrap();
        }

        let out = tmp.path().join("out");
        let report = convert(&mcap_path, &out).unwrap();
        assert_eq!(report.skipped_topics.len(), 0);
        let t = &report.topics[0];
        assert_eq!(t.topic, "/reading");
        assert_eq!(t.columns, 2); // value, seq

        let file = fs::File::open(&t.path).unwrap();
        let batch = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let schema = batch.schema();
        use arrow::array::Float64Array;
        let value = batch
            .column(schema.index_of("value").unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(value.values(), &[2.5]);
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
