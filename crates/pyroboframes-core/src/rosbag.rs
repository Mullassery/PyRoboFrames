//! ROS 2 bag (`rosbag2` SQLite `.db3`) → columnar (Parquet) conversion.
//!
//! A rosbag2 SQLite bag stores `topics` (name, type, serialization format) and `messages`
//! (topic_id, timestamp, CDR blob); modern bags also embed each type's `ros2msg` definition in a
//! `message_definitions` table. This reader decodes the CDR payloads of `cdr`-serialized topics
//! against those definitions (via [`crate::ros2`]) and writes one flattened Parquet table per
//! topic — reusing the same columnar machinery as the MCAP converter. Topics whose definition is
//! missing or whose serialization format isn't CDR are reported as skipped.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use rusqlite::{Connection, OpenFlags};

use crate::mcap::{flatten, write_all, ConversionReport, TopicAccum};
use crate::ros2::Ros2Schema;
use crate::{Error, Result};

fn db_err(context: &str, e: rusqlite::Error) -> Error {
    Error::Conversion(format!("{context}: {e}"))
}

/// Convert a rosbag2 SQLite bag at `input` into one Parquet table per CDR topic under `out_dir`.
pub fn convert(input: &Path, out_dir: &Path) -> Result<ConversionReport> {
    let conn = Connection::open_with_flags(input, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| db_err(&format!("opening rosbag2 db `{}`", input.display()), e))?;

    let definitions = read_message_definitions(&conn)?;
    let topics = read_topics(&conn)?;

    // Resolve a decoder per topic id where we have a ros2msg definition and CDR serialization.
    let mut schemas: HashMap<i64, Ros2Schema> = HashMap::new();
    let mut topic_names: HashMap<i64, String> = HashMap::new();
    let mut skipped: BTreeSet<String> = BTreeSet::new();
    for t in &topics {
        topic_names.insert(t.id, t.name.clone());
        if !t.serialization_format.eq_ignore_ascii_case("cdr") {
            skipped.insert(t.name.clone());
            continue;
        }
        match definitions.get(&t.msg_type).and_then(|d| Ros2Schema::parse(d).ok()) {
            Some(schema) => {
                schemas.insert(t.id, schema);
            }
            None => {
                skipped.insert(t.name.clone());
            }
        }
    }

    // Stream messages in time order, decoding each topic's CDR blob.
    let mut accums: BTreeMap<String, TopicAccum> = BTreeMap::new();
    let mut stmt = conn
        .prepare("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp")
        .map_err(|e| db_err("querying messages", e))?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        })
        .map_err(|e| db_err("reading messages", e))?;

    for row in rows {
        let (topic_id, timestamp, data) = row.map_err(|e| db_err("reading message row", e))?;
        let (Some(schema), Some(name)) = (schemas.get(&topic_id), topic_names.get(&topic_id)) else {
            continue;
        };
        // Tolerate the odd undecodable payload rather than failing the whole bag.
        let Ok(value) = crate::ros2::decode_cdr(schema, &data) else {
            continue;
        };
        let mut leaves = BTreeMap::new();
        flatten("", &value, &mut leaves);
        let acc = accums.entry(name.clone()).or_default();
        acc.log_times.push(timestamp);
        acc.rows.push(leaves);
    }

    Ok(ConversionReport {
        topics: write_all(out_dir, &accums)?,
        skipped_topics: skipped.into_iter().collect(),
    })
}

struct TopicRow {
    id: i64,
    name: String,
    msg_type: String,
    serialization_format: String,
}

fn read_topics(conn: &Connection) -> Result<Vec<TopicRow>> {
    let mut stmt = conn
        .prepare("SELECT id, name, type, serialization_format FROM topics")
        .map_err(|e| db_err("querying topics", e))?;
    let rows = stmt
        .query_map([], |row| {
            Ok(TopicRow {
                id: row.get(0)?,
                name: row.get(1)?,
                msg_type: row.get(2)?,
                serialization_format: row.get(3)?,
            })
        })
        .map_err(|e| db_err("reading topics", e))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| db_err("reading topic row", e))
}

/// Read `message_definitions` (type → `ros2msg` text). The table is absent in older bags, in which
/// case there's nothing to decode against — return an empty map rather than erroring.
fn read_message_definitions(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut stmt = match conn
        .prepare("SELECT topic_type, encoded_message_definition FROM message_definitions")
    {
        Ok(stmt) => stmt,
        Err(_) => return Ok(HashMap::new()),
    };
    let rows = stmt
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))
        .map_err(|e| db_err("reading message_definitions", e))?;
    let mut map = HashMap::new();
    for row in rows {
        let (ty, text) = row.map_err(|e| db_err("reading definition row", e))?;
        map.insert(ty, text);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;

    #[test]
    fn converts_a_rosbag2_sqlite_bag() {
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("bag.db3");
        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT,
                     serialization_format TEXT, offered_qos_profiles TEXT);
                 CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER,
                     data BLOB);
                 CREATE TABLE message_definitions(id INTEGER PRIMARY KEY, topic_type TEXT,
                     encoding TEXT, encoded_message_definition TEXT);",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO topics VALUES(1, '/reading', 'demo/Reading', 'cdr', '')",
                [],
            )
            .unwrap();
            // A non-CDR topic must be reported as skipped.
            conn.execute(
                "INSERT INTO topics VALUES(2, '/raw', 'demo/Raw', 'protobuf', '')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO message_definitions VALUES(1, 'demo/Reading', 'ros2msg', ?1)",
                params!["float64 value\nint32 seq\n"],
            )
            .unwrap();

            // CDR LE: header, value (f64 @ offset 8), seq (i32 @ offset 16).
            let msg = |value: f64, seq: i32| {
                let mut b = vec![0x00u8, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
                b.extend_from_slice(&value.to_le_bytes());
                b.extend_from_slice(&seq.to_le_bytes());
                b
            };
            conn.execute(
                "INSERT INTO messages VALUES(1, 1, 100, ?1)",
                params![msg(2.5, 9)],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO messages VALUES(2, 1, 200, ?1)",
                params![msg(3.5, 10)],
            )
            .unwrap();
        }

        let out = tmp.path().join("out");
        let report = convert(&db, &out).unwrap();

        assert_eq!(report.skipped_topics, vec!["/raw".to_string()]);
        assert_eq!(report.topics.len(), 1);
        let t = &report.topics[0];
        assert_eq!(t.topic, "/reading");
        assert_eq!(t.messages, 2);
        assert_eq!(t.columns, 2); // value, seq

        let file = std::fs::File::open(&t.path).unwrap();
        let batch = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let schema = batch.schema();
        use arrow::array::{Float64Array, Int64Array};
        let log_time = batch
            .column(schema.index_of("log_time").unwrap())
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(log_time.values(), &[100, 200]);
        let value = batch
            .column(schema.index_of("value").unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(value.values(), &[2.5, 3.5]);
    }
}
