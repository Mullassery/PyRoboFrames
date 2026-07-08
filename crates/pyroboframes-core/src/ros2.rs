//! Dynamic ROS 2 message decoding: parse a `ros2msg` schema (the `.msg` text rosbag2 / MCAP embed)
//! and decode the matching **CDR**-serialized payloads into a `serde_json::Value` tree — no codegen
//! or installed ROS packages needed. Used by the MCAP converter (`cdr` topics) and the ROS 2 bag
//! reader; the resulting `Value` is flattened to columns by [`crate::mcap`].
//!
//! Scope: primitives, fixed / bounded / unbounded arrays, strings, and **nested message types**
//! (the schema text carries the full dependency closure, separated by `MSG:` blocks). CDR follows
//! the OMG XCDR1 rules rosbag2 uses: a 4-byte encapsulation header (endianness in byte 1), then
//! primitives aligned to their size relative to the start of the stream.

use std::collections::HashMap;

use serde_json::{Map, Number, Value};

use crate::{Error, Result};

/// A parsed `ros2msg` schema: the root message's fields plus a registry of referenced types.
pub struct Ros2Schema {
    root: Vec<FieldDef>,
    /// Type name (both `pkg/Type` and short `Type`) → its fields.
    types: HashMap<String, Vec<FieldDef>>,
}

struct FieldDef {
    name: String,
    base: String,
    array: ArrayKind,
}

enum ArrayKind {
    Single,
    Fixed(usize),
    Variable,
}

enum Prim {
    Bool,
    U8,
    I8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    Str,
}

fn primitive(base: &str) -> Option<Prim> {
    // Bounded strings arrive as `string<=N`; strip the bound before matching.
    let base = base.split("<=").next().unwrap_or(base);
    Some(match base {
        "bool" => Prim::Bool,
        "byte" | "uint8" | "char" => Prim::U8,
        "int8" => Prim::I8,
        "int16" => Prim::I16,
        "uint16" => Prim::U16,
        "int32" => Prim::I32,
        "uint32" => Prim::U32,
        "int64" => Prim::I64,
        "uint64" => Prim::U64,
        "float32" => Prim::F32,
        "float64" => Prim::F64,
        "string" | "wstring" => Prim::Str,
        _ => return None,
    })
}

impl Ros2Schema {
    /// Parse a `ros2msg` schema (root definition first, dependency types after `MSG:`/`===` blocks).
    pub fn parse(text: &str) -> Result<Self> {
        // Split into blocks at separator lines (`===...`); a block's name comes from its `MSG:` line.
        let mut blocks: Vec<(Option<String>, Vec<&str>)> = Vec::new();
        let mut name: Option<String> = None;
        let mut lines: Vec<&str> = Vec::new();
        for raw in text.lines() {
            let trimmed = raw.trim();
            if trimmed.len() >= 3 && trimmed.bytes().all(|b| b == b'=') {
                blocks.push((name.take(), std::mem::take(&mut lines)));
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("MSG:") {
                name = Some(rest.trim().to_string());
                continue;
            }
            lines.push(raw);
        }
        blocks.push((name.take(), std::mem::take(&mut lines)));

        let parse_fields =
            |ls: &[&str]| -> Vec<FieldDef> { ls.iter().filter_map(|l| parse_field(l)).collect() };

        let mut types: HashMap<String, Vec<FieldDef>> = HashMap::new();
        let mut root: Option<Vec<FieldDef>> = None;
        for (block_name, ls) in &blocks {
            match block_name {
                None => {
                    if root.is_none() {
                        root = Some(parse_fields(ls));
                    }
                }
                Some(n) => {
                    let fields = parse_fields(ls);
                    // Register under both the full `pkg/Type` key and the short `Type` name.
                    if let Some(short) = n.rsplit('/').next() {
                        if short != n {
                            types.insert(short.to_string(), parse_fields(ls));
                        }
                    }
                    types.insert(n.clone(), fields);
                }
            }
        }
        let root = root.ok_or_else(|| Error::Conversion("empty ros2msg schema".into()))?;
        Ok(Ros2Schema { root, types })
    }

    fn lookup(&self, base: &str) -> Option<&[FieldDef]> {
        self.types
            .get(base)
            .or_else(|| base.rsplit('/').next().and_then(|s| self.types.get(s)))
            .map(|v| v.as_slice())
    }
}

/// Parse one schema field line into a [`FieldDef`], or `None` for blanks / comments / constants.
fn parse_field(line: &str) -> Option<FieldDef> {
    let line = line.split('#').next().unwrap_or("").trim();
    if line.is_empty() {
        return None;
    }
    // Constants are `TYPE NAME=VALUE` — they aren't serialized, so skip any line with `=`.
    if line.contains('=') {
        return None;
    }
    let mut it = line.split_whitespace();
    let type_tok = it.next()?;
    let name = it.next()?.to_string();

    let (base, array) = match type_tok.find('[') {
        Some(i) if type_tok.ends_with(']') => {
            let base = type_tok[..i].to_string();
            let inside = &type_tok[i + 1..type_tok.len() - 1];
            let array = if inside.is_empty() || inside.starts_with("<=") {
                ArrayKind::Variable
            } else {
                inside
                    .parse::<usize>()
                    .map(ArrayKind::Fixed)
                    .unwrap_or(ArrayKind::Variable)
            };
            (base, array)
        }
        _ => (type_tok.to_string(), ArrayKind::Single),
    };
    Some(FieldDef { name, base, array })
}

/// Decode a CDR payload (with its 4-byte encapsulation header) against `schema`'s root type.
pub fn decode_cdr(schema: &Ros2Schema, data: &[u8]) -> Result<Value> {
    if data.len() < 4 {
        return Err(Error::Conversion("CDR payload too short for header".into()));
    }
    // Encapsulation header: byte 1 bit 0 selects endianness (1 = little-endian, as rosbag2 emits).
    let little_endian = data[1] & 1 == 1;
    let mut cdr = Cdr {
        buf: data,
        pos: 4,
        le: little_endian,
    };
    decode_fields(&mut cdr, &schema.root, schema)
}

fn decode_fields(cdr: &mut Cdr, fields: &[FieldDef], schema: &Ros2Schema) -> Result<Value> {
    let mut obj = Map::new();
    for f in fields {
        obj.insert(f.name.clone(), decode_field(cdr, f, schema)?);
    }
    Ok(Value::Object(obj))
}

fn decode_field(cdr: &mut Cdr, f: &FieldDef, schema: &Ros2Schema) -> Result<Value> {
    match f.array {
        ArrayKind::Single => decode_one(cdr, &f.base, schema),
        ArrayKind::Fixed(n) => {
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                items.push(decode_one(cdr, &f.base, schema)?);
            }
            Ok(Value::Array(items))
        }
        ArrayKind::Variable => {
            let n = cdr.read_u32()? as usize;
            let mut items = Vec::with_capacity(n.min(1 << 16));
            for _ in 0..n {
                items.push(decode_one(cdr, &f.base, schema)?);
            }
            Ok(Value::Array(items))
        }
    }
}

fn decode_one(cdr: &mut Cdr, base: &str, schema: &Ros2Schema) -> Result<Value> {
    if let Some(prim) = primitive(base) {
        return decode_prim(cdr, prim);
    }
    let fields = schema
        .lookup(base)
        .ok_or_else(|| Error::Conversion(format!("unknown ros2 type `{base}`")))?;
    decode_fields(cdr, fields, schema)
}

fn decode_prim(cdr: &mut Cdr, prim: Prim) -> Result<Value> {
    let num_i = |x: i64| Value::Number(Number::from(x));
    let num_u = |x: u64| Value::Number(Number::from(x));
    Ok(match prim {
        Prim::Bool => Value::Bool(cdr.read_u8()? != 0),
        Prim::U8 => num_u(cdr.read_u8()? as u64),
        Prim::I8 => num_i(cdr.read_u8()? as i8 as i64),
        Prim::I16 => num_i(cdr.read_i16()? as i64),
        Prim::U16 => num_u(cdr.read_u16()? as u64),
        Prim::I32 => num_i(cdr.read_i32()? as i64),
        Prim::U32 => num_u(cdr.read_u32()? as u64),
        Prim::I64 => num_i(cdr.read_i64()?),
        Prim::U64 => num_u(cdr.read_u64()?),
        Prim::F32 => float_value(cdr.read_f32()? as f64),
        Prim::F64 => float_value(cdr.read_f64()?),
        Prim::Str => Value::String(cdr.read_string()?),
    })
}

/// JSON can't hold non-finite floats; map NaN/±Inf to null (dropped by the flattener).
fn float_value(x: f64) -> Value {
    Number::from_f64(x)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// A little/big-endian CDR cursor with size-based alignment relative to the buffer start.
struct Cdr<'a> {
    buf: &'a [u8],
    pos: usize,
    le: bool,
}

impl Cdr<'_> {
    fn align(&mut self, n: usize) {
        let rem = self.pos % n;
        if rem != 0 {
            self.pos += n - rem;
        }
    }

    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::Conversion("CDR length overflow".into()))?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| Error::Conversion("CDR read past end of payload".into()))?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        self.align(2);
        let b = self.take(2)?;
        let a = [b[0], b[1]];
        Ok(if self.le {
            u16::from_le_bytes(a)
        } else {
            u16::from_be_bytes(a)
        })
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.align(4);
        let b = self.take(4)?;
        let a = [b[0], b[1], b[2], b[3]];
        Ok(if self.le {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        })
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.align(8);
        let b = self.take(8)?;
        let a = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        Ok(if self.le {
            u64::from_le_bytes(a)
        } else {
            u64::from_be_bytes(a)
        })
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.take(len)?;
        // The length includes the trailing NUL; drop it.
        let end = if bytes.last() == Some(&0) {
            len - 1
        } else {
            len
        };
        Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_decodes_primitives_arrays_and_strings() {
        // Schema: x (f64), a (i32), vals (f64[]), name (string).
        let schema = Ros2Schema::parse(
            "float64 x\n\
             int32 a\n\
             float64[] vals\n\
             string name\n",
        )
        .unwrap();

        // Hand-built little-endian CDR (offsets include the 4-byte header; primitives self-aligned).
        let bytes: Vec<u8> = vec![
            0x00, 0x01, 0x00, 0x00, // encapsulation header (CDR_LE)
            0x00, 0x00, 0x00, 0x00, // pad to align x (f64) to offset 8
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x3F, // x = 1.5
            0x07, 0x00, 0x00, 0x00, // a = 7
            0x02, 0x00, 0x00, 0x00, // vals length = 2
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // 2.0
            0x03, 0x00, 0x00, 0x00, // name length = 3 (incl. NUL)
            0x68, 0x69, 0x00, // "hi\0"
        ];

        let v = decode_cdr(&schema, &bytes).unwrap();
        assert_eq!(v["x"], serde_json::json!(1.5));
        assert_eq!(v["a"], serde_json::json!(7));
        assert_eq!(v["vals"], serde_json::json!([1.0, 2.0]));
        assert_eq!(v["name"], serde_json::json!("hi"));
    }

    #[test]
    fn decodes_nested_message_types() {
        // Root references a nested Point type carried in a later MSG block.
        let schema = Ros2Schema::parse(
            "geometry_msgs/Point position\n\
             bool ok\n\
             ================================================================================\n\
             MSG: geometry_msgs/Point\n\
             float64 x\n\
             float64 y\n",
        )
        .unwrap();

        // position.x=1.0, position.y=2.0, ok=true. Nested fields align within the same stream.
        let bytes: Vec<u8> = vec![
            0x00, 0x01, 0x00, 0x00, // header
            0x00, 0x00, 0x00, 0x00, // pad to offset 8 for first f64
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // x = 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y = 2.0
            0x01, // ok = true (offset 24, byte-aligned)
        ];

        let v = decode_cdr(&schema, &bytes).unwrap();
        assert_eq!(v["position"]["x"], serde_json::json!(1.0));
        assert_eq!(v["position"]["y"], serde_json::json!(2.0));
        assert_eq!(v["ok"], serde_json::json!(true));
    }

    #[test]
    fn skips_comments_and_constants() {
        let schema = Ros2Schema::parse(
            "# a comment\n\
             uint8 STATUS_OK=1\n\
             int32 value  # trailing comment\n",
        )
        .unwrap();
        assert_eq!(schema.root.len(), 1);
        assert_eq!(schema.root[0].name, "value");
    }
}
