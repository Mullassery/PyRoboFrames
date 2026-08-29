//! Fuzzes `pyroboframes_core::ros2::decode_cdr`, the hand-rolled CDR (OMG
//! XCDR1) decoder shared by the MCAP `cdr` topic path (`mcap.rs`) and the
//! rosbag2 reader (`rosbag.rs`). It's the most boundary-sensitive parser in
//! the crate: manual offset tracking, size-based alignment, and
//! length-prefixed arrays/strings read directly off attacker-controlled
//! bytes. The schema is held fixed (covering primitives, a fixed-size array,
//! a variable-length array, a string, and a nested message type) so the
//! fuzzer's input lands entirely on the payload bytes being decoded against
//! it -- truncated payloads, huge/garbage length prefixes, and misaligned
//! data are exactly what a real CDR fuzzer would produce.
#![no_main]

use libfuzzer_sys::fuzz_target;
use pyroboframes_core::ros2::{decode_cdr, Ros2Schema};
use std::sync::OnceLock;

const SCHEMA_TEXT: &str = "\
float64 x\n\
int32 a\n\
uint8 flags[4]\n\
float64[] vals\n\
string name\n\
geometry_msgs/Point position\n\
================================================================================\n\
MSG: geometry_msgs/Point\n\
float64 x\n\
float64 y\n\
float64 z\n";

fn schema() -> &'static Ros2Schema {
    static SCHEMA: OnceLock<Ros2Schema> = OnceLock::new();
    SCHEMA.get_or_init(|| Ros2Schema::parse(SCHEMA_TEXT).expect("fixed schema text is valid"))
}

fuzz_target!(|data: &[u8]| {
    let _ = decode_cdr(schema(), data);
});
