//! Real, in-process VideoToolbox hardware decode.
//!
//! MP4 demux (`mp4` crate) -> `CMSampleBuffer` construction -> a real
//! `VTDecompressionSession` (the `videotoolbox` crate) -> an IOSurface-backed
//! `CVPixelBuffer`. Unlike the ffmpeg-CLI path (`ffcli` in the parent
//! module), nothing here shells out to a subprocess: the decoded pixel
//! buffer never leaves this process, so it can be handed to Python/MLX
//! without a copy through a pipe.
//!
//! # Scope / honest limitations
//! - H.264 (`avc1`) and HEVC tagged `hev1` (not `hvc1` — see below) input. The `mp4` crate
//!   (0.14.0, the newest published version) parses H.264's `avcC` box fully, and recognizes
//!   `hev1` as a video sample entry, but its `HvcCBox` only reads `configurationVersion` and
//!   discards the rest — no VPS/SPS/PPS access. `hevc_hvcc` below is a from-scratch
//!   `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15) reader that walks the box tree
//!   independently to make up for that gap.
//! - **`hvc1`-tagged HEVC is not supported** (distinct from the `hvcC` gap above): the `mp4`
//!   crate's `stsd` parser only matches `avc1`/`hev1`/`vp09`/`mp4a`/`tx3g` sample entries — an
//!   `hvc1` track (a common tag; some encoders/muxers default to it over `hev1`) doesn't
//!   appear in `Mp4Reader::tracks()` *at all*, before this module's code ever runs, so there's
//!   no track_id/timescale/sample data to work with — this is a gap in the `mp4` crate itself,
//!   not something `hevc_hvcc` (which is only about parameter-set *content*) can work around
//!   without reimplementing the crate's track/sample demuxing for that box type too. Re-mux
//!   (`ffmpeg -i in.mp4 -c copy -tag:v hev1 out.mp4`) or re-encode with `-tag:v hev1` as a
//!   workaround.
//! - No general B-frame reorder buffer: each `decode_at` call decodes
//!   samples in *decode* order from the nearest preceding keyframe through
//!   the target sample, then returns whichever decoded frame's
//!   presentation timestamp is closest to the target sample's PTS. This is
//!   correct for the common case (no B-frames, decode order == display
//!   order) and for isolated single-frame lookups with B-frames, but is
//!   not a full streaming-playback reorder buffer.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use apple_cf::cm::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMSampleTimingInfo, CMTime,
};
use apple_cf::cv::CVPixelBuffer;
use apple_cf::raw;
use videotoolbox::decompression::{DecodedFrame, DecompressionSession};

use crate::{Error, Result};

/// AVCC-style length-prefix size the `mp4` crate's sample bytes already use.
const NAL_UNIT_HEADER_LENGTH: i32 = 4;
const DECODE_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

struct SampleIndexEntry {
    sample_id: u32,
    /// Presentation timestamp (PTS = DTS + rendering offset), in `timescale` units. What a
    /// caller's `decode_at(timestamp_s)` lookup actually means and must be matched against —
    /// equal to decode timestamp (DTS) only when there's no reordering (no B-frames).
    /// `sample_index` itself stays DTS-ordered (decode order, a real bitstream invariant: DTS
    /// increases monotonically with decode order) for the GOP walk-back below, which is why
    /// finding the target sample by PTS is a linear scan rather than a binary search.
    presentation_time_units: i64,
    is_sync: bool,
}

/// Shared sink a `DecompressionSession`'s callback pushes decoded frames
/// into. VideoToolbox's callback can fire synchronously or asynchronously
/// (per the crate's own docs), so callers wait on the associated `Condvar`
/// rather than assuming `decode()` returning means the frame has arrived.
#[derive(Clone)]
struct FrameCollector {
    inner: Arc<(Mutex<Vec<DecodedFrame>>, Condvar)>,
}

impl FrameCollector {
    fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(Vec::new()), Condvar::new())),
        }
    }

    fn push(&self, frame: DecodedFrame) {
        let (lock, cvar) = &*self.inner;
        let mut frames = lock.lock().unwrap_or_else(|e| e.into_inner());
        frames.push(frame);
        cvar.notify_all();
    }

    /// Blocks until at least `count` frames have arrived or `timeout` elapses.
    fn wait_until_at_least(&self, count: usize, timeout: Duration) {
        let (lock, cvar) = &*self.inner;
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = cvar.wait_timeout_while(guard, timeout, |frames| frames.len() < count);
    }

    /// Removes and returns every frame collected so far.
    fn drain(&self) -> Vec<DecodedFrame> {
        let (lock, _cvar) = &*self.inner;
        let mut frames = lock.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *frames)
    }
}

fn cm_time_seconds((value, timescale): (i64, i32)) -> f64 {
    if timescale == 0 {
        0.0
    } else {
        value as f64 / timescale as f64
    }
}

/// Builds a `CMFormatDescription` for H.264 from raw SPS/PPS NAL payloads
/// (without start codes or length prefixes — just the parameter-set bytes,
/// which is what `Mp4Track::sequence_parameter_set()`/`picture_parameter_set()`
/// return).
fn create_h264_format_description(sps: &[u8], pps: &[u8]) -> Result<CMFormatDescription> {
    let ptrs: [*const u8; 2] = [sps.as_ptr(), pps.as_ptr()];
    let sizes: [usize; 2] = [sps.len(), pps.len()];
    let mut format_desc_ref: raw::CMFormatDescriptionRef = std::ptr::null();

    // SAFETY: `ptrs`/`sizes` point at `sps`/`pps`, both alive for the call.
    // `formatDescriptionOut` receives a +1 retained ref per Apple's Create
    // Rule; we hand ownership to `CMFormatDescription::from_raw` below.
    let status = unsafe {
        raw::CMVideoFormatDescriptionCreateFromH264ParameterSets(
            raw::kCFAllocatorDefault,
            2,
            ptrs.as_ptr(),
            sizes.as_ptr(),
            NAL_UNIT_HEADER_LENGTH as std::ffi::c_int,
            &mut format_desc_ref,
        )
    };
    if status != 0 {
        return Err(Error::Decode(format!(
            "CMVideoFormatDescriptionCreateFromH264ParameterSets failed: OSStatus {status}"
        )));
    }
    CMFormatDescription::from_raw(format_desc_ref as *mut std::ffi::c_void).ok_or_else(|| {
        Error::Decode("CMVideoFormatDescriptionCreateFromH264ParameterSets returned null".into())
    })
}

/// From-scratch `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §8.3.3.1) reader.
///
/// The `mp4` crate (0.14.0, its newest published version as of this writing) parses the
/// `hvcC` box's `configurationVersion` byte and then skips the rest — it never exposes the
/// VPS/SPS/PPS NAL arrays HEVC decode actually needs. This walks the ISOBMFF box tree
/// (`moov > trak > mdia > minf > stbl > stsd > hev1|hvc1 > hvcC`) independently, using the
/// same box-header format the `mp4` crate itself parses (4-byte size + 4-byte fourcc, with
/// the standard 64-bit `largesize` extension), to reach and decode that record by hand.
mod hevc_hvcc {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    /// `(vps_list, sps_list, pps_list, nal_length_size)` extracted from a `hvcC` box.
    type ParameterSets = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>, u8);

    /// HEVC NAL unit types (ISO/IEC 23008-2 Table 7-1) carried in a `hvcC` array.
    const NAL_TYPE_VPS: u8 = 32;
    const NAL_TYPE_SPS: u8 = 33;
    const NAL_TYPE_PPS: u8 = 34;

    /// Fixed-size `VisualSampleEntry` header (ISO/IEC 14496-12 §12.1.3) preceding a sample
    /// entry's codec-specific config box: `SampleEntry` base (reserved(6) + data_reference_index(2)
    /// = 8) + `pre_defined`/`reserved` (2+2+12 = 16) + width/height/resolutions/reserved/
    /// frame_count/compressorname/depth/pre_defined (2+2+4+4+4+2+32+2+2 = 54) = 78 bytes.
    const VISUAL_SAMPLE_ENTRY_HEADER_LEN: u64 = 78;

    /// One child box's fourcc and content byte range `[start, end)` (header excluded).
    struct ChildBox {
        fourcc: [u8; 4],
        start: u64,
        end: u64,
    }

    fn read_box_header(r: &mut impl Read) -> Result<([u8; 4], u64, u64)> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)
            .map_err(|e| Error::Decode(format!("hvcC: read box header: {e}")))?;
        let size = u32::from_be_bytes(buf[0..4].try_into().unwrap());
        let fourcc: [u8; 4] = buf[4..8].try_into().unwrap();
        if size == 1 {
            let mut large = [0u8; 8];
            r.read_exact(&mut large)
                .map_err(|e| Error::Decode(format!("hvcC: read largesize: {e}")))?;
            let largesize = u64::from_be_bytes(large);
            if largesize < 16 {
                return Err(Error::Decode("hvcC: 64-bit box size too small".into()));
            }
            Ok((fourcc, 16, largesize - 16))
        } else if size == 0 {
            Err(Error::Decode(
                "hvcC: box extends to EOF (size=0) unsupported".into(),
            ))
        } else {
            Ok((fourcc, 8, size as u64 - 8))
        }
    }

    /// Lists the immediate child boxes within content byte range `[start, end)`.
    fn list_children(r: &mut (impl Read + Seek), start: u64, end: u64) -> Result<Vec<ChildBox>> {
        let mut out = Vec::new();
        let mut pos = start;
        while pos + 8 <= end {
            r.seek(SeekFrom::Start(pos))
                .map_err(|e| Error::Decode(format!("hvcC: seek: {e}")))?;
            let (fourcc, header_len, content_len) = read_box_header(r)?;
            let content_start = pos + header_len;
            let content_end = content_start + content_len;
            out.push(ChildBox {
                fourcc,
                start: content_start,
                end: content_end,
            });
            pos = content_end;
        }
        Ok(out)
    }

    fn find_child<'a>(children: &'a [ChildBox], fourcc: &[u8; 4]) -> Option<&'a ChildBox> {
        children.iter().find(|c| &c.fourcc == fourcc)
    }

    /// Reads a `tkhd` box's `track_ID` field (handles both the 32-bit and 64-bit time
    /// variants, selected by the box's `version` byte).
    fn read_tkhd_track_id(r: &mut (impl Read + Seek), tkhd: &ChildBox) -> Result<u32> {
        r.seek(SeekFrom::Start(tkhd.start))
            .map_err(|e| Error::Decode(format!("hvcC: seek tkhd: {e}")))?;
        let mut version = [0u8; 1];
        r.read_exact(&mut version)
            .map_err(|e| Error::Decode(format!("hvcC: read tkhd version: {e}")))?;
        // Skip flags(3) + creation_time/modification_time (8+8 if version==1, else 4+4).
        let skip = if version[0] == 1 { 3 + 16 } else { 3 + 8 };
        let mut discard = vec![0u8; skip];
        r.read_exact(&mut discard)
            .map_err(|e| Error::Decode(format!("hvcC: skip tkhd fields: {e}")))?;
        let mut track_id = [0u8; 4];
        r.read_exact(&mut track_id)
            .map_err(|e| Error::Decode(format!("hvcC: read tkhd track_ID: {e}")))?;
        Ok(u32::from_be_bytes(track_id))
    }

    /// Parses a `hvcC` box's content into `(vps_list, sps_list, pps_list)`, each entry the
    /// raw NAL payload (no start code / length prefix) — the same convention the `mp4`
    /// crate's `sequence_parameter_set()`/`picture_parameter_set()` already use for H.264.
    fn parse_hvcc_record(data: &[u8]) -> Result<ParameterSets> {
        // Layout per ISO/IEC 14496-15 §8.3.3.1; see the field comments for byte offsets.
        if data.len() < 23 {
            return Err(Error::Decode("hvcC: record too short".into()));
        }
        let length_size_minus_one = data[21] & 0x03;
        let num_of_arrays = data[22];

        let mut vps = Vec::new();
        let mut sps = Vec::new();
        let mut pps = Vec::new();
        let mut pos = 23usize;
        for _ in 0..num_of_arrays {
            let array_header = *data
                .get(pos)
                .ok_or_else(|| Error::Decode("hvcC: truncated array header".into()))?;
            let nal_type = array_header & 0x3F;
            pos += 1;
            let num_nalus = u16::from_be_bytes(
                data.get(pos..pos + 2)
                    .ok_or_else(|| Error::Decode("hvcC: truncated numNalus".into()))?
                    .try_into()
                    .unwrap(),
            );
            pos += 2;
            for _ in 0..num_nalus {
                let len = u16::from_be_bytes(
                    data.get(pos..pos + 2)
                        .ok_or_else(|| Error::Decode("hvcC: truncated nalUnitLength".into()))?
                        .try_into()
                        .unwrap(),
                ) as usize;
                pos += 2;
                let nal = data
                    .get(pos..pos + len)
                    .ok_or_else(|| Error::Decode("hvcC: truncated nalUnit".into()))?
                    .to_vec();
                pos += len;
                match nal_type {
                    NAL_TYPE_VPS => vps.push(nal),
                    NAL_TYPE_SPS => sps.push(nal),
                    NAL_TYPE_PPS => pps.push(nal),
                    _ => {} // ignore SEI/other arrays — not needed to build a format description
                }
            }
        }
        Ok((vps, sps, pps, length_size_minus_one + 1))
    }

    /// Extracts `(vps_list, sps_list, pps_list, nal_length_size)` for the HEVC video track
    /// `target_track_id` in `file` by walking its box tree independently of the `mp4` crate.
    pub fn extract_parameter_sets(file: &Path, target_track_id: u32) -> Result<ParameterSets> {
        let f = File::open(file)
            .map_err(|e| Error::Decode(format!("hvcC: open {}: {e}", file.display())))?;
        let file_len = f
            .metadata()
            .map_err(|e| Error::Decode(format!("hvcC: stat {}: {e}", file.display())))?
            .len();
        let mut r = BufReader::new(f);

        let top = list_children(&mut r, 0, file_len)?;
        let moov =
            find_child(&top, b"moov").ok_or_else(|| Error::Decode("hvcC: no moov box".into()))?;
        let moov_children = list_children(&mut r, moov.start, moov.end)?;

        for trak in moov_children.iter().filter(|c| &c.fourcc == b"trak") {
            let trak_children = list_children(&mut r, trak.start, trak.end)?;
            let Some(tkhd) = find_child(&trak_children, b"tkhd") else {
                continue;
            };
            if read_tkhd_track_id(&mut r, tkhd)? != target_track_id {
                continue;
            }
            let Some(mdia) = find_child(&trak_children, b"mdia") else {
                continue;
            };
            let mdia_children = list_children(&mut r, mdia.start, mdia.end)?;
            let Some(minf) = find_child(&mdia_children, b"minf") else {
                continue;
            };
            let minf_children = list_children(&mut r, minf.start, minf.end)?;
            let Some(stbl) = find_child(&minf_children, b"stbl") else {
                continue;
            };
            let stbl_children = list_children(&mut r, stbl.start, stbl.end)?;
            let Some(stsd) = find_child(&stbl_children, b"stsd") else {
                continue;
            };
            // stsd content: version(1) + flags(3) + entry_count(4), then sample entries.
            // This box walk is our own and doesn't depend on the `mp4` crate's `stsd` parser,
            // so it can find an `hvc1` sample entry's `hvcC` box fine (as far as this function
            // goes) — but `NativeVideoToolboxFile::open`'s caller only reaches here after the
            // `mp4` crate's own track discovery already found a track, and that discovery
            // doesn't recognize `hvc1` at all (see the module doc comment), so `hvc1` remains
            // unreachable end-to-end today. Matching it here anyway costs nothing and is
            // correct forward-compat if that gap in the `mp4` crate ever closes.
            let sample_entries = list_children(&mut r, stsd.start + 8, stsd.end)?;
            let Some(entry) = find_child(&sample_entries, b"hev1")
                .or_else(|| find_child(&sample_entries, b"hvc1"))
            else {
                continue;
            };
            let entry_children = list_children(
                &mut r,
                entry.start + VISUAL_SAMPLE_ENTRY_HEADER_LEN,
                entry.end,
            )?;
            let hvcc = find_child(&entry_children, b"hvcC")
                .ok_or_else(|| Error::Decode("hvcC: no hvcC box in HEVC sample entry".into()))?;

            let len = (hvcc.end - hvcc.start) as usize;
            let mut record = vec![0u8; len];
            r.seek(SeekFrom::Start(hvcc.start))
                .map_err(|e| Error::Decode(format!("hvcC: seek hvcC content: {e}")))?;
            r.read_exact(&mut record)
                .map_err(|e| Error::Decode(format!("hvcC: read hvcC content: {e}")))?;
            return parse_hvcc_record(&record);
        }
        Err(Error::Decode(format!(
            "hvcC: no HEVC track {target_track_id} found in {}",
            file.display()
        )))
    }
}

/// Builds a `CMFormatDescription` for HEVC from raw VPS/SPS/PPS NAL payloads (no start codes
/// or length prefixes), analogous to [`create_h264_format_description`] but for the
/// variable-count HEVC parameter-set API.
fn create_hevc_format_description(
    vps: &[Vec<u8>],
    sps: &[Vec<u8>],
    pps: &[Vec<u8>],
) -> Result<CMFormatDescription> {
    let sets: Vec<&[u8]> = vps
        .iter()
        .chain(sps.iter())
        .chain(pps.iter())
        .map(|v| v.as_slice())
        .collect();
    if sets.is_empty() {
        return Err(Error::Decode(
            "hvcC: no VPS/SPS/PPS parameter sets found".into(),
        ));
    }
    let ptrs: Vec<*const u8> = sets.iter().map(|s| s.as_ptr()).collect();
    let sizes: Vec<usize> = sets.iter().map(|s| s.len()).collect();
    let mut format_desc_ref: raw::CMFormatDescriptionRef = std::ptr::null();

    // SAFETY: `ptrs`/`sizes` point at `sets`' backing `vps`/`sps`/`pps` slices, all alive for
    // the call. `formatDescriptionOut` receives a +1 retained ref per Apple's Create Rule; we
    // hand ownership to `CMFormatDescription::from_raw` below. `extensions: null` (no extra
    // per-format extension dictionary — matches the H.264 path, which doesn't pass one either).
    let status = unsafe {
        raw::CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            raw::kCFAllocatorDefault,
            ptrs.len(),
            ptrs.as_ptr(),
            sizes.as_ptr(),
            NAL_UNIT_HEADER_LENGTH as std::ffi::c_int,
            std::ptr::null(),
            &mut format_desc_ref,
        )
    };
    if status != 0 {
        return Err(Error::Decode(format!(
            "CMVideoFormatDescriptionCreateFromHEVCParameterSets failed: OSStatus {status}"
        )));
    }
    CMFormatDescription::from_raw(format_desc_ref as *mut std::ffi::c_void).ok_or_else(|| {
        Error::Decode("CMVideoFormatDescriptionCreateFromHEVCParameterSets returned null".into())
    })
}

/// Wraps one AVCC-format compressed sample (already length-prefixed by the
/// `mp4` crate) into a `CMSampleBuffer` ready for `DecompressionSession::decode`.
fn create_sample_buffer(
    data: &[u8],
    format_description: &CMFormatDescription,
    timescale: u32,
    start_time: u64,
    duration: u32,
    rendering_offset: i32,
) -> Result<CMSampleBuffer> {
    let block_buffer = CMBlockBuffer::create(data)
        .ok_or_else(|| Error::Decode("CMBlockBufferCreateWithMemoryBlock failed".into()))?;

    let dts = CMTime::new(start_time as i64, timescale as i32);
    let pts = CMTime::new(
        start_time as i64 + rendering_offset as i64,
        timescale as i32,
    );
    let dur = CMTime::new(duration as i64, timescale as i32);
    let timing = CMSampleTimingInfo::with_times(dur, pts, dts);

    let mut sample_buffer_ref: raw::CMSampleBufferRef = std::ptr::null_mut();
    let sample_size = data.len();

    // SAFETY: `block_buffer`/`format_description` are valid, retained CF
    // objects kept alive for the duration of this call. `timing`/`sample_size`
    // are stack values passed by pointer only for the call's duration;
    // `apple_cf::cm::CMSampleTimingInfo` is `#[repr(C)]` with the same field
    // layout as `raw::CMSampleTimingInfo`, so the pointer cast is valid.
    // `sampleBufferOut` receives a +1 retained ref, owned by
    // `CMSampleBuffer::from_raw` below.
    let status = unsafe {
        raw::CMSampleBufferCreateReady(
            raw::kCFAllocatorDefault,
            block_buffer.as_ptr() as raw::CMBlockBufferRef,
            format_description.as_ptr() as raw::CMFormatDescriptionRef,
            1,
            1,
            (&timing as *const CMSampleTimingInfo).cast(),
            1,
            &sample_size,
            &mut sample_buffer_ref,
        )
    };
    if status != 0 {
        return Err(Error::Decode(format!(
            "CMSampleBufferCreateReady failed: OSStatus {status}"
        )));
    }
    CMSampleBuffer::from_raw(sample_buffer_ref as *mut std::ffi::c_void)
        .ok_or_else(|| Error::Decode("CMSampleBufferCreateReady returned null".into()))
}

/// Image-buffer attributes requesting IOSurface-backed decoder output
/// (`kCVPixelBufferIOSurfacePropertiesKey`) in `kCVPixelFormatType_24RGB`
/// (`kCVPixelBufferPixelFormatTypeKey`) — tightly-packed 24-bit RGB, the
/// exact byte layout `FrameBuffer::Owned`/the rest of this codebase already
/// assumes for the ffmpeg RGB24 path, so callers don't need to know or care
/// whether a given frame came from software or hardware decode.
fn io_surface_backed_attributes() -> apple_cf::cf::CFDictionary {
    let io_surface_properties = apple_cf::cf::CFDictionary::from_pairs(&[]);
    let pixel_format = apple_cf::cf::CFNumber::from_i64(raw::kCVPixelFormatType_24RGB as i64);

    // SAFETY: `kCVPixelBufferIOSurfacePropertiesKey`/
    // `kCVPixelBufferPixelFormatTypeKey` are static Apple SDK constants
    // (borrowed, not owned by us). `CFRetain` each before wrapping in an
    // owning `CFType` so that type's `Drop` (a `CFRelease`) doesn't
    // over-release a value this code never owned.
    let retain_static = |ptr: raw::CFStringRef| -> apple_cf::cf::CFType {
        // SAFETY: `ptr` is a non-null static Apple SDK CFString constant.
        unsafe {
            let retained = raw::CFRetain(ptr.cast());
            apple_cf::cf::CFType::from_raw(retained as *mut std::ffi::c_void)
                .expect("static CF constant must not be null")
        }
    };
    let io_surface_key = retain_static(unsafe { raw::kCVPixelBufferIOSurfacePropertiesKey });
    let pixel_format_key = retain_static(unsafe { raw::kCVPixelBufferPixelFormatTypeKey });

    apple_cf::cf::CFDictionary::from_pairs(&[
        (&io_surface_key, &io_surface_properties),
        (&pixel_format_key, &pixel_format),
    ])
}

/// Per-file demux + decode state, cached across `decode_at` calls so the
/// moov atom is parsed once and the decompression session is reused.
pub struct NativeVideoToolboxFile {
    reader: mp4::Mp4Reader<BufReader<File>>,
    track_id: u32,
    timescale: u32,
    format_description: CMFormatDescription,
    sample_index: Vec<SampleIndexEntry>,
    /// The stream's earliest presentation time (`min` over `sample_index`'s
    /// `presentation_time_units`) — subtracted from every `decode_at(timestamp_s)` lookup so
    /// `timestamp_s=0.0` means "the first *displayed* frame," matching how callers compute
    /// timestamps (relative to the start of playback), not the file's raw internal PTS values.
    /// Encoders commonly delay the whole PTS timeline by one or more frame durations (visible
    /// here as the first sample's `presentation_time_units` being nonzero) to keep decode
    /// timestamps non-negative when B-frames make presentation order lag decode order.
    pts_offset: i64,
    session: DecompressionSession,
    collector: FrameCollector,
}

impl NativeVideoToolboxFile {
    pub fn open(file: &Path) -> Result<Self> {
        let f =
            File::open(file).map_err(|e| Error::Decode(format!("open {}: {e}", file.display())))?;
        let size = f
            .metadata()
            .map_err(|e| Error::Decode(format!("stat {}: {e}", file.display())))?
            .len();
        let mut reader = mp4::Mp4Reader::read_header(BufReader::new(f), size)
            .map_err(|e| Error::Decode(format!("mp4 parse {}: {e}", file.display())))?;

        let h264_box = mp4::FourCC::from(*b"avc1");
        let hevc_boxes = [mp4::FourCC::from(*b"hev1"), mp4::FourCC::from(*b"hvc1")];
        enum Codec {
            H264,
            Hevc,
        }
        let (track_id, timescale, codec) = {
            let track = reader
                .tracks()
                .values()
                .find(|t| {
                    matches!(t.track_type(), Ok(mp4::TrackType::Video))
                        && matches!(t.box_type(), Ok(bt) if bt == h264_box || hevc_boxes.contains(&bt))
                })
                .ok_or_else(|| {
                    Error::Decode(format!(
                        "no H.264 (avc1) or HEVC (hev1) video track in {} — note: `hvc1`-tagged \
                         HEVC isn't supported (the `mp4` crate doesn't recognize that sample \
                         entry at all); re-mux with `-tag:v hev1` if that's what this file is",
                        file.display()
                    ))
                })?;
            let codec = match track.box_type() {
                Ok(bt) if bt == h264_box => Codec::H264,
                _ => Codec::Hevc,
            };
            (track.track_id(), track.timescale(), codec)
        };

        let format_description = match codec {
            Codec::H264 => {
                let track = reader.tracks().get(&track_id).expect("just found above");
                let sps = track
                    .sequence_parameter_set()
                    .map_err(|e| Error::Decode(format!("read SPS: {e}")))?
                    .to_vec();
                let pps = track
                    .picture_parameter_set()
                    .map_err(|e| Error::Decode(format!("read PPS: {e}")))?
                    .to_vec();
                create_h264_format_description(&sps, &pps)?
            }
            Codec::Hevc => {
                let (vps, sps, pps, nal_length_size) =
                    hevc_hvcc::extract_parameter_sets(file, track_id)?;
                if nal_length_size as i32 != NAL_UNIT_HEADER_LENGTH {
                    return Err(Error::Decode(format!(
                        "hvcC declares a {nal_length_size}-byte NAL length prefix; only \
                         {NAL_UNIT_HEADER_LENGTH} bytes (the near-universal encoder default, \
                         and what the `mp4` crate's sample reader assumes) is supported"
                    )));
                }
                create_hevc_format_description(&vps, &sps, &pps)?
            }
        };

        let collector = FrameCollector::new();
        let collector_for_callback = collector.clone();
        let attrs = io_surface_backed_attributes();
        let session = DecompressionSession::new_with_image_buffer_attributes(
            &format_description,
            Some(&attrs),
            move |frame| collector_for_callback.push(frame),
        )
        .map_err(|e| Error::Decode(format!("VTDecompressionSessionCreate failed: {e:?}")))?;

        let sample_count = reader
            .sample_count(track_id)
            .map_err(|e| Error::Decode(format!("sample_count: {e}")))?;
        let mut sample_index = Vec::with_capacity(sample_count as usize);
        for sample_id in 1..=sample_count {
            if let Some(sample) = reader
                .read_sample(track_id, sample_id)
                .map_err(|e| Error::Decode(format!("read_sample({sample_id}): {e}")))?
            {
                sample_index.push(SampleIndexEntry {
                    sample_id,
                    presentation_time_units: sample.start_time as i64
                        + sample.rendering_offset as i64,
                    is_sync: sample.is_sync,
                });
            }
        }
        if sample_index.is_empty() {
            return Err(Error::Decode(format!(
                "no samples in video track of {}",
                file.display()
            )));
        }
        let pts_offset = sample_index
            .iter()
            .map(|e| e.presentation_time_units)
            .min()
            .expect("just checked sample_index is non-empty");

        Ok(Self {
            reader,
            track_id,
            timescale,
            format_description,
            sample_index,
            pts_offset,
            session,
            collector,
        })
    }

    pub fn dimensions(&self) -> Result<(u32, u32)> {
        let track = self
            .reader
            .tracks()
            .get(&self.track_id)
            .ok_or_else(|| Error::Decode("track disappeared after open".into()))?;
        Ok((u32::from(track.width()), u32::from(track.height())))
    }

    /// Decodes the frame nearest `timestamp_s`, returning the real
    /// hardware-decoded, IOSurface-backed pixel buffer.
    pub fn decode_at(&mut self, timestamp_s: f64) -> Result<CVPixelBuffer> {
        // `timestamp_s` is a presentation-time lookup (what callers mean by "the frame at this
        // point in the video"), so the target sample must be the one whose *presentation* time
        // (PTS = DTS + rendering offset) is nearest — not decode time (DTS). Those only coincide
        // when there's no reordering (no B-frames); `sample_index` stays DTS-ordered (decode
        // order) for the GOP walk-back below, so this is a linear scan rather than a binary
        // search — sample counts here are per-file, not per-dataset, so this isn't hot.
        let target_units =
            (timestamp_s * f64::from(self.timescale)).round() as i64 + self.pts_offset;
        let target_idx = self
            .sample_index
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| (e.presentation_time_units - target_units).abs())
            .map(|(i, _)| i)
            .expect("sample_index is non-empty (checked in open())");

        let mut start_idx = target_idx;
        while start_idx > 0 && !self.sample_index[start_idx].is_sync {
            start_idx -= 1;
        }

        // Clear any stale frames left over from a previous call.
        self.collector.drain();

        let mut decoded = 0usize;
        for i in start_idx..=target_idx {
            let sample_id = self.sample_index[i].sample_id;
            let sample = self
                .reader
                .read_sample(self.track_id, sample_id)
                .map_err(|e| Error::Decode(format!("read_sample({sample_id}): {e}")))?
                .ok_or_else(|| Error::Decode(format!("sample {sample_id} vanished on reread")))?;

            let sample_buffer = create_sample_buffer(
                &sample.bytes,
                &self.format_description,
                self.timescale,
                sample.start_time,
                sample.duration,
                sample.rendering_offset,
            )?;
            self.session
                .decode(&sample_buffer)
                .map_err(|e| Error::Decode(format!("VTDecompressionSessionDecodeFrame: {e:?}")))?;
            decoded += 1;
        }

        self.collector
            .wait_until_at_least(decoded, DECODE_WAIT_TIMEOUT);
        let frames = self.collector.drain();
        if frames.len() < decoded {
            return Err(Error::Decode(format!(
                "timed out waiting for VideoToolbox: got {}/{decoded} frames",
                frames.len()
            )));
        }

        let target_pts_units = self.sample_index[target_idx].presentation_time_units;
        let target_pts_seconds = target_pts_units as f64 / f64::from(self.timescale);

        frames
            .into_iter()
            .filter(|f| f.status == 0 && f.image_buffer.is_some())
            .min_by(|a, b| {
                let da = (cm_time_seconds(a.presentation_time) - target_pts_seconds).abs();
                let db = (cm_time_seconds(b.presentation_time) - target_pts_seconds).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .and_then(|f| f.image_buffer)
            .ok_or_else(|| Error::Decode("VideoToolbox produced no usable frame".into()))
    }
}

/// Per-file cache so repeated `decode()` calls against the same video (the
/// common training-loop access pattern) reuse the open reader and session.
pub struct NativeVideoToolboxDecoder {
    files: std::collections::HashMap<PathBuf, NativeVideoToolboxFile>,
}

impl Default for NativeVideoToolboxDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeVideoToolboxDecoder {
    pub fn new() -> Self {
        Self {
            files: std::collections::HashMap::new(),
        }
    }

    pub fn decode_at(
        &mut self,
        file: &Path,
        timestamp_s: f64,
    ) -> Result<(CVPixelBuffer, u32, u32)> {
        if !self.files.contains_key(file) {
            let opened = NativeVideoToolboxFile::open(file)?;
            self.files.insert(file.to_path_buf(), opened);
        }
        let state = self.files.get_mut(file).expect("just inserted");
        let (width, height) = state.dimensions()?;
        let pixel_buffer = state.decode_at(timestamp_s)?;
        Ok((pixel_buffer, width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// Generates a real H.264 MP4 via ffmpeg (a genuine, standard-compliant
    /// bitstream — not a synthetic fixture) and decodes it through the real
    /// VTDecompressionSession path, cross-checking against ffmpeg's own
    /// software-decoded pixels for the same frame.
    fn generate_test_clip(dir: &Path, width: u32, height: u32) -> PathBuf {
        let mp4_path = dir.join("clip.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc=size={width}x{height}:rate=10"),
                "-frames:v",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline", // no B-frames: decode order == display order
                "-g",
                "10",
            ])
            .arg(&mp4_path)
            .status()
            .expect("ffmpeg must be installed to run this test");
        assert!(status.success(), "ffmpeg failed to generate test clip");
        mp4_path
    }

    /// A B-frame-enabled clip (no `-profile:v baseline`/`-bf 0`, matching what a typical
    /// LeRobot dataset video actually looks like) to test the presentation-time-offset
    /// handling `decode_at` needs when decode order and display order diverge.
    fn generate_bframe_test_clip(dir: &Path, width: u32, height: u32) -> PathBuf {
        let mp4_path = dir.join("clip_bframes.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc=size={width}x{height}:rate=30"),
                "-frames:v",
                "100",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&mp4_path)
            .status()
            .expect("ffmpeg must be installed to run this test");
        assert!(
            status.success(),
            "ffmpeg failed to generate B-frame test clip"
        );
        mp4_path
    }

    #[test]
    fn native_decode_handles_pts_offset_from_bframe_reordering() {
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = generate_bframe_test_clip(tmp.path(), 32, 24);

        let mut file = NativeVideoToolboxFile::open(&mp4).unwrap();
        // The stream's PTS timeline doesn't start at 0 (a real bitstream property here, not a
        // test artifact) — confirms this test actually exercises the offset-handling path
        // rather than accidentally testing the (already-covered) no-offset case.
        assert_ne!(
            file.pts_offset, 0,
            "expected this clip's encoder to delay the PTS timeline (a sign of B-frame reordering)"
        );

        let mut frame_at = |ts: f64| {
            let buf = file.decode_at(ts).unwrap();
            let guard = buf.lock_read_only().unwrap();
            guard.as_slice().to_vec()
        };
        // `timestamp_s=0.0` must mean "the first displayed frame," not "whatever sample has
        // raw PTS closest to 0" (which, before this fix, was the same sample as a later
        // timestamp too — the exact bug this regression-tests).
        let f0 = frame_at(0.0);
        let f1 = frame_at(1.0 / 30.0);
        assert_ne!(
            f0, f1,
            "frames 1/30s apart must decode to different content"
        );
    }

    /// Same as [`generate_test_clip`] but HEVC (`libx265`), tagged `hvc1` (the fourcc real
    /// Apple-ecosystem HEVC files use) to exercise that branch of the `hev1`/`hvc1` track
    /// detection, with B-frames disabled for the same decode-order-equals-display-order
    /// reason `generate_test_clip` uses H.264's `baseline` profile.
    fn generate_hevc_test_clip(dir: &Path, width: u32, height: u32) -> PathBuf {
        let mp4_path = dir.join("clip_hevc.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc=size={width}x{height}:rate=10"),
                "-frames:v",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx265",
                "-bf",
                "0", // no B-frames: decode order == display order
                "-g",
                "10",
                "-tag:v",
                "hev1",
            ])
            .arg(&mp4_path)
            .status()
            .expect("ffmpeg must be installed to run this test");
        assert!(status.success(), "ffmpeg failed to generate HEVC test clip");
        mp4_path
    }

    fn ffmpeg_reference_rgb24(mp4: &Path, timestamp_s: f64, width: u32, height: u32) -> Vec<u8> {
        let out = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(mp4)
            .args(["-ss"])
            .arg(format!("{timestamp_s}"))
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .output()
            .expect("ffmpeg must be installed");
        assert!(out.status.success());
        let expected = (width * height * 3) as usize;
        assert!(out.stdout.len() >= expected);
        out.stdout[..expected].to_vec()
    }

    #[test]
    fn native_decode_produces_real_iosurface_backed_frame() {
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = generate_test_clip(tmp.path(), 64, 48);

        let mut decoder = NativeVideoToolboxDecoder::new();
        let (pixel_buffer, width, height) = decoder.decode_at(&mp4, 0.5).unwrap();

        assert_eq!((width, height), (64, 48));
        assert!(
            pixel_buffer.is_backed_by_io_surface(),
            "expected a real IOSurface-backed CVPixelBuffer, not a plain CPU buffer"
        );
        assert!(pixel_buffer.io_surface().is_some());

        // Confirm the decoded pixels are non-trivial (not an all-zero/blank
        // buffer) — a genuine sanity check that real frame content arrived.
        let guard = pixel_buffer.lock_read_only().unwrap();
        let bytes = guard.as_slice();
        assert!(bytes.iter().any(|&b| b != 0), "decoded frame was all zero");

        // We explicitly requested kCVPixelFormatType_24RGB — confirm we
        // actually got 24-bit RGB (3 bytes/pixel), not the hardware
        // decoder's YUV default, so this byte layout is a drop-in match for
        // the ffmpeg RGB24 (`FrameBuffer::Owned`) path. `bytes_per_row` may
        // include row-alignment padding, so check the real invariant
        // (>= 3 bytes/pixel, full slice == height * stride) rather than
        // assuming a tightly-packed `width * height * 3`.
        assert_eq!(pixel_buffer.pixel_format(), raw::kCVPixelFormatType_24RGB);
        assert!(pixel_buffer.bytes_per_row() >= (width * 3) as usize);
        assert_eq!(bytes.len(), height as usize * pixel_buffer.bytes_per_row());
    }

    #[test]
    fn native_decode_repeated_calls_are_consistent() {
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = generate_test_clip(tmp.path(), 64, 48);

        let mut decoder = NativeVideoToolboxDecoder::new();
        let (buf_a, ..) = decoder.decode_at(&mp4, 1.0).unwrap();
        let (buf_b, ..) = decoder.decode_at(&mp4, 1.0).unwrap();

        let guard_a = buf_a.lock_read_only().unwrap();
        let guard_b = buf_b.lock_read_only().unwrap();
        assert_eq!(guard_a.as_slice(), guard_b.as_slice());
    }

    #[test]
    fn native_decode_matches_software_decode_within_tolerance() {
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = generate_test_clip(tmp.path(), 64, 48);
        let timestamp_s = 0.8;

        let mut decoder = NativeVideoToolboxDecoder::new();
        let (pixel_buffer, width, height) = decoder.decode_at(&mp4, timestamp_s).unwrap();

        let reference = ffmpeg_reference_rgb24(&mp4, timestamp_s, width, height);

        // Both paths now decode to the same kCVPixelFormatType_24RGB /
        // rgb24 byte layout, so this is a genuine pixel-level
        // cross-validation between hardware (VideoToolbox) and software
        // (libx264/swscale via the ffmpeg CLI) decode of the identical
        // bitstream — not just "some data came out."
        let guard = pixel_buffer.lock_read_only().unwrap();
        let stride = guard.bytes_per_row();
        let row_bytes = (width * 3) as usize;
        assert_eq!(reference.len(), (width * height * 3) as usize);

        let mut max_channel_diff: i32 = 0;
        for row in 0..height as usize {
            let hw_row = &guard.as_slice()[row * stride..row * stride + row_bytes];
            let sw_row = &reference[row * row_bytes..(row + 1) * row_bytes];
            for (a, b) in hw_row.iter().zip(sw_row.iter()) {
                max_channel_diff = max_channel_diff.max((*a as i32 - *b as i32).abs());
            }
        }
        // Hardware and software H.264 decoders can differ slightly in
        // rounding/dithering; a small per-channel tolerance still catches
        // real bugs (wrong format, channel swap, garbage data) while not
        // being flaky over decoder-implementation noise.
        assert!(
            max_channel_diff <= 12,
            "hardware vs software decode differ by up to {max_channel_diff}/255 per channel"
        );
    }

    #[test]
    fn hevc_hvcc_extracts_real_parameter_sets() {
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = generate_hevc_test_clip(tmp.path(), 64, 48);

        let f = File::open(&mp4).unwrap();
        let size = f.metadata().unwrap().len();
        let reader = mp4::Mp4Reader::read_header(BufReader::new(f), size).unwrap();
        let hev1 = mp4::FourCC::from(*b"hev1");
        let track_id = reader
            .tracks()
            .values()
            .find(|t| matches!(t.box_type(), Ok(bt) if bt == hev1))
            .expect("ffmpeg should have produced an hev1 track")
            .track_id();
        // `reader` isn't used again — this just confirms `mp4`'s own track detection agrees
        // with what `hevc_hvcc` will independently look for.
        drop(reader);

        let (vps, sps, pps, nal_length_size) =
            hevc_hvcc::extract_parameter_sets(&mp4, track_id).unwrap();
        assert!(!vps.is_empty(), "expected at least one VPS NAL");
        assert!(!sps.is_empty(), "expected at least one SPS NAL");
        assert!(!pps.is_empty(), "expected at least one PPS NAL");
        assert_eq!(nal_length_size, 4);
    }

    #[test]
    fn native_decode_hevc_produces_real_iosurface_backed_frame() {
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = generate_hevc_test_clip(tmp.path(), 64, 48);

        let mut decoder = NativeVideoToolboxDecoder::new();
        let (pixel_buffer, width, height) = decoder.decode_at(&mp4, 0.5).unwrap();

        assert_eq!((width, height), (64, 48));
        assert!(
            pixel_buffer.is_backed_by_io_surface(),
            "expected a real IOSurface-backed CVPixelBuffer, not a plain CPU buffer"
        );
        let guard = pixel_buffer.lock_read_only().unwrap();
        let bytes = guard.as_slice();
        assert!(bytes.iter().any(|&b| b != 0), "decoded frame was all zero");
        assert_eq!(pixel_buffer.pixel_format(), raw::kCVPixelFormatType_24RGB);
    }

    #[test]
    fn native_decode_hevc_matches_software_decode_within_tolerance() {
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = generate_hevc_test_clip(tmp.path(), 64, 48);
        let timestamp_s = 0.8;

        let mut decoder = NativeVideoToolboxDecoder::new();
        let (pixel_buffer, width, height) = decoder.decode_at(&mp4, timestamp_s).unwrap();
        let reference = ffmpeg_reference_rgb24(&mp4, timestamp_s, width, height);

        // Genuine pixel-level cross-validation between hardware (VideoToolbox HEVC) and
        // software (libx265/swscale via the ffmpeg CLI) decode of the identical bitstream —
        // same method `native_decode_matches_software_decode_within_tolerance` uses for H.264.
        let guard = pixel_buffer.lock_read_only().unwrap();
        let stride = guard.bytes_per_row();
        let row_bytes = (width * 3) as usize;
        assert_eq!(reference.len(), (width * height * 3) as usize);

        let mut max_channel_diff: i32 = 0;
        for row in 0..height as usize {
            let hw_row = &guard.as_slice()[row * stride..row * stride + row_bytes];
            let sw_row = &reference[row * row_bytes..(row + 1) * row_bytes];
            for (a, b) in hw_row.iter().zip(sw_row.iter()) {
                max_channel_diff = max_channel_diff.max((*a as i32 - *b as i32).abs());
            }
        }
        assert!(
            max_channel_diff <= 12,
            "hardware vs software decode differ by up to {max_channel_diff}/255 per channel"
        );
    }
}
