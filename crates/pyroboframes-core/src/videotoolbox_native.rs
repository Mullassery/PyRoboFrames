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
//! - H.264 (`avc1`) only. HEVC parameter-set extraction (VPS+SPS+PPS,
//!   different NAL header handling) isn't implemented here.
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
    start_time: u64,
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
        let (track_id, timescale, sps, pps) = {
            let track = reader
                .tracks()
                .values()
                .find(|t| {
                    matches!(t.track_type(), Ok(mp4::TrackType::Video))
                        && matches!(t.box_type(), Ok(bt) if bt == h264_box)
                })
                .ok_or_else(|| {
                    Error::Decode(format!("no H.264 (avc1) video track in {}", file.display()))
                })?;
            let sps = track
                .sequence_parameter_set()
                .map_err(|e| Error::Decode(format!("read SPS: {e}")))?
                .to_vec();
            let pps = track
                .picture_parameter_set()
                .map_err(|e| Error::Decode(format!("read PPS: {e}")))?
                .to_vec();
            (track.track_id(), track.timescale(), sps, pps)
        };

        let format_description = create_h264_format_description(&sps, &pps)?;

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
                    start_time: sample.start_time,
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

        Ok(Self {
            reader,
            track_id,
            timescale,
            format_description,
            sample_index,
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
        let target_units = (timestamp_s * f64::from(self.timescale)).round().max(0.0) as u64;

        let target_idx = match self
            .sample_index
            .binary_search_by_key(&target_units, |e| e.start_time)
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };

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

        let target_pts_units = self.sample_index[target_idx].start_time;
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
}
