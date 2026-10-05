//! Video decode: the `Decoder` trait, platform backend selection, a decoded-frame LRU cache,
//! and a frame-buffer pool.
//!
//! The actual hardware backends — VideoToolbox on macOS, FFmpeg (VAAPI/NVDEC + software) on
//! Linux — are gated behind the `videotoolbox` / `ffmpeg` cargo features and are stubbed
//! pending the Phase 0 spikes (real HW decode needs platform crates + video fixtures). Every
//! platform-agnostic piece here — the trait, the batched-seek default, the cache (Robo-DM's
//! biggest lever), and the buffer pool — is implemented and tested.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lru::LruCache;

use crate::Result;

/// Pixel storage for a decoded frame.
#[derive(Debug, Clone)]
pub enum FrameBuffer {
    /// Row-major `H×W×C` 8-bit pixels owned on the heap (software / FFmpeg path). Wrapped in an
    /// `Arc` so the cache and consumers can share a frame without a deep copy.
    Owned { data: Arc<Vec<u8>>, channels: u8 },
    /// IOSurface-backed buffer for zero-copy hand-off to MLX/Metal (macOS VideoToolbox path).
    ///
    /// This is a *real* `CVPixelBuffer` produced directly by
    /// `VTDecompressionSession` (see `videotoolbox_native`), backed by an
    /// IOSurface — not a copy through ffmpeg's stdout pipe. `Arc` because
    /// the frame cache clones `Frame`s freely; the `CVPixelBuffer` wrapper's
    /// own `Drop` impl handles the CFRelease.
    ///
    /// **Zero-copy hand-off flow:**
    /// 1. VideoToolbox decodes to `CVPixelBuffer` (IOSurface-backed) — done.
    /// 2. Rust wraps it here.
    /// 3. Python's `__dlpack__`/`__dlpack_device__` (see `pyroboframes-py`)
    ///    exports it via the DLPack protocol for CPU-side zero-copy access.
    /// 4. `mx.array(frame)` (or any DLPack-consuming array library) reads
    ///    straight from the IOSurface's locked base address — no copy.
    ///
    /// GPU-resident (Metal-device) DLPack export — skipping even the CPU
    /// lock/unlock — is a further optimization not implemented here.
    #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
    IOSurface {
        pixel_buffer: Arc<apple_cf::cv::CVPixelBuffer>,
    },
}

impl FrameBuffer {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            FrameBuffer::Owned { data, .. } => data,
            #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
            FrameBuffer::IOSurface { .. } => {
                panic!(
                    "IOSurface buffers require a lock scope; use with_locked_bytes() or the \
                     DLPack export instead of as_bytes()"
                )
            }
        }
    }

    pub fn channels(&self) -> u8 {
        match self {
            FrameBuffer::Owned { channels, .. } => *channels,
            #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
            FrameBuffer::IOSurface { .. } => 3,
        }
    }

    /// Safe CPU-side access to an IOSurface-backed buffer's pixel data,
    /// locking it for the duration of `f`. Returns `None` for `Owned`
    /// buffers (use [`as_bytes`](Self::as_bytes) directly for those).
    #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
    pub fn with_locked_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        match self {
            FrameBuffer::IOSurface { pixel_buffer } => {
                let guard = pixel_buffer.lock_read_only().ok()?;
                Some(f(guard.as_slice()))
            }
            FrameBuffer::Owned { .. } => None,
        }
    }

    /// Extract the underlying IOSurface's raw pointer for direct GPU/Metal
    /// access (macOS only). Returns `None` for `Owned` buffers or if the
    /// pixel buffer isn't actually IOSurface-backed.
    #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
    pub fn as_iosurface(&self) -> Option<u64> {
        match self {
            FrameBuffer::IOSurface { pixel_buffer } => {
                pixel_buffer.io_surface().map(|s| s.as_ptr() as u64)
            }
            FrameBuffer::Owned { .. } => None,
        }
    }
}

/// A decoded video frame.
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub camera: String,
    pub timestamp: f64,
    pub pixels: FrameBuffer,
}

impl Frame {
    /// Tightly-packed RGB24 bytes (`width * height * 3`, no row padding)
    /// regardless of which `FrameBuffer` variant backs this frame. `Owned`
    /// is already tightly packed and copies directly; `IOSurface` locks the
    /// buffer and strips any row-stride padding along the way. Prefer
    /// `pixels.as_bytes()` (zero-copy) or `pixels.with_locked_bytes()` when
    /// the caller can deal with either variant directly — this method
    /// exists for call sites (e.g. batch-array construction) that need one
    /// uniform tightly-packed layout no matter the source.
    pub fn to_rgb24_bytes(&self) -> Vec<u8> {
        let row_bytes = self.width as usize * 3;
        let mut out = vec![0u8; row_bytes * self.height as usize];
        self.write_rgb24_into(&mut out);
        out
    }

    /// Write tightly-packed RGB24 bytes (`width * height * 3`) directly into `dst`, which must
    /// be exactly that length. Same normalization as [`to_rgb24_bytes`](Self::to_rgb24_bytes)
    /// (`Owned` copied as-is, `IOSurface` locked and stripped of row-stride padding), but without
    /// an intermediate per-frame allocation — lets a batch assembler copy straight from the
    /// decoded frame into its slot in the combined batch array instead of copying once into a
    /// throwaway `Vec` and again into the batch array.
    pub fn write_rgb24_into(&self, dst: &mut [u8]) {
        let row_bytes = self.width as usize * 3;
        debug_assert_eq!(dst.len(), row_bytes * self.height as usize);
        match &self.pixels {
            FrameBuffer::Owned { data, .. } => {
                let n = dst.len().min(data.len());
                dst[..n].copy_from_slice(&data[..n]);
            }
            #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
            FrameBuffer::IOSurface { pixel_buffer } => {
                if let Ok(guard) = pixel_buffer.lock_read_only() {
                    let stride = guard.bytes_per_row();
                    let slice = guard.as_slice();
                    for row in 0..self.height as usize {
                        let src_start = row * stride;
                        let dst_start = row * row_bytes;
                        if src_start + row_bytes <= slice.len() {
                            dst[dst_start..dst_start + row_bytes]
                                .copy_from_slice(&slice[src_start..src_start + row_bytes]);
                        }
                    }
                }
            }
        }
    }
}

/// A hardware (or software) video decoder, selected per platform (see [`Backend`]).
pub trait Decoder: Send {
    /// Decode the frame of `camera` in `file` nearest `timestamp` (seconds).
    fn decode(&mut self, camera: &str, file: &Path, timestamp: f64) -> Result<Frame>;

    /// Decode several timestamps from one video at once. The default decodes one-by-one.
    /// **The native macOS `VideoToolboxDecoder` overrides this** (as of 2026-09-29) with
    /// real ordered-seek, GOP-reuse decoding — see
    /// `NativeVideoToolboxFile::decode_batch_at`. The original ~14x-slower-than-`lerobot`
    /// finding (verified 2026-09-27 via real-world benchmarking against `lerobot/pusht`)
    /// applied to every backend at the time; the ffmpeg-subprocess fallback
    /// `VideoToolboxDecoder` and other backends still use this default. See
    /// `ROADMAP_HONEST.md`.
    fn decode_batch(
        &mut self,
        camera: &str,
        file: &Path,
        timestamps: &[f64],
    ) -> Result<Vec<Frame>> {
        timestamps
            .iter()
            .map(|&t| self.decode(camera, file, t))
            .collect()
    }
}

/// The decode backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Apple Media Engine (macOS).
    VideoToolbox,
    /// NVIDIA NVDEC + CUDA (Linux with CUDA libraries present).
    Cuda,
    /// FFmpeg — VAAPI/NVDEC where available, software otherwise (Linux default).
    Ffmpeg,
    /// Pure software decode (portable fallback).
    Software,
}

impl Backend {
    /// The preferred backend for the current build target.
    /// - macOS: [`Backend::VideoToolbox`] if the `videotoolbox` feature (native
    ///   `VTDecompressionSession`) or the `ffmpeg`/`cuda` fallback is available, else Software.
    /// - Linux: [`Backend::Cuda`] if compiled with `--features cuda`; else [`Backend::Ffmpeg`].
    ///
    /// Real *runtime* auto-detection (probe the GPU, fall back to Software) is a future enhancement.
    pub fn preferred() -> Backend {
        if cfg!(target_os = "macos") {
            #[cfg(any(feature = "videotoolbox", feature = "ffmpeg", feature = "cuda"))]
            {
                Backend::VideoToolbox
            }
            #[cfg(not(any(feature = "videotoolbox", feature = "ffmpeg", feature = "cuda")))]
            {
                Backend::Software
            }
        } else if cfg!(feature = "cuda") {
            Backend::Cuda
        } else if cfg!(feature = "ffmpeg") {
            Backend::Ffmpeg
        } else {
            Backend::Software
        }
    }
}

/// A decoded-frame LRU cache wrapping a decoder. Shuffled, multi-epoch training re-requests the
/// same frames; caching avoids re-decoding them — the single biggest lever in Robo-DM's ~50×
/// speedup. Capacity is measured in frames.
pub struct FrameCache {
    lru: LruCache<FrameKey, Frame>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FrameKey {
    camera: String,
    file: PathBuf,
    /// Timestamp quantized to microseconds so an `f64` can serve as a hash key.
    ts_micros: i64,
}

impl FrameKey {
    fn new(camera: &str, file: &Path, timestamp: f64) -> Self {
        Self {
            camera: camera.to_string(),
            file: file.to_path_buf(),
            ts_micros: (timestamp * 1e6).round() as i64,
        }
    }
}

impl FrameCache {
    /// Create a cache holding up to `capacity_frames` frames (minimum 1).
    pub fn new(capacity_frames: usize) -> Self {
        let cap = NonZeroUsize::new(capacity_frames.max(1)).expect("capacity >= 1");
        Self {
            lru: LruCache::new(cap),
        }
    }

    pub fn len(&self) -> usize {
        self.lru.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lru.is_empty()
    }

    /// Return the cached frame, or decode it with `decoder` and cache it.
    pub fn get_or_decode(
        &mut self,
        decoder: &mut dyn Decoder,
        camera: &str,
        file: &Path,
        timestamp: f64,
    ) -> Result<Frame> {
        let key = FrameKey::new(camera, file, timestamp);
        if let Some(frame) = self.lru.get(&key) {
            return Ok(frame.clone());
        }
        let frame = decoder.decode(camera, file, timestamp)?;
        self.lru.put(key, frame.clone());
        Ok(frame)
    }
}

/// Recycles pixel buffers to avoid per-frame heap allocation in the decode hot path.
pub struct FramePool {
    free: Vec<Vec<u8>>,
    buf_len: usize,
}

impl FramePool {
    pub fn new(buf_len: usize) -> Self {
        Self {
            free: Vec::new(),
            buf_len,
        }
    }

    /// Take a zeroed buffer of `buf_len` bytes, reusing a freed one when available.
    pub fn take(&mut self) -> Vec<u8> {
        match self.free.pop() {
            Some(mut b) => {
                b.clear();
                b.resize(self.buf_len, 0);
                b
            }
            None => vec![0u8; self.buf_len],
        }
    }

    /// Return a buffer for reuse.
    pub fn give(&mut self, buf: Vec<u8>) {
        self.free.push(buf);
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }
}

// --- Hardware backends (stubs pending Phase 0 spikes) -------------------------------------

/// macOS VideoToolbox decoder: real, in-process `VTDecompressionSession`
/// hardware decode producing IOSurface-backed frames (see
/// `crate::videotoolbox_native`) — not a shell-out to the `ffmpeg` CLI.
#[cfg(all(target_os = "macos", feature = "videotoolbox"))]
pub use macos::VideoToolboxDecoder;

#[cfg(all(target_os = "macos", feature = "videotoolbox"))]
mod macos {
    use super::*;
    use crate::videotoolbox_native::NativeVideoToolboxDecoder;

    /// macOS VideoToolbox decoder using Apple's Media Engine for H.264
    /// hardware decode, driven directly (no ffmpeg subprocess): the decoded
    /// `CVPixelBuffer` is IOSurface-backed and stays in-process, enabling
    /// real zero-copy hand-off (see `FrameBuffer::IOSurface`).
    ///
    /// **Real AV1 fallback (2026-09-29):** the native `VTDecompressionSession`
    /// path above is H.264/HEVC-specific -- NAL-unit parsing, AVCC/HVCC
    /// `CMFormatDescription` construction -- none of which applies to AV1's
    /// completely different OBU-based bitstream. Every current real-world
    /// LeRobot v3.0 dataset ships AV1 by default (see `ROADMAP_HONEST.md`),
    /// so being unable to open those files at all was a hard blocker, not
    /// just a missed optimization. Rather than reimplementing a second,
    /// from-scratch native hardware-decode path for a different codec,
    /// real AV1 decode is delegated to `ffmpeg` (built with a real AV1
    /// decoder -- `dav1d` -- on this project's target platforms), the same
    /// subprocess pattern `macos_ffmpeg_fallback` already uses. This trades
    /// zero-copy IOSurface hand-off for AV1 files specifically (they land
    /// in `FrameBuffer::Owned` instead) in exchange for being able to open
    /// them at all; H.264/HEVC files are completely unaffected and keep
    /// the fast, zero-copy native path. Requires the `ffmpeg` feature,
    /// which the real published wheel already enables alongside
    /// `videotoolbox` (see `pyproject.toml`).
    #[derive(Default)]
    pub struct VideoToolboxDecoder {
        native: NativeVideoToolboxDecoder,
        #[cfg(any(feature = "ffmpeg", feature = "cuda"))]
        av1_dims: std::collections::HashMap<PathBuf, (u32, u32)>,
    }

    #[cfg(any(feature = "ffmpeg", feature = "cuda"))]
    impl VideoToolboxDecoder {
        /// Probes the real codec of `file`'s video stream via `ffprobe`.
        /// Cheap relative to decoding, but still a real subprocess spawn --
        /// callers cache the result (`av1_dims`) rather than re-probing
        /// every frame.
        fn is_av1(file: &Path) -> bool {
            super::ffcli::probe_codec(file)
                .map(|codec| codec == "av1")
                .unwrap_or(false)
        }

        fn av1_dims(&mut self, file: &Path) -> Result<(u32, u32)> {
            if let Some(&d) = self.av1_dims.get(file) {
                return Ok(d);
            }
            let dims = super::ffcli::probe_dims(file)?;
            self.av1_dims.insert(file.to_path_buf(), dims);
            Ok(dims)
        }
    }

    impl Decoder for VideoToolboxDecoder {
        fn decode(&mut self, camera: &str, file: &Path, timestamp: f64) -> Result<Frame> {
            #[cfg(any(feature = "ffmpeg", feature = "cuda"))]
            if Self::is_av1(file) {
                let (width, height) = self.av1_dims(file)?;
                // No `-hwaccel videotoolbox`: verified empirically (even on an
                // Apple M5) that ffmpeg's VideoToolbox hwaccel path does not
                // support AV1 here ("Your platform doesn't support hardware
                // accelerated AV1 decoding") -- real AV1 hardware decode isn't
                // reliably exposed through ffmpeg's hwaccel layer even where the
                // silicon nominally supports it. Software decode via `dav1d`
                // (this project's real, working AV1 decoder) is what's actually
                // used here.
                return super::ffcli::decode_frame(camera, file, timestamp, width, height, None);
            }

            let (pixel_buffer, width, height) = self.native.decode_at(file, timestamp)?;
            Ok(Frame {
                width,
                height,
                camera: camera.to_string(),
                timestamp,
                pixels: FrameBuffer::IOSurface {
                    pixel_buffer: Arc::new(pixel_buffer),
                },
            })
        }

        /// Real GOP-reuse override: see
        /// `NativeVideoToolboxFile::decode_batch_at`. Replaces the trait's
        /// default (independent per-timestamp seeks, ~14x slower than a
        /// real competitor's dataloader on a real benchmark -- see
        /// `ROADMAP_HONEST.md`) with ordered seeks that resume decoding from
        /// the last-decoded sample instead of re-walking back to the
        /// nearest keyframe for every single timestamp. AV1 files (see
        /// `is_av1` above) fall back to the trait's default per-timestamp
        /// behavior via `self.decode(...)` in a loop -- real and correct,
        /// just not GOP-reuse-optimized, since that optimization is
        /// specific to driving `VTDecompressionSession` directly.
        fn decode_batch(
            &mut self,
            camera: &str,
            file: &Path,
            timestamps: &[f64],
        ) -> Result<Vec<Frame>> {
            #[cfg(any(feature = "ffmpeg", feature = "cuda"))]
            if Self::is_av1(file) {
                return timestamps
                    .iter()
                    .map(|&t| self.decode(camera, file, t))
                    .collect();
            }

            let decoded = self.native.decode_batch_at(file, timestamps)?;
            Ok(decoded
                .into_iter()
                .zip(timestamps.iter())
                .map(|((pixel_buffer, width, height), &timestamp)| Frame {
                    width,
                    height,
                    camera: camera.to_string(),
                    timestamp,
                    pixels: FrameBuffer::IOSurface {
                        pixel_buffer: Arc::new(pixel_buffer),
                    },
                })
                .collect())
        }
    }
}

/// Fallback macOS VideoToolbox path via `ffmpeg -hwaccel videotoolbox`
/// (CPU RGB24 output) — used when the `videotoolbox` feature (native
/// VTDecompressionSession) isn't enabled but `ffmpeg` is, so macOS builds
/// without direct framework bindings still get *some* hardware-accelerated
/// decode, just without the zero-copy IOSurface hand-off.
#[cfg(all(
    target_os = "macos",
    not(feature = "videotoolbox"),
    any(feature = "ffmpeg", feature = "cuda")
))]
pub use macos_ffmpeg_fallback::VideoToolboxDecoder;

#[cfg(all(
    target_os = "macos",
    not(feature = "videotoolbox"),
    any(feature = "ffmpeg", feature = "cuda")
))]
mod macos_ffmpeg_fallback {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    pub struct VideoToolboxDecoder {
        dims: HashMap<PathBuf, (u32, u32)>,
        frame_index: HashMap<PathBuf, Vec<ffcli::ProbedFrame>>,
        /// Real count of `ffmpeg` subprocess invocations (see
        /// `linux::FfmpegDecoder::ffmpeg_invocations` for the equivalent on
        /// the other fallback decoder) -- used by
        /// `tests::macos_fallback_decode_batch_matches_per_frame_decode_and_reduces_ffmpeg_invocations`.
        pub(crate) ffmpeg_invocations: usize,
    }

    impl VideoToolboxDecoder {
        fn dimensions(&mut self, file: &Path) -> Result<(u32, u32)> {
            if let Some(&d) = self.dims.get(file) {
                return Ok(d);
            }
            let dims = ffcli::probe_dims(file)?;
            self.dims.insert(file.to_path_buf(), dims);
            Ok(dims)
        }

        fn frame_index(&mut self, file: &Path) -> Result<&[ffcli::ProbedFrame]> {
            if !self.frame_index.contains_key(file) {
                let index = ffcli::probe_frame_index(file)?;
                self.frame_index.insert(file.to_path_buf(), index);
            }
            Ok(&self.frame_index[file])
        }

        /// The frame-index position whose real presentation time is nearest
        /// `timestamp`.
        fn nearest_sample_index(index: &[ffcli::ProbedFrame], timestamp: f64) -> usize {
            index
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a.pts_seconds - timestamp)
                        .abs()
                        .partial_cmp(&(b.pts_seconds - timestamp).abs())
                        .expect("pts_seconds is always finite")
                })
                .map(|(i, _)| i)
                .expect("frame_index is non-empty (probe_frame_index errors on empty)")
        }

        /// Walks back from `target_idx` to the nearest preceding (or equal)
        /// keyframe -- the earliest frame `ffmpeg` needs to start decoding
        /// from to correctly produce `target_idx`.
        fn keyframe_walkback(index: &[ffcli::ProbedFrame], target_idx: usize) -> usize {
            let mut start = target_idx;
            while start > 0 && !index[start].is_keyframe {
                start -= 1;
            }
            start
        }
    }

    impl Decoder for VideoToolboxDecoder {
        fn decode(&mut self, camera: &str, file: &Path, timestamp: f64) -> Result<Frame> {
            let (width, height) = self.dimensions(file)?;
            self.ffmpeg_invocations += 1;
            ffcli::decode_frame(camera, file, timestamp, width, height, Some("videotoolbox"))
        }

        /// Real GOP-reuse override, mirroring
        /// `NativeVideoToolboxFile::decode_batch_at`'s strategy but adapted
        /// for subprocess decoding (which can't keep a decoder session open
        /// across separate `ffmpeg` invocations the way `VTDecompressionSession`
        /// can across separate `decode()` calls): rather than letting the
        /// trait's default call `decode()` once per timestamp -- each of
        /// which re-decodes from the *start of the file* every time (see
        /// `decode_rgb24`'s accurate output-seek) -- this probes the real
        /// frame/keyframe index once (`frame_index`, cached per file), finds
        /// the single keyframe covering the batch's earliest request, and
        /// decodes the whole span up through the batch's latest request in
        /// one continuous `ffmpeg` subprocess (`decode_rgb24_sequential`),
        /// picking out each request's frame from that one decoded run
        /// instead of re-decoding shared GOP prefixes once per request.
        /// Verified pixel-identical to `decode()` in a loop; see
        /// `tests::macos_fallback_decode_batch_matches_per_frame_decode_and_reduces_ffmpeg_invocations`.
        fn decode_batch(
            &mut self,
            camera: &str,
            file: &Path,
            timestamps: &[f64],
        ) -> Result<Vec<Frame>> {
            if timestamps.is_empty() {
                return Ok(Vec::new());
            }

            let (width, height) = self.dimensions(file)?;
            let index = self.frame_index(file)?.to_vec();

            let targets: Vec<usize> = timestamps
                .iter()
                .map(|&t| Self::nearest_sample_index(&index, t))
                .collect();

            let min_target = *targets.iter().min().expect("timestamps is non-empty");
            let max_target = *targets.iter().max().expect("timestamps is non-empty");
            let start_idx = Self::keyframe_walkback(&index, min_target);

            let frame_count = max_target - start_idx + 1;
            self.ffmpeg_invocations += 1;
            let decoded = ffcli::decode_rgb24_sequential(
                file,
                index[start_idx].pts_seconds,
                frame_count,
                width,
                height,
                Some("videotoolbox"),
            )?;

            timestamps
                .iter()
                .zip(targets.iter())
                .map(|(&timestamp, &target_idx)| {
                    let data = decoded
                        .get(target_idx - start_idx)
                        .ok_or_else(|| {
                            crate::Error::Decode(format!(
                                "decode_rgb24_sequential returned {} frames, needed index {}",
                                decoded.len(),
                                target_idx - start_idx
                            ))
                        })?
                        .clone();
                    Ok(Frame {
                        width,
                        height,
                        camera: camera.to_string(),
                        timestamp,
                        pixels: FrameBuffer::Owned {
                            data: Arc::new(data),
                            channels: 3,
                        },
                    })
                })
                .collect()
        }
    }
}

/// Shared `ffmpeg`/`ffprobe` CLI helpers for the FFmpeg + CUDA (NVDEC) backends.
#[cfg(any(feature = "ffmpeg", feature = "cuda"))]
mod ffcli {
    use super::*;
    use std::process::Command;

    /// The real video stream codec name (e.g. `"av1"`, `"h264"`, `"hevc"`) via `ffprobe` --
    /// used by the native macOS `VideoToolboxDecoder` to detect AV1 files its
    /// `VTDecompressionSession` path can't parse and route them through the ffmpeg
    /// subprocess fallback instead.
    pub(crate) fn probe_codec(file: &Path) -> Result<String> {
        let out = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
            ])
            .arg(file)
            .output()
            .map_err(|e| crate::Error::Decode(format!("ffprobe not runnable: {e}")))?;
        if !out.status.success() {
            return Err(crate::Error::Decode(format!(
                "ffprobe failed for {}",
                file.display()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    /// Video stream width/height via `ffprobe`.
    pub(crate) fn probe_dims(file: &Path) -> Result<(u32, u32)> {
        let out = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
            ])
            .arg(file)
            .output()
            .map_err(|e| crate::Error::Decode(format!("ffprobe not runnable: {e}")))?;
        if !out.status.success() {
            return Err(crate::Error::Decode(format!(
                "ffprobe failed for {}",
                file.display()
            )));
        }
        let s = String::from_utf8_lossy(&out.stdout);
        let (w, h) = s.trim().split_once('x').ok_or_else(|| {
            crate::Error::Decode(format!("unexpected ffprobe output: {:?}", s.trim()))
        })?;
        let parse = |v: &str| {
            v.trim()
                .parse::<u32>()
                .map_err(|_| crate::Error::Decode(format!("bad dimension {v:?}")))
        };
        Ok((parse(w)?, parse(h)?))
    }

    /// Decode one RGB24 frame at `timestamp`. `hwaccel` (e.g. `Some("cuda")` for NVDEC) inserts
    /// `-hwaccel <name>` before the input; `None` uses the ffmpeg build's default decode path.
    pub(crate) fn decode_rgb24(
        file: &Path,
        timestamp: f64,
        width: u32,
        height: u32,
        hwaccel: Option<&str>,
    ) -> Result<Vec<u8>> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-nostdin", "-v", "error"]);
        if let Some(hw) = hwaccel {
            cmd.args(["-hwaccel", hw]);
        }
        // Accurate (output) seek: `-ss` after `-i` decodes to the exact timestamp.
        cmd.arg("-i")
            .arg(file)
            .args(["-ss"])
            .arg(format!("{timestamp}"))
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
        let out = cmd
            .output()
            .map_err(|e| crate::Error::Decode(format!("ffmpeg not runnable: {e}")))?;
        if !out.status.success() {
            return Err(crate::Error::Decode(format!(
                "ffmpeg failed to decode {} @ {timestamp}s",
                file.display()
            )));
        }
        let expected = width as usize * height as usize * 3;
        if out.stdout.len() < expected {
            return Err(crate::Error::Decode(format!(
                "short frame from {}: got {} bytes, expected {expected}",
                file.display(),
                out.stdout.len()
            )));
        }
        let mut data = out.stdout;
        data.truncate(expected);
        Ok(data)
    }

    fn frame(camera: &str, timestamp: f64, width: u32, height: u32, data: Vec<u8>) -> Frame {
        Frame {
            width,
            height,
            camera: camera.to_string(),
            timestamp,
            pixels: FrameBuffer::Owned {
                data: Arc::new(data),
                channels: 3,
            },
        }
    }

    pub(crate) fn decode_frame(
        camera: &str,
        file: &Path,
        timestamp: f64,
        width: u32,
        height: u32,
        hwaccel: Option<&str>,
    ) -> Result<Frame> {
        let data = decode_rgb24(file, timestamp, width, height, hwaccel)?;
        Ok(frame(camera, timestamp, width, height, data))
    }

    /// One video stream frame's real presentation timestamp and whether it's
    /// a sync/keyframe -- used by the ffmpeg-subprocess GOP-reuse batch path
    /// (`macos_ffmpeg_fallback::VideoToolboxDecoder::decode_batch`) the same
    /// way the native `VTDecompressionSession` path uses its own
    /// container-parsed sample index (see `videotoolbox_native.rs`).
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct ProbedFrame {
        pub(crate) pts_seconds: f64,
        pub(crate) is_keyframe: bool,
    }

    /// Every frame's real presentation time and keyframe flag, via one
    /// `ffprobe` metadata-only pass (no decode) over the whole video stream.
    /// `best_effort_timestamp_time` (not `pts_time`) is used because it's
    /// what `ffprobe` itself recommends for a frame's effective timestamp
    /// when a container doesn't carry an explicit PTS for every frame.
    /// Frames are returned in the same order `ffmpeg`'s own `rawvideo`
    /// decode output emits them (presentation/display order), which is what
    /// lets `decode_rgb24_sequential` below match decoded frames back to
    /// this index by position alone.
    pub(crate) fn probe_frame_index(file: &Path) -> Result<Vec<ProbedFrame>> {
        let out = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=key_frame,best_effort_timestamp_time",
                "-of",
                "csv=p=0",
            ])
            .arg(file)
            .output()
            .map_err(|e| crate::Error::Decode(format!("ffprobe not runnable: {e}")))?;
        if !out.status.success() {
            return Err(crate::Error::Decode(format!(
                "ffprobe failed for {}",
                file.display()
            )));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut frames = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let mut fields = line.split(',');
            let key_frame = fields.next().ok_or_else(|| {
                crate::Error::Decode(format!("malformed ffprobe frame line: {line:?}"))
            })?;
            let pts_time = fields.next().ok_or_else(|| {
                crate::Error::Decode(format!("malformed ffprobe frame line: {line:?}"))
            })?;
            let pts_seconds: f64 = pts_time.trim().parse().map_err(|_| {
                crate::Error::Decode(format!(
                    "unparseable frame timestamp {pts_time:?} in {line:?}"
                ))
            })?;
            frames.push(ProbedFrame {
                pts_seconds,
                is_keyframe: key_frame.trim() == "1",
            });
        }
        if frames.is_empty() {
            return Err(crate::Error::Decode(format!(
                "ffprobe returned no frames for {}",
                file.display()
            )));
        }
        Ok(frames)
    }

    /// Decodes `frame_count` consecutive frames starting at the real
    /// keyframe timestamp `start_pts_seconds`, in one `ffmpeg` subprocess --
    /// the GOP-reuse building block: decoding N consecutive frames in one
    /// continuous run does real, less redundant work than N separate
    /// subprocess invocations each independently re-decoding from the start
    /// of the file (what `decode_rgb24`'s accurate output-seek does,
    /// correct but with no reuse across calls). `start_pts_seconds` must be
    /// a real keyframe's timestamp (fast input-seek before `-i` only lands
    /// reliably on keyframe boundaries); a small epsilon is subtracted to
    /// guard against landing just after it due to floating-point rounding.
    pub(crate) fn decode_rgb24_sequential(
        file: &Path,
        start_pts_seconds: f64,
        frame_count: usize,
        width: u32,
        height: u32,
        hwaccel: Option<&str>,
    ) -> Result<Vec<Vec<u8>>> {
        let seek = (start_pts_seconds - 0.001).max(0.0);
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-nostdin", "-v", "error", "-ss"])
            .arg(format!("{seek}"));
        if let Some(hw) = hwaccel {
            cmd.args(["-hwaccel", hw]);
        }
        cmd.arg("-i")
            .arg(file)
            .args(["-frames:v", &frame_count.to_string()])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
        let out = cmd
            .output()
            .map_err(|e| crate::Error::Decode(format!("ffmpeg not runnable: {e}")))?;
        if !out.status.success() {
            return Err(crate::Error::Decode(format!(
                "ffmpeg failed to decode {} from {start_pts_seconds}s ({frame_count} frames)",
                file.display()
            )));
        }
        let frame_size = width as usize * height as usize * 3;
        let expected = frame_size * frame_count;
        if out.stdout.len() < expected {
            return Err(crate::Error::Decode(format!(
                "short sequential decode from {}: got {} bytes, expected {expected} ({frame_count} frames)",
                file.display(),
                out.stdout.len()
            )));
        }
        Ok(out
            .stdout
            .chunks(frame_size)
            .take(frame_count)
            .map(|chunk| chunk.to_vec())
            .collect())
    }
}

#[cfg(feature = "cuda")]
pub use cuda::CudaDecoder;

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use std::collections::HashMap;

    /// NVIDIA **NVDEC** decoder driving `ffmpeg -hwaccel cuda`. v1 downloads each decoded frame to
    /// CPU memory as RGB24 (GPU-resident DLPack output is a later milestone). Requires a CUDA/NVDEC-
    /// enabled ffmpeg build with `ffmpeg`/`ffprobe` on `PATH`; selected on Linux when built with
    /// `--features cuda`. **Functional verification is deferred to NVIDIA hardware** — on this code
    /// path the only thing CI checks is that it compiles and lints.
    #[derive(Default)]
    pub struct CudaDecoder {
        dims: HashMap<PathBuf, (u32, u32)>,
    }

    impl CudaDecoder {
        fn dimensions(&mut self, file: &Path) -> Result<(u32, u32)> {
            if let Some(&d) = self.dims.get(file) {
                return Ok(d);
            }
            let dims = ffcli::probe_dims(file)?;
            self.dims.insert(file.to_path_buf(), dims);
            Ok(dims)
        }
    }

    impl Decoder for CudaDecoder {
        fn decode(&mut self, camera: &str, file: &Path, timestamp: f64) -> Result<Frame> {
            let (width, height) = self.dimensions(file)?;
            ffcli::decode_frame(camera, file, timestamp, width, height, Some("cuda"))
        }
    }
}

#[cfg(feature = "ffmpeg")]
pub use linux::FfmpegDecoder;

#[cfg(feature = "ffmpeg")]
mod linux {
    use super::*;
    use std::collections::HashMap;

    /// Video decoder driving the `ffmpeg` CLI (cross-platform: uses the platform's hwaccel —
    /// VAAPI/NVDEC on Linux, VideoToolbox on macOS — when the ffmpeg build supports it, software
    /// otherwise). Requires `ffmpeg` and `ffprobe` on `PATH`. Decodes a single RGB24 frame per
    /// call; the [`FrameCache`] avoids repeated work. A libav-linked path is a future optimization.
    #[derive(Default)]
    pub struct FfmpegDecoder {
        dims: HashMap<PathBuf, (u32, u32)>,
        frame_index: HashMap<PathBuf, Vec<ffcli::ProbedFrame>>,
        /// Real count of `ffmpeg` subprocess invocations this decoder has
        /// made (incremented at the actual `Command::new("ffmpeg")` call
        /// sites in `decode`/`decode_batch` below) -- used by
        /// `tests::decode_batch_matches_per_frame_decode_and_reduces_ffmpeg_invocations`
        /// to verify the GOP-reuse efficiency claim against a real spawn
        /// count, not inferred from timing.
        pub(crate) ffmpeg_invocations: usize,
    }

    impl FfmpegDecoder {
        /// Video stream width/height (cached per file), via `ffprobe`.
        fn dimensions(&mut self, file: &Path) -> Result<(u32, u32)> {
            if let Some(&d) = self.dims.get(file) {
                return Ok(d);
            }
            let dims = ffcli::probe_dims(file)?;
            self.dims.insert(file.to_path_buf(), dims);
            Ok(dims)
        }

        fn frame_index(&mut self, file: &Path) -> Result<&[ffcli::ProbedFrame]> {
            if !self.frame_index.contains_key(file) {
                let index = ffcli::probe_frame_index(file)?;
                self.frame_index.insert(file.to_path_buf(), index);
            }
            Ok(&self.frame_index[file])
        }

        /// The frame-index position whose real presentation time is nearest
        /// `timestamp`. Same approach as the macOS ffmpeg-fallback decoder's
        /// (see `macos_ffmpeg_fallback::VideoToolboxDecoder`) -- duplicated
        /// rather than shared because the two decoders' state (dims +
        /// frame_index caches) lives in different structs and pulling this
        /// out into a free function would need passing the index by value
        /// either way.
        fn nearest_sample_index(index: &[ffcli::ProbedFrame], timestamp: f64) -> usize {
            index
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a.pts_seconds - timestamp)
                        .abs()
                        .partial_cmp(&(b.pts_seconds - timestamp).abs())
                        .expect("pts_seconds is always finite")
                })
                .map(|(i, _)| i)
                .expect("frame_index is non-empty (probe_frame_index errors on empty)")
        }

        /// Walks back from `target_idx` to the nearest preceding (or equal)
        /// keyframe.
        fn keyframe_walkback(index: &[ffcli::ProbedFrame], target_idx: usize) -> usize {
            let mut start = target_idx;
            while start > 0 && !index[start].is_keyframe {
                start -= 1;
            }
            start
        }
    }

    impl Decoder for FfmpegDecoder {
        fn decode(&mut self, camera: &str, file: &Path, timestamp: f64) -> Result<Frame> {
            let (width, height) = self.dimensions(file)?;
            self.ffmpeg_invocations += 1;
            ffcli::decode_frame(camera, file, timestamp, width, height, None)
        }

        /// Real GOP-reuse override -- same strategy as
        /// `macos_ffmpeg_fallback::VideoToolboxDecoder::decode_batch`, with
        /// no `-hwaccel` flag (this decoder's `decode()` doesn't pass one
        /// either; VAAPI/NVDEC auto-selection isn't wired up here, matching
        /// existing behavior). See that implementation's doc comment for
        /// the full rationale.
        fn decode_batch(
            &mut self,
            camera: &str,
            file: &Path,
            timestamps: &[f64],
        ) -> Result<Vec<Frame>> {
            if timestamps.is_empty() {
                return Ok(Vec::new());
            }

            let (width, height) = self.dimensions(file)?;
            let index = self.frame_index(file)?.to_vec();

            let targets: Vec<usize> = timestamps
                .iter()
                .map(|&t| Self::nearest_sample_index(&index, t))
                .collect();

            let min_target = *targets.iter().min().expect("timestamps is non-empty");
            let max_target = *targets.iter().max().expect("timestamps is non-empty");
            let start_idx = Self::keyframe_walkback(&index, min_target);

            let frame_count = max_target - start_idx + 1;
            self.ffmpeg_invocations += 1;
            let decoded = ffcli::decode_rgb24_sequential(
                file,
                index[start_idx].pts_seconds,
                frame_count,
                width,
                height,
                None,
            )?;

            timestamps
                .iter()
                .zip(targets.iter())
                .map(|(&timestamp, &target_idx)| {
                    let data = decoded
                        .get(target_idx - start_idx)
                        .ok_or_else(|| {
                            crate::Error::Decode(format!(
                                "decode_rgb24_sequential returned {} frames, needed index {}",
                                decoded.len(),
                                target_idx - start_idx
                            ))
                        })?
                        .clone();
                    Ok(Frame {
                        width,
                        height,
                        camera: camera.to_string(),
                        timestamp,
                        pixels: FrameBuffer::Owned {
                            data: Arc::new(data),
                            channels: 3,
                        },
                    })
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic in-memory decoder that counts how many times it actually decoded.
    struct MockDecoder {
        calls: usize,
    }

    impl MockDecoder {
        fn new() -> Self {
            Self { calls: 0 }
        }
    }

    impl Decoder for MockDecoder {
        fn decode(&mut self, camera: &str, _file: &Path, timestamp: f64) -> Result<Frame> {
            self.calls += 1;
            Ok(Frame {
                width: 2,
                height: 1,
                camera: camera.to_string(),
                timestamp,
                pixels: FrameBuffer::Owned {
                    data: Arc::new(vec![timestamp as u8, 0, 0, 0, 0, 0]),
                    channels: 3,
                },
            })
        }
    }

    #[test]
    fn cache_avoids_redecoding_same_frame() {
        let mut dec = MockDecoder::new();
        let mut cache = FrameCache::new(8);
        let p = Path::new("videos/top.mp4");

        cache.get_or_decode(&mut dec, "top", p, 1.0).unwrap();
        cache.get_or_decode(&mut dec, "top", p, 1.0).unwrap(); // cache hit
        cache.get_or_decode(&mut dec, "top", p, 2.0).unwrap(); // miss

        assert_eq!(dec.calls, 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_evicts_least_recently_used() {
        let mut dec = MockDecoder::new();
        let mut cache = FrameCache::new(1);
        let p = Path::new("v.mp4");

        cache.get_or_decode(&mut dec, "c", p, 1.0).unwrap();
        cache.get_or_decode(&mut dec, "c", p, 2.0).unwrap(); // evicts ts=1.0
        cache.get_or_decode(&mut dec, "c", p, 1.0).unwrap(); // re-decode

        assert_eq!(dec.calls, 3);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn decode_batch_defaults_to_per_frame() {
        let mut dec = MockDecoder::new();
        let frames = dec
            .decode_batch("top", Path::new("v.mp4"), &[0.0, 1.0, 2.0])
            .unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(dec.calls, 3);
        assert_eq!(frames[2].timestamp, 2.0);
    }

    /// Real end-to-end regression test for the AV1 fallback: previously
    /// every real-world LeRobot v3.0 dataset (which all ship AV1 by
    /// default) could not be opened at all by this decoder. Generates a
    /// real AV1 (`libsvtav1`) clip via `ffmpeg`, decodes it through the
    /// actual public `VideoToolboxDecoder` used in production (not a
    /// direct call into the fallback internals), and cross-checks the
    /// decoded pixels against ffmpeg's own reference decode of the same
    /// frame -- the same cross-validation pattern the existing native
    /// H.264/HEVC tests use.
    #[cfg(all(target_os = "macos", feature = "videotoolbox", feature = "ffmpeg"))]
    #[test]
    fn av1_file_decodes_via_real_ffmpeg_fallback_and_matches_reference() {
        use std::process::Command;

        let tmp = tempfile::tempdir().unwrap();
        let mp4_path = tmp.path().join("clip_av1.mp4");
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=10",
                "-frames:v",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libsvtav1",
                "-g",
                "10",
            ])
            .arg(&mp4_path)
            .status()
            .expect("ffmpeg must be installed to run this test");
        assert!(status.success(), "ffmpeg failed to generate AV1 test clip");

        let mut decoder = VideoToolboxDecoder::default();
        let frame = decoder.decode("cam", &mp4_path, 0.5).unwrap();
        assert_eq!((frame.width, frame.height), (64, 48));

        let FrameBuffer::Owned { data, channels } = &frame.pixels else {
            panic!("AV1 fallback frames must be FrameBuffer::Owned (ffmpeg subprocess path), not IOSurface");
        };
        assert_eq!(*channels, 3);
        assert!(data.iter().any(|&b| b != 0), "decoded AV1 frame was all zero");

        // Cross-check against ffmpeg's own reference decode of the same timestamp.
        let reference = Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-i"])
            .arg(&mp4_path)
            .args(["-ss", "0.5", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .output()
            .expect("ffmpeg must be installed");
        assert!(reference.status.success());
        let expected_len = 64 * 48 * 3;
        assert_eq!(data.as_slice(), &reference.stdout[..expected_len]);

        // Real GOP-reuse batch path falls back to per-frame decode for AV1
        // (documented, not optimized) -- verify it still produces correct,
        // distinct frames for multiple real timestamps.
        let batch = decoder
            .decode_batch("cam", &mp4_path, &[0.0, 0.5, 1.0])
            .unwrap();
        assert_eq!(batch.len(), 3);
        let FrameBuffer::Owned { data: d0, .. } = &batch[0].pixels else {
            panic!("expected Owned")
        };
        let FrameBuffer::Owned { data: d1, .. } = &batch[2].pixels else {
            panic!("expected Owned")
        };
        assert_ne!(d0, d1, "frames 1s apart must decode to different content");
    }

    #[test]
    fn write_rgb24_into_matches_to_rgb24_bytes() {
        let frame = Frame {
            width: 2,
            height: 2,
            camera: "c".to_string(),
            timestamp: 0.0,
            pixels: FrameBuffer::Owned {
                data: Arc::new((0u8..12).collect()),
                channels: 3,
            },
        };
        let mut dst = vec![0xFFu8; 12];
        frame.write_rgb24_into(&mut dst);
        assert_eq!(dst, frame.to_rgb24_bytes());
        assert_eq!(dst, (0u8..12).collect::<Vec<u8>>());
    }

    #[test]
    fn write_rgb24_into_writes_only_its_own_slice_of_a_larger_buffer() {
        // Simulates a batch assembler writing several frames into one shared buffer: writing
        // frame 1 into its slot must not touch frame 0's or frame 2's bytes.
        let frame_len = 2 * 2 * 3; // width=2, height=2, RGB24
        let make_frame = |fill: u8| Frame {
            width: 2,
            height: 2,
            camera: "c".to_string(),
            timestamp: 0.0,
            pixels: FrameBuffer::Owned {
                data: Arc::new(vec![fill; frame_len]),
                channels: 3,
            },
        };

        let mut batch = vec![0u8; frame_len * 3];
        for (idx, frame) in [make_frame(1), make_frame(2), make_frame(3)]
            .iter()
            .enumerate()
        {
            frame.write_rgb24_into(&mut batch[idx * frame_len..(idx + 1) * frame_len]);
        }

        assert!(batch[0..frame_len].iter().all(|&b| b == 1));
        assert!(batch[frame_len..2 * frame_len].iter().all(|&b| b == 2));
        assert!(batch[2 * frame_len..3 * frame_len].iter().all(|&b| b == 3));
    }

    #[test]
    fn pool_reuses_buffers() {
        let mut pool = FramePool::new(4);
        let b = pool.take();
        assert_eq!(b.len(), 4);
        pool.give(b);
        assert_eq!(pool.free_count(), 1);
        let _b2 = pool.take();
        assert_eq!(pool.free_count(), 0);
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn ffmpeg_decoder_decodes_a_real_frame() {
        use std::process::Command;
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = tmp.path().join("v.mp4");
        // Generate a 64x48 @ 30fps, 2s test clip.
        let status = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=30",
                "-frames:v",
                "60",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&mp4)
            .status()
            .expect("ffmpeg must be installed to run this test");
        assert!(status.success());

        let mut dec = FfmpegDecoder::default();
        let frame = dec.decode("top", &mp4, 0.5).unwrap();
        assert_eq!((frame.width, frame.height), (64, 48));
        assert_eq!(frame.pixels.as_bytes().len(), 64 * 48 * 3);
        assert_eq!(frame.pixels.channels(), 3);
    }

    /// Generates a real H.264 MP4 (baseline profile, no B-frames, so decode
    /// order == display order -- matching `videotoolbox_native.rs`'s own
    /// GOP-reuse test clip) via `ffmpeg`: 20 frames, GOP size 10 (2 GOPs).
    #[cfg(feature = "ffmpeg")]
    fn generate_gop_test_clip(dir: &std::path::Path, width: u32, height: u32) -> PathBuf {
        use std::process::Command;
        let mp4_path = dir.join("gop_clip.mp4");
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
                "baseline",
                "-g",
                "10",
            ])
            .arg(&mp4_path)
            .status()
            .expect("ffmpeg must be installed to run this test");
        assert!(status.success(), "ffmpeg failed to generate GOP test clip");
        mp4_path
    }

    /// Real regression test for the ffmpeg-subprocess-fallback GOP-reuse
    /// fix (`FfmpegDecoder::decode_batch`): must produce pixel-identical
    /// results to calling `decode()` per timestamp in a loop (correctness),
    /// while invoking `ffmpeg` far fewer times than that naive loop would
    /// (the actual efficiency claim). Mirrors
    /// `videotoolbox_native::tests::decode_batch_reuses_gop_state_and_matches_per_frame_decode`'s
    /// structure for the subprocess-based decoder.
    #[cfg(feature = "ffmpeg")]
    #[test]
    fn decode_batch_matches_per_frame_decode_and_reduces_ffmpeg_invocations() {
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = generate_gop_test_clip(tmp.path(), 64, 48);

        // All 20 real frame timestamps, as in the native-path test: one
        // decode_batch call for all of them should need far fewer ffmpeg
        // subprocess invocations than 20 separate decode() calls.
        let timestamps: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();

        let mut batch_dec = FfmpegDecoder::default();
        let batch_frames = batch_dec.decode_batch("cam", &mp4, &timestamps).unwrap();
        assert_eq!(batch_frames.len(), 20);

        let mut naive_dec = FfmpegDecoder::default();
        for (i, &t) in timestamps.iter().enumerate() {
            let naive_frame = naive_dec.decode("cam", &mp4, t).unwrap();
            assert_eq!(
                naive_frame.pixels.as_bytes(),
                batch_frames[i].pixels.as_bytes(),
                "frame {i} (t={t}) differs between decode_batch and per-frame decode"
            );
            assert_eq!(naive_frame.timestamp, batch_frames[i].timestamp);
        }

        // The real efficiency claim: decode_batch's whole 20-frame span
        // (2 GOPs, GOP size 10) should need far fewer real `ffmpeg`
        // subprocess invocations than 20 separate decode() calls -- counted
        // for real via each decoder's `ffmpeg_invocations` field (see
        // `FfmpegDecoder`), incremented at the actual `Command::new("ffmpeg")`
        // call sites, not inferred from timing or re-derived logic.
        assert_eq!(
            batch_dec.ffmpeg_invocations, 1,
            "decode_batch over one 2-GOP clip should need exactly 1 ffmpeg subprocess"
        );
        assert_eq!(
            naive_dec.ffmpeg_invocations, 20,
            "20 separate decode() calls should need 20 ffmpeg subprocesses (no reuse)"
        );
    }

    /// Same regression test as
    /// `decode_batch_matches_per_frame_decode_and_reduces_ffmpeg_invocations`
    /// above, but for the macOS `ffmpeg -hwaccel videotoolbox` fallback
    /// decoder (`macos_ffmpeg_fallback::VideoToolboxDecoder`, re-exported
    /// here as `VideoToolboxDecoder` via `use super::*` since this cfg
    /// combination is exactly the one under which that module -- not the
    /// native `VTDecompressionSession` one -- is what gets compiled).
    #[cfg(all(target_os = "macos", not(feature = "videotoolbox"), feature = "ffmpeg"))]
    #[test]
    fn macos_fallback_decode_batch_matches_per_frame_decode_and_reduces_ffmpeg_invocations() {
        let tmp = tempfile::tempdir().unwrap();
        let mp4 = generate_gop_test_clip(tmp.path(), 64, 48);
        let timestamps: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();

        let mut batch_dec = VideoToolboxDecoder::default();
        let batch_frames = batch_dec.decode_batch("cam", &mp4, &timestamps).unwrap();
        assert_eq!(batch_frames.len(), 20);

        let mut naive_dec = VideoToolboxDecoder::default();
        for (i, &t) in timestamps.iter().enumerate() {
            let naive_frame = naive_dec.decode("cam", &mp4, t).unwrap();
            assert_eq!(
                naive_frame.pixels.as_bytes(),
                batch_frames[i].pixels.as_bytes(),
                "frame {i} (t={t}) differs between decode_batch and per-frame decode"
            );
            assert_eq!(naive_frame.timestamp, batch_frames[i].timestamp);
        }

        assert_eq!(
            batch_dec.ffmpeg_invocations, 1,
            "decode_batch over one 2-GOP clip should need exactly 1 ffmpeg subprocess"
        );
        assert_eq!(
            naive_dec.ffmpeg_invocations, 20,
            "20 separate decode() calls should need 20 ffmpeg subprocesses (no reuse)"
        );
    }

    #[test]
    fn preferred_backend_matches_platform() {
        let b = Backend::preferred();
        if cfg!(target_os = "macos") {
            if cfg!(any(
                feature = "videotoolbox",
                feature = "ffmpeg",
                feature = "cuda"
            )) {
                assert_eq!(b, Backend::VideoToolbox);
            } else {
                assert_eq!(b, Backend::Software);
            }
        } else if cfg!(feature = "cuda") {
            assert_eq!(b, Backend::Cuda);
        } else if cfg!(feature = "ffmpeg") {
            assert_eq!(b, Backend::Ffmpeg);
        } else {
            assert_eq!(b, Backend::Software);
        }
    }
}
