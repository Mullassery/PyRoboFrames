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
    /// **Zero-copy hand-off flow (once mlx#2855 lands):**
    /// 1. VideoToolbox decodes to CVPixelBuffer (IOSurface-backed)
    /// 2. Rust wraps IOSurfaceRef in FrameBuffer::IOSurface
    /// 3. Python extracts the opaque IOSurface pointer
    /// 4. MLX initializes array directly over IOSurface (no copy)
    /// 5. MLX array is immediately GPU-usable (unified memory)
    /// 6. IOSurfaceUseCount tracks lifetime; buffer recycled after Metal finishes
    ///
    /// **Current status:** Blocked by [mlx#2855](https://github.com/ml-explore/mlx/issues/2855).
    /// v1 uses ffmpeg RGB24 output (Owned variant); true zero-copy awaits MLX support.
    #[cfg(target_os = "macos")]
    IOSurface {
        /// Opaque IOSurface reference (size varies by platform; encoded as u64 on 64-bit).
        /// Kept behind a type boundary so Python bindings can safely extract it.
        surface: u64,
        /// Pixel format code (kCVPixelFormatType_*; RFC 3394 "pixel format" enums).
        format: u32,
    },
}

impl FrameBuffer {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            FrameBuffer::Owned { data, .. } => data,
            #[cfg(target_os = "macos")]
            FrameBuffer::IOSurface { .. } => {
                panic!("IOSurface buffers are GPU-resident; use MLX/Metal APIs for access")
            }
        }
    }

    pub fn channels(&self) -> u8 {
        match self {
            FrameBuffer::Owned { channels, .. } => *channels,
            #[cfg(target_os = "macos")]
            FrameBuffer::IOSurface { .. } => 3, // kCVPixelFormatType_24RGB variant
        }
    }

    /// Extract the IOSurface pointer for direct GPU access (macOS only).
    /// Returns None for Owned buffers (CPU-resident). Pending MLX support for zero-copy.
    #[cfg(target_os = "macos")]
    pub fn as_iosurface(&self) -> Option<u64> {
        match self {
            FrameBuffer::IOSurface { surface, .. } => Some(*surface),
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

/// A hardware (or software) video decoder, selected per platform (see [`Backend`]).
pub trait Decoder: Send {
    /// Decode the frame of `camera` in `file` nearest `timestamp` (seconds).
    fn decode(&mut self, camera: &str, file: &Path, timestamp: f64) -> Result<Frame>;

    /// Decode several timestamps from one video at once. The default decodes one-by-one;
    /// hardware backends override this to order seeks and reuse GOP decode state (à la
    /// torchcodec), which is much faster than independent seeks.
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
    /// - macOS: [`Backend::VideoToolbox`] if available (with ffmpeg feature), else Software.
    /// - Linux: [`Backend::Cuda`] if compiled with `--features cuda`; else [`Backend::Ffmpeg`].
    /// Real *runtime* auto-detection (probe the GPU, fall back to Software) is a future enhancement.
    pub fn preferred() -> Backend {
        if cfg!(target_os = "macos") {
            #[cfg(all(feature = "videotoolbox", any(feature = "ffmpeg", feature = "cuda")))]
            {
                return Backend::VideoToolbox;
            }
            #[cfg(not(all(feature = "videotoolbox", any(feature = "ffmpeg", feature = "cuda"))))]
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

/// macOS VideoToolbox decoder stub — requires FFmpeg CLI with `-hwaccel videotoolbox` support.
/// (VideoToolbox integration is feature-gated; macOS builds without ffmpeg feature fall back to Software decoder.)
#[cfg(all(feature = "videotoolbox", any(feature = "ffmpeg", feature = "cuda")))]
pub use macos::VideoToolboxDecoder;

#[cfg(all(feature = "videotoolbox", any(feature = "ffmpeg", feature = "cuda")))]
mod macos {
    use super::*;
    use std::collections::HashMap;

    /// macOS VideoToolbox decoder using Apple's Media Engine for H.264/HEVC hardware decode.
    /// Outputs RGB24 frames via ffmpeg with `-hwaccel videotoolbox` (CPU-resident v1;
    /// GPU-resident / IOSurface optimization is pending).
    #[derive(Default)]
    pub struct VideoToolboxDecoder {
        dims: HashMap<PathBuf, (u32, u32)>,
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
    }

    impl Decoder for VideoToolboxDecoder {
        fn decode(&mut self, camera: &str, file: &Path, timestamp: f64) -> Result<Frame> {
            let (width, height) = self.dimensions(file)?;
            // VideoToolbox decode via ffmpeg hwaccel (macOS default) → CPU RGB24.
            // TODO: Direct VideoToolbox session management + IOSurface output for zero-copy.
            ffcli::decode_frame(camera, file, timestamp, width, height, None)
        }
    }
}

/// Shared `ffmpeg`/`ffprobe` CLI helpers for the FFmpeg + CUDA (NVDEC) backends.
#[cfg(any(feature = "ffmpeg", feature = "cuda"))]
mod ffcli {
    use super::*;
    use std::process::Command;

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
    }

    impl Decoder for FfmpegDecoder {
        fn decode(&mut self, camera: &str, file: &Path, timestamp: f64) -> Result<Frame> {
            let (width, height) = self.dimensions(file)?;
            ffcli::decode_frame(camera, file, timestamp, width, height, None)
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

    #[test]
    fn preferred_backend_matches_platform() {
        let b = Backend::preferred();
        if cfg!(target_os = "macos") {
            assert_eq!(b, Backend::VideoToolbox);
        } else if cfg!(feature = "cuda") {
            assert_eq!(b, Backend::Cuda);
        } else {
            assert_eq!(b, Backend::Ffmpeg);
        }
    }
}
