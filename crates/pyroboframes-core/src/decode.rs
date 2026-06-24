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
    // Future (macOS): an IOSurface handle for zero-copy hand-off to MLX/Metal.
}

impl FrameBuffer {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            FrameBuffer::Owned { data, .. } => data,
        }
    }

    pub fn channels(&self) -> u8 {
        match self {
            FrameBuffer::Owned { channels, .. } => *channels,
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
    VideoToolbox,
    Ffmpeg,
    Software,
}

impl Backend {
    /// The preferred backend for the current platform. Real auto-detection — probing whether the
    /// hardware path is actually available and falling back to [`Backend::Software`] otherwise —
    /// lands with the backend implementations after the spikes.
    pub fn preferred() -> Backend {
        if cfg!(target_os = "macos") {
            Backend::VideoToolbox
        } else {
            Backend::Ffmpeg
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

#[cfg(feature = "videotoolbox")]
pub use macos::VideoToolboxDecoder;

#[cfg(feature = "videotoolbox")]
mod macos {
    use super::*;

    /// macOS VideoToolbox decoder. Stub pending Spike B (VideoToolbox→IOSurface) and the
    /// `videotoolbox` crate integration.
    #[derive(Default)]
    pub struct VideoToolboxDecoder;

    impl Decoder for VideoToolboxDecoder {
        fn decode(&mut self, _camera: &str, _file: &Path, _timestamp: f64) -> Result<Frame> {
            Err(crate::Error::Decode(
                "VideoToolbox backend not yet implemented (pending Spike B)".into(),
            ))
        }
    }
}

#[cfg(feature = "ffmpeg")]
pub use linux::FfmpegDecoder;

#[cfg(feature = "ffmpeg")]
mod linux {
    use super::*;

    /// FFmpeg decoder for Linux (VAAPI/NVDEC + software fallback). Stub pending system-FFmpeg
    /// integration.
    #[derive(Default)]
    pub struct FfmpegDecoder;

    impl Decoder for FfmpegDecoder {
        fn decode(&mut self, _camera: &str, _file: &Path, _timestamp: f64) -> Result<Frame> {
            Err(crate::Error::Decode(
                "FFmpeg backend not yet implemented (pending system FFmpeg integration)".into(),
            ))
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

    #[test]
    fn preferred_backend_matches_platform() {
        let b = Backend::preferred();
        if cfg!(target_os = "macos") {
            assert_eq!(b, Backend::VideoToolbox);
        } else {
            assert_eq!(b, Backend::Ffmpeg);
        }
    }
}
