# Video Codec Selection in PyRoboFrames v0.4.0

PyRoboFrames now supports multiple video codecs for storing camera data in LeRobot datasets. This guide helps you choose the right codec for your use case.

## Codec Comparison

| Codec | Compression | Encode Speed | Decode Speed | Hardware Support | Use Case |
|-------|-------------|--------------|--------------|------------------|----------|
| **H.264** (default) | Baseline | Fast | Very Fast | Universal | Compatibility, real-time playback, all devices |
| **HEVC** | 30-40% smaller | Slower | Faster | Good (RTX 20+, Apple T2+) | Storage efficiency, archival, modern hardware |
| **AV1** | 50-60% smaller | Very Slow | Slower | Limited (RTX 50xx+, recent Apple) | Extreme compression, cloud storage, future-proof |

## Quick Start

### Default Behavior (H.264)

By default, datasets use H.264, the most compatible codec:

```python
import pyroboframes as prf
import numpy as np

features = {
    "observation.state": np.zeros((1000, 7), dtype=np.float32),
    "action": np.zeros((1000, 7), dtype=np.float32),
}
episode_lengths = [500, 500]

# Uses H.264 (default)
prf.write_lerobot_dataset("my_dataset/", features, episode_lengths, fps=30.0)
```

### Using HEVC (30-40% Storage Savings)

For most modern hardware, HEVC offers a good balance of compression and speed:

```python
prf.write_lerobot_dataset(
    "my_dataset/",
    features,
    episode_lengths,
    fps=30.0,
    video_codec="hevc",
    video_profile="main"  # 8-bit encoding (standard quality)
)
```

### Using AV1 (Best Compression, Slowest Encoding)

Use AV1 when storage is critical and encoding time is not a constraint:

```python
prf.write_lerobot_dataset(
    "my_dataset/",
    features,
    episode_lengths,
    fps=30.0,
    video_codec="av1"
)
```

## Codec Details

### H.264 (AVC)

**Best For:** Maximum compatibility, real-time playback, embedded devices, mobile

**Pros:**
- Universal hardware support (all devices, browsers, mobile)
- Fast encoding and decoding
- Mature codec, widely tested
- Sufficient quality for most robotics applications

**Cons:**
- Larger file sizes than newer codecs
- Patent licensing concerns in some regions

**Encoding:** Uses `libx264` (software) or hardware accelerators (`h264_nvenc` on NVIDIA, `h264_videotoolbox` on Apple)

**Profiles:** `baseline` (streaming), `main` (standard), `high` (quality)

**Example:**
```python
prf.write_lerobot_dataset(
    "dataset/",
    features,
    episode_lengths,
    video_codec="h264",
    video_profile="high"
)
```

---

### HEVC/H.265

**Best For:** Archival, cloud storage, datasets with high-quality camera data, modern compute infrastructure

**Pros:**
- 30-40% smaller files than H.264 (same quality)
- Good hardware support on modern GPUs (RTX 20 series, RTX 30+, Apple T2+)
- Royalty-free in many jurisdictions
- Good for long-term archival

**Cons:**
- Slower encoding than H.264 (~2-3x slower)
- Not supported on older devices (pre-2017 GPUs, old mobile)
- Licensing complexity in some regions (though improving)

**Encoding:** Uses `libx265` (software) or hardware accelerators (`hevc_nvenc` on NVIDIA, `hevc_videotoolbox` on Apple)

**Profiles:** `main` (8-bit), `main10` (10-bit, higher quality, requires more bitrate)

**Example:**
```python
prf.write_lerobot_dataset(
    "dataset/",
    features,
    episode_lengths,
    video_codec="hevc",
    video_profile="main"
)
```

**Typical Savings:**
- 1000-frame 480p video: ~500 MB (H.264) → ~300 MB (HEVC)
- 10,000-frame dataset: ~5 GB → ~3 GB

---

### AV1

**Best For:** Extreme compression needs, cloud long-term storage, 5-10 year archival

**Pros:**
- 50-60% smaller than H.264 (same quality)
- Royalty-free, future-proof
- Excellent for cloud/edge storage at massive scale

**Cons:**
- Very slow encoding (10-50x slower than H.264, depending on settings)
- Limited hardware decode support (RTX 50 series, recent Apple)
- Not suitable for real-time encoding

**Encoding:** Uses `libaom-av1` (software)

**Example:**
```python
prf.write_lerobot_dataset(
    "dataset/",
    features,
    episode_lengths,
    video_codec="av1"
)
```

**Typical Savings:**
- 1000-frame 480p video: ~500 MB (H.264) → ~200 MB (AV1)
- Encoding time: ~20-60 minutes for 1000 frames (vs. ~1-2 minutes for H.264)

---

## Loading Datasets with Different Codecs

Datasets automatically detect the codec from metadata:

```python
import pyroboframes as prf

# Load dataset (codec is auto-detected from meta/info.json)
ds = prf.RoboFrameDataset.from_path("my_dataset/")

# The loader handles all codecs transparently
loader = ds.loader(cameras=["observation.images.top"])

for batch in loader:
    # Works the same regardless of codec
    images = batch["observation.images.top"]  # [batch_size, H, W, 3]
```

## Backwards Compatibility

- Datasets created with PyRoboFrames v0.3.x default to H.264
- The `video_codec` field in `meta/info.json` defaults to `"h264"` if not present
- Existing datasets work unchanged

## Performance Benchmarks

Encoding time on Apple Silicon M3 (single thread), 1000 × 480p frames:

| Codec | Time | Speed |
|-------|------|-------|
| H.264 | ~2 min | Fast |
| HEVC | ~6 min | 3x slower |
| AV1 | ~45 min | 22x slower |

Actual times depend on:
- CPU/GPU capability
- Video resolution (480p vs 720p vs 1080p)
- Encoder preset (software encoders have quality/speed tradeoffs)

## When to Use Each Codec

### Use H.264 if:
- ✅ You need maximum compatibility (mobile, web, embedded)
- ✅ You have limited encoding time
- ✅ Your hardware is older (pre-2017)
- ✅ You're uncertain (it's the safe default)

### Use HEVC if:
- ✅ You have modern hardware (RTX 20+, Apple T2+)
- ✅ Storage is a concern (~30% savings)
- ✅ You don't need to decode on mobile/legacy devices
- ✅ You can tolerate 2-3x slower encoding

### Use AV1 if:
- ✅ Storage is critical (cloud/edge scenarios with petabyte-scale data)
- ✅ You have days/weeks for encoding
- ✅ You're archiving data for 5-10 years
- ✅ You have recent hardware with AV1 decode (RTX 50+, 2024+ Apple)

## Troubleshooting

### "My GPU doesn't support this codec"

PyRoboFrames uses software fallback (libx264, libx265, libaom-av1). Hardware encoders are optimized but optional.

To check available encoders:
```bash
ffmpeg -codecs | grep "HEVC"  # Check for hevc support
ffmpeg -codecs | grep "AV1"   # Check for av1 support
```

### "Encoding is very slow"

This is expected:
- H.264: 1-3 minutes per 1000 frames (normal)
- HEVC: 3-10 minutes per 1000 frames (slower, expected)
- AV1: 30-60 minutes per 1000 frames (very slow, expected)

If you have an NVIDIA GPU, hardware encoding is ~10x faster:
- Install: `pip install nvidia-cuda-toolkit`
- FFmpeg will auto-detect and use `hevc_nvenc` or `h264_nvenc`

### "My dataset won't load"

Ensure `meta/info.json` has the codec field:
```json
{
  "fps": 30,
  "video_codec": "h264",  // or "hevc", "av1"
  "features": { ... }
}
```

If missing, the loader defaults to H.264.

## API Reference

### write_lerobot_dataset()

```python
prf.write_lerobot_dataset(
    path: str,
    features: dict[str, np.ndarray],
    episode_lengths: list[int],
    fps: float = 30.0,
    robot_type: str | None = None,
    video_codec: str = "h264",  # "h264" | "hevc" | "av1"
    video_profile: str | None = None,  # e.g., "main" for HEVC
) -> None
```

**Parameters:**
- `video_codec`: Video codec ("h264" [default], "hevc", "av1")
- `video_profile`: Codec-specific profile (e.g., "main" for HEVC)

**Raises:**
- `ValueError`: If `video_codec` is not "h264", "hevc", or "av1"

### Metadata Format

Dataset metadata is stored in `meta/info.json`:

```json
{
  "codebase_version": "v3.0",
  "fps": 30.0,
  "total_episodes": 2,
  "total_frames": 1000,
  "video_codec": "hevc",
  "video_profile": "main",
  "features": { ... }
}
```

## Further Reading

- [H.264 Codec Reference](https://en.wikipedia.org/wiki/Advanced_Video_Coding)
- [HEVC Codec Reference](https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding)
- [AV1 Codec Reference](https://en.wikipedia.org/wiki/AV1)
- [FFmpeg Encoding Guide](https://trac.ffmpeg.org/wiki/Encode)
