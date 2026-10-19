"""Optional torchcodec decode helper.

PyRoboFrames' native path is VideoToolbox on macOS. On Linux/CI, torchcodec
(PyTorch's video decoder, BSD-3) provides an accelerated fallback. This helper
returns RGB frames as NumPy arrays; it does not replace the Rust dataset
pipeline and is not imported unless called.
"""

from __future__ import annotations

import numpy as np


class TorchCodecUnavailable(RuntimeError):
    """Raised when torchcodec is not installed."""


def _require():
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError as exc:
        raise TorchCodecUnavailable(
            "torchcodec is not installed. Install the optional decoder with "
            '`pip install "pyroboframes[torchcodec]"` (requires PyTorch).'
        ) from exc
    return VideoDecoder


def decode_video_frames(
    path: str,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Decode a time range to an ``(N, H, W, 3)`` uint8 RGB array."""
    VideoDecoder = _require()
    decoder = VideoDecoder(path, device=device)
    metadata = decoder.metadata
    fps = float(getattr(metadata, "average_fps", 0.0) or 0.0)
    if fps <= 0:
        fps = 30.0
    start = max(0, round(start_seconds * fps))
    if end_seconds is None:
        end = int(getattr(metadata, "num_frames", start + 1))
    else:
        end = max(start + 1, round(end_seconds * fps))
    try:
        frames = decoder.get_frames_at(list(range(start, end)))
    except AttributeError:
        frames = decoder[start:end]
    array = np.asarray(frames)
    if array.ndim == 4 and array.shape[1] == 3:
        array = np.transpose(array, (0, 2, 3, 1))
    return array.astype(np.uint8, copy=False)


__all__ = ["TorchCodecUnavailable", "decode_video_frames"]
