"""torchcodec helper tests with a fake decoder module."""

from __future__ import annotations

import sys
import types

import numpy as np
from pyroboframes.torchcodec_decode import TorchCodecUnavailable, decode_video_frames


class _Meta:
    average_fps = 10.0
    num_frames = 30


class _Decoder:
    def __init__(self, path, device="cpu"):
        self.metadata = _Meta()

    def get_frames_at(self, indices):
        return np.zeros((len(indices), 3, 4, 4), dtype=np.uint8)


def test_decode_frames_transposes_to_rgb(monkeypatch):
    module = types.ModuleType("torchcodec.decoders")
    module.VideoDecoder = _Decoder
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", module)
    frames = decode_video_frames("x.mp4", start_seconds=1.0, end_seconds=2.0)
    assert frames.shape == (10, 4, 4, 3)


def test_missing_torchcodec_is_actionable(monkeypatch):
    import pytest

    monkeypatch.setitem(sys.modules, "torchcodec", None)
    with pytest.raises(TorchCodecUnavailable, match="pyroboframes\\[torchcodec\\]"):
        decode_video_frames("x.mp4")
