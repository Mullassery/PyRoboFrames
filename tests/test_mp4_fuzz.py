"""Subprocess-isolated fuzz smoke test for the optional torchcodec MP4 decoder.

Skipped (not faked) unless both ``torch`` and ``torchcodec`` are importable and
``ffmpeg`` can generate a tiny real H.264 fixture. Decode runs in a child
process with a hard timeout, so a native decoder crash/hang fails the test
instead of taking down pytest.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
except ImportError:
    pytest.skip(
        "torch is not installed; MP4 fuzz decode requires the optional "
        '`pip install "pyroboframes[torchcodec]"` stack',
        allow_module_level=True,
    )

try:
    import torchcodec  # noqa: F401
except ImportError:
    pytest.skip(
        "torchcodec is not installed; install it with "
        '`pip install "pyroboframes[torchcodec]"`',
        allow_module_level=True,
    )

if shutil.which("ffmpeg") is None:
    pytest.skip(
        "ffmpeg is not on PATH, so a real tiny MP4 fixture cannot be generated",
        allow_module_level=True,
    )

_DECODE_SCRIPT = r"""
import sys
from pyroboframes.torchcodec_decode import decode_video_frames

try:
    decode_video_frames(sys.argv[1])
except Exception as exc:  # decoder-specific clean failure
    print(f"{type(exc).__name__}: {exc}")
    raise SystemExit(0)
raise SystemExit(3)  # expected a rejection for a guaranteed-corrupt file
"""


def _tiny_mp4(path: Path) -> None:
    import numpy as np

    from pyroboframes.lerobot import encode_video_frames

    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    frames[..., 0] = 255
    encode_video_frames(frames, str(path), fps=10.0, codec="h264")


def test_mutated_mp4_decode_finishes_in_subprocess(tmp_path: Path) -> None:
    valid = tmp_path / "valid.mp4"
    try:
        _tiny_mp4(valid)
    except RuntimeError as exc:
        pytest.skip(f"could not generate tiny MP4 fixture: {exc}")

    raw = valid.read_bytes()
    mutated_payloads = [
        raw[:0],
        raw[:8],
        b"NOTANMP4" + raw[8:],
        bytes([raw[0] ^ 0xFF]) + raw[1:],
    ]

    for index, payload in enumerate(mutated_payloads):
        corrupted = tmp_path / f"mutated-{index}.mp4"
        corrupted.write_bytes(payload)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _DECODE_SCRIPT, str(corrupted)],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"torchcodec decode hung on mutated MP4 case {index}")
        assert proc.returncode == 0, (
            f"mutated MP4 case {index} crashed or returned {proc.returncode}: "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
