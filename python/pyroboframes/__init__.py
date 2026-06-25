"""PyRoboFrames — zero-copy, hardware-accelerated robot-learning dataloader for Apple Silicon.

The heavy lifting (dataset reading, VideoToolbox decode, zero-copy windowing) lives in the
compiled Rust extension ``pyroboframes._core``. This package is the ergonomic Python surface:
the ``RoboFrameDataset`` / ``Loader`` API and the MLX / NumPy / PyTorch adapters.

The public API is still being built (see the project ROADMAP); today the package exposes the
engine version so wheels and the build can be smoke-tested end to end.
"""

from __future__ import annotations

from . import _core, backend, transforms
from ._core import Loader, RoboFrameDataset
from .backend import available_backends, resolve_device
from .dataloader import DataLoader

__all__ = [
    "__version__",
    "engine_version",
    "RoboFrameDataset",
    "Loader",
    "DataLoader",
    "transforms",
    "backend",
    "resolve_device",
    "available_backends",
]

__version__: str = _core.__version__


def engine_version() -> str:
    """Return the version of the underlying Rust engine."""
    return _core.engine_version()
