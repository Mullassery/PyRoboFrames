"""Runtime backend / device selection — the seam behind "Train Anywhere".

The same script targets CUDA on NVIDIA, MLX or MPS on Apple Silicon, or CPU, chosen at runtime
rather than by editing code. Resolution order for ``device="auto"``:

1. the ``PYROBOFRAMES_DEVICE`` environment variable, if set;
2. CUDA, if a CUDA-capable PyTorch is available;
3. on Apple Silicon: MLX if installed, else MPS (PyTorch) if available;
4. CPU (NumPy) otherwise.
"""

from __future__ import annotations

import os
import platform

VALID_DEVICES = ("cuda", "mps", "mlx", "cpu")


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:
        return None


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401,PLC0415

        return True
    except Exception:
        return False


def resolve_device(device: str = "auto") -> str:
    """Resolve ``device`` to a concrete backend in :data:`VALID_DEVICES`.

    A concrete value (e.g. ``"cuda"``) is returned as-is; ``"auto"`` (or ``None``) is detected,
    honoring the ``PYROBOFRAMES_DEVICE`` override.
    """
    if not device:
        device = "auto"
    if device == "auto":
        device = os.environ.get("PYROBOFRAMES_DEVICE", "auto")
    if device != "auto":
        if device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES} or 'auto' (got {device!r})")
        return device

    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if _is_apple_silicon():
        if _has_mlx():
            return "mlx"
        mps = getattr(getattr(torch, "backends", None), "mps", None) if torch else None
        if mps is not None and mps.is_available():
            return "mps"
    return "cpu"


def available_backends() -> dict[str, bool]:
    """Map each backend to whether it's usable in this environment (for diagnostics)."""
    torch = _torch()
    return {
        "cuda": bool(torch and torch.cuda.is_available()),
        "mps": bool(
            torch
            and getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        ),
        "mlx": _has_mlx() and _is_apple_silicon(),
        "cpu": True,
    }
