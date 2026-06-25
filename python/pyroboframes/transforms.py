"""Image transforms for camera-frame batches.

Operate on a `[N, H, W, C]` array (the shape the loader yields per camera) and return a
transformed array. This is the CPU/NumPy implementation — the same op surface is what the
CV-CUDA (NVIDIA) and MLX backends will plug into later, so a transform script written today keeps
working as those land.

```python
from pyroboframes import transforms as T
tf = T.Compose([T.Resize(224, 224), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
```

Note: `Resize` uses nearest-neighbor sampling (dependency-free, deterministic); higher-quality
interpolation comes with the GPU backends.
"""

from __future__ import annotations

import numpy as np


class Compose:
    """Chain transforms, applied left to right."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"


class Resize:
    """Resize each frame to ``(height, width)`` with nearest-neighbor sampling."""

    def __init__(self, height: int, width: int):
        if height <= 0 or width <= 0:
            raise ValueError("Resize dimensions must be positive")
        self.height = height
        self.width = width

    def __call__(self, x):
        _, h, w, _ = x.shape
        ys = (np.arange(self.height) * h) // self.height
        xs = (np.arange(self.width) * w) // self.width
        return x[:, ys][:, :, xs]

    def __repr__(self) -> str:
        return f"Resize({self.height}, {self.width})"


class CenterCrop:
    """Crop the centered ``(height, width)`` region of each frame (clamped to the frame size)."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, x):
        _, h, w, _ = x.shape
        ch = min(self.height, h)
        cw = min(self.width, w)
        top = (h - ch) // 2
        left = (w - cw) // 2
        return x[:, top : top + ch, left : left + cw, :]

    def __repr__(self) -> str:
        return f"CenterCrop({self.height}, {self.width})"


class Normalize:
    """Scale to ``[0, 1]`` (divide by ``scale``) then standardize per channel: ``(x - mean) / std``.

    Returns ``float32``. ``mean``/``std`` are per-channel (length C).
    """

    def __init__(self, mean, std, scale: float = 255.0):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.scale = float(scale)

    def __call__(self, x):
        x = x.astype(np.float32) / self.scale
        return (x - self.mean) / self.std

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean.tolist()}, std={self.std.tolist()})"
