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
    """Resize each frame to ``(height, width)``.

    ``interpolation`` is ``"bilinear"`` (default, half-pixel aligned) or ``"nearest"``. Bilinear
    returns ``float32``; nearest preserves the input dtype.
    """

    def __init__(self, height: int, width: int, interpolation: str = "bilinear"):
        if height <= 0 or width <= 0:
            raise ValueError("Resize dimensions must be positive")
        if interpolation not in ("bilinear", "nearest"):
            raise ValueError("interpolation must be 'bilinear' or 'nearest'")
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, x):
        _, h, w, _ = x.shape
        if self.interpolation == "nearest":
            ys = (np.arange(self.height) * h) // self.height
            xs = (np.arange(self.width) * w) // self.width
            return x[:, ys][:, :, xs]
        return _bilinear_resize(x, self.height, self.width)

    def __repr__(self) -> str:
        return f"Resize({self.height}, {self.width}, interpolation={self.interpolation!r})"


def _bilinear_resize(x, out_h: int, out_w: int):
    """Vectorized half-pixel-aligned bilinear resize of `[N, H, W, C]` -> `[N, out_h, out_w, C]`."""
    _, h, w, _ = x.shape
    xf = x.astype(np.float32)
    # Half-pixel centers map output coords back to input space, clamped to the edges.
    src_y = (np.arange(out_h, dtype=np.float32) + 0.5) * (h / out_h) - 0.5
    src_x = (np.arange(out_w, dtype=np.float32) + 0.5) * (w / out_w) - 0.5
    src_y = np.clip(src_y, 0, h - 1)
    src_x = np.clip(src_x, 0, w - 1)
    y0 = np.floor(src_y).astype(np.intp)
    x0 = np.floor(src_x).astype(np.intp)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (src_y - y0)[:, None]  # [out_h, 1]
    wx = (src_x - x0)[None, :]  # [1, out_w]

    # Gather the four neighbors: [N, out_h, out_w, C].
    top = xf[:, y0][:, :, x0] * (1 - wx)[..., None] + xf[:, y0][:, :, x1] * wx[..., None]
    bot = xf[:, y1][:, :, x0] * (1 - wx)[..., None] + xf[:, y1][:, :, x1] * wx[..., None]
    return (top * (1 - wy)[..., None] + bot * wy[..., None]).astype(np.float32)


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


class RandomHorizontalFlip:
    """Flip each frame left-right independently with probability ``p``. ``seed`` makes it
    reproducible (the RNG advances per call)."""

    def __init__(self, p: float = 0.5, seed: int | None = None):
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def __call__(self, x):
        flip = self._rng.random(x.shape[0]) < self.p
        out = x.copy()
        out[flip] = x[flip, :, ::-1, :]
        return out

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomCrop:
    """Crop a random ``(height, width)`` window per frame (clamped to the frame size)."""

    def __init__(self, height: int, width: int, seed: int | None = None):
        self.height = height
        self.width = width
        self._rng = np.random.default_rng(seed)

    def __call__(self, x):
        n, h, w, c = x.shape
        ch, cw = min(self.height, h), min(self.width, w)
        tops = self._rng.integers(0, h - ch + 1, size=n)
        lefts = self._rng.integers(0, w - cw + 1, size=n)
        out = np.empty((n, ch, cw, c), dtype=x.dtype)
        for i in range(n):
            out[i] = x[i, tops[i] : tops[i] + ch, lefts[i] : lefts[i] + cw, :]
        return out

    def __repr__(self) -> str:
        return f"RandomCrop({self.height}, {self.width})"


class ColorJitter:
    """Multiply each frame by a random per-sample brightness factor in ``[1-b, 1+b]``.

    Integer inputs are clamped to ``[0, 255]`` and keep their dtype; float inputs pass through.
    """

    def __init__(self, brightness: float = 0.0, seed: int | None = None):
        if brightness < 0:
            raise ValueError("brightness must be >= 0")
        self.brightness = float(brightness)
        self._rng = np.random.default_rng(seed)

    def __call__(self, x):
        n = x.shape[0]
        factors = self._rng.uniform(
            1.0 - self.brightness, 1.0 + self.brightness, size=n
        ).astype(np.float32)
        out = x.astype(np.float32) * factors[:, None, None, None]
        if np.issubdtype(x.dtype, np.integer):
            return np.clip(out, 0, 255).astype(x.dtype)
        return out

    def __repr__(self) -> str:
        return f"ColorJitter(brightness={self.brightness})"
