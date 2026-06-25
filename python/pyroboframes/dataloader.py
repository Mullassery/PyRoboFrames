"""Device-aware loader wrapper.

Wraps a (NumPy-output) loader from :meth:`RoboFrameDataset.loader`, applies image transforms to
camera-frame entries, and moves every array to the resolved backend — so the same loop runs on
MLX, MPS, CUDA, or CPU without code changes (see :mod:`pyroboframes.backend`).

```python
import pyroboframes as prf
from pyroboframes import transforms as T

ds = prf.RoboFrameDataset.from_path("…")
raw = ds.loader(batch_size=32, cameras=["observation.images.top"])  # output="numpy"
loader = prf.DataLoader(raw, transforms=T.Compose([T.Resize(224, 224)]), device="auto")
for batch in loader:        # arrays already transformed + on the chosen device/framework
    ...
```
"""

from __future__ import annotations

from .backend import resolve_device


def _is_image(arr) -> bool:
    """Heuristic: a camera-frame batch is `[N, H, W, C]` with C in {1, 3, 4}."""
    return getattr(arr, "ndim", 0) == 4 and arr.shape[-1] in (1, 3, 4)


def _to_device(batch: dict, device: str) -> dict:
    if device == "cpu":
        return batch  # leave as NumPy
    if device == "mlx":
        import mlx.core as mx  # noqa: PLC0415

        return {k: mx.array(v) for k, v in batch.items()}
    if device in ("cuda", "mps"):
        import torch  # noqa: PLC0415

        return {k: torch.as_tensor(v).to(device) for k, v in batch.items()}
    raise ValueError(f"unsupported device {device!r}")


class DataLoader:
    """Iterates ``inner``, applying ``transforms`` to image entries and moving to ``device``.

    ``inner`` should be a NumPy-output loader (the default). ``transforms`` is applied to every
    camera-frame array (``[N, H, W, C]``); other entries (state/action) pass through unchanged.
    """

    def __init__(self, inner, transforms=None, device: str = "cpu"):
        self._inner = inner
        self._transforms = transforms
        self.device = resolve_device(device)

    def __iter__(self):
        for batch in self._inner:
            if self._transforms is not None:
                for key, value in list(batch.items()):
                    if _is_image(value):
                        batch[key] = self._transforms(value)
            yield _to_device(batch, self.device)

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def position(self) -> int:
        """Frames/rows consumed so far this epoch (delegated to the wrapped loader)."""
        return self._inner.position

    def __repr__(self) -> str:
        return f"DataLoader(device={self.device!r}, transforms={self._transforms!r})"
