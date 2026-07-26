"""Unified tensor output abstraction for cross-framework compatibility.

Provides consistent API for converting PyRoboFrames outputs to numpy/torch/jax/mlx/tensorflow
without framework-specific branching in downstream code.

Example:
    >>> adapter = ToTensorAdapter(framework="auto")
    >>> batch_numpy = loader.get_batch()
    >>> batch_torch = adapter.to_tensor(batch_numpy)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import mlx.core
    import torch


class TensorAdapter:
    """Base class for framework-specific tensor conversions."""

    def to_tensor(self, data: np.ndarray) -> Any:
        """Convert NumPy array to framework tensor."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if framework is installed."""
        raise NotImplementedError

    def get_device(self) -> str | None:
        """Get current device name (e.g., 'cuda', 'mps')."""
        return None


class NumpyAdapter(TensorAdapter):
    """NumPy (no-op) adapter."""

    def to_tensor(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data)

    def is_available(self) -> bool:
        return True


class TorchAdapter(TensorAdapter):
    """PyTorch adapter with device awareness."""

    def __init__(self, device: str | None = None):
        self.device = device or "cpu"

    def to_tensor(self, data: np.ndarray) -> torch.Tensor:
        import torch

        tensor = torch.from_numpy(data.copy() if data.dtype == np.float64 else data)
        if self.device and self.device != "cpu":
            tensor = tensor.to(self.device)
        return tensor

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def get_device(self) -> str | None:
        return self.device


class JAXAdapter(TensorAdapter):
    """JAX adapter."""

    def to_tensor(self, data: np.ndarray) -> Any:
        import jax.numpy as jnp

        return jnp.asarray(data)

    def is_available(self) -> bool:
        try:
            import jax  # noqa: F401
            return True
        except ImportError:
            return False


class MLXAdapter(TensorAdapter):
    """MLX adapter for Apple Silicon."""

    def to_tensor(self, data: np.ndarray) -> mlx.core.array:
        import mlx.core as mx

        # MLX works best with float32
        if data.dtype == np.float64:
            data = data.astype(np.float32)
        return mx.array(data)

    def is_available(self) -> bool:
        try:
            import mlx.core  # noqa: F401
            return True
        except ImportError:
            return False

    def get_device(self) -> str | None:
        return "mps"


class TensorFlowAdapter(TensorAdapter):
    """TensorFlow adapter."""

    def __init__(self, device: str | None = None):
        self.device = device

    def to_tensor(self, data: np.ndarray) -> Any:
        import tensorflow as tf

        tensor = tf.convert_to_tensor(data)
        if self.device:
            with tf.device(self.device):
                tensor = tf.identity(tensor)
        return tensor

    def is_available(self) -> bool:
        try:
            import tensorflow  # noqa: F401
            return True
        except ImportError:
            return False

    def get_device(self) -> str | None:
        return self.device


class ToTensorAdapter:
    """Unified tensor adapter with auto-framework detection.

    Automatically selects the best available framework or uses specified one.
    Priority: CUDA → MLX (Apple) → PyTorch → JAX → TensorFlow → NumPy
    """

    FRAMEWORK_PRIORITY = ["cuda", "mlx", "torch", "jax", "tensorflow", "numpy"]

    def __init__(
        self,
        framework: Literal["auto", "numpy", "torch", "jax", "mlx", "tensorflow"] = "auto",
        device: str | None = None,
    ):
        """Initialize unified adapter.

        Args:
            framework: "auto" for best available, or explicit framework name
            device: Target device (e.g., "cuda:0", "cpu", "mps"). Ignored for MLX/JAX
        """
        self.framework = framework
        self.device = device
        self.adapter = self._select_adapter()

    def _select_adapter(self) -> TensorAdapter:
        """Select best available adapter."""
        if self.framework != "auto":
            return self._get_adapter_for_framework(self.framework)

        # Auto-detect framework
        for fw_name in self.FRAMEWORK_PRIORITY:
            try:
                adapter = self._get_adapter_for_framework(fw_name)
                if adapter.is_available():
                    return adapter
            except Exception:
                # Skip unavailable frameworks
                continue

        # Fallback to NumPy
        return NumpyAdapter()

    def _get_adapter_for_framework(self, name: str) -> TensorAdapter:
        """Get adapter for specific framework."""
        if name == "numpy":
            return NumpyAdapter()
        elif name == "torch":
            device = self.device or "cpu"
            return TorchAdapter(device=device)
        elif name == "jax":
            return JAXAdapter()
        elif name == "mlx":
            return MLXAdapter()
        elif name == "tensorflow":
            return TensorFlowAdapter(device=self.device)
        elif name == "cuda":
            # Try CUDA torch first, fall back to basic torch
            try:
                import torch

                if torch.cuda.is_available():
                    return TorchAdapter(device=self.device or "cuda:0")
            except ImportError:
                pass
            raise ValueError(f"CUDA requested but PyTorch not available")
        else:
            raise ValueError(f"Unknown framework: {name}")

    def to_tensor(self, data: np.ndarray) -> Any:
        """Convert NumPy array to framework tensor."""
        return self.adapter.to_tensor(data)

    def batch_to_tensors(self, batch: dict[str, np.ndarray]) -> dict[str, Any]:
        """Convert all arrays in a batch dict.

        Args:
            batch: Dict mapping feature name → NumPy array

        Returns:
            Dict with same keys, values converted to framework tensors
        """
        return {key: self.to_tensor(value) for key, value in batch.items()}

    def get_framework_name(self) -> str:
        """Get the name of the selected framework."""
        adapter_class = type(self.adapter).__name__
        if adapter_class == "NumpyAdapter":
            return "numpy"
        elif adapter_class == "TorchAdapter":
            return "torch"
        elif adapter_class == "JAXAdapter":
            return "jax"
        elif adapter_class == "MLXAdapter":
            return "mlx"
        elif adapter_class == "TensorFlowAdapter":
            return "tensorflow"
        return "unknown"

    def get_device(self) -> str | None:
        """Get current device."""
        return self.adapter.get_device()

    def __repr__(self) -> str:
        return f"ToTensorAdapter(framework={self.get_framework_name()}, device={self.get_device()})"


def detect_best_framework() -> str:
    """Detect the best available framework for this system."""
    adapter = ToTensorAdapter(framework="auto")
    return adapter.get_framework_name()


def create_adapter_for_device(device: str | None = None) -> ToTensorAdapter:
    """Create an adapter optimized for the given device.

    Args:
        device: "cuda", "mps", "cpu", or None for auto-detect

    Returns:
        Configured ToTensorAdapter
    """
    if device == "cuda":
        return ToTensorAdapter(framework="torch", device="cuda:0")
    elif device == "mps":
        # Try MLX first (native Apple Silicon), fall back to torch
        adapter = ToTensorAdapter(framework="mlx")
        if adapter.adapter.is_available():
            return adapter
        return ToTensorAdapter(framework="torch", device="mps")
    elif device == "cpu":
        return ToTensorAdapter(framework="numpy")
    else:
        return ToTensorAdapter(framework="auto")
