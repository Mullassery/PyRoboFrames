"""GPU acceleration for video transforms and image processing.

Supports NVIDIA (CuPy), Apple Silicon (MLX), and CPU fallbacks.
"""

import numpy as np
from typing import Optional, Union, Dict, Any
import warnings


class GPUTransforms:
    """GPU-accelerated image transforms using CuPy or MLX."""

    def __init__(self, device: str = "auto"):
        """Initialize GPU transforms.

        Args:
            device: "cuda", "mlx", or "auto" (picks best available)
        """
        self.device = device
        self.backend = self._resolve_backend()

    def _resolve_backend(self) -> str:
        """Resolve the best available GPU backend."""
        if self.device == "cuda":
            try:
                import cupy  # noqa: F401
                return "cupy"
            except ImportError:
                warnings.warn("CuPy not available, falling back to NumPy")
                return "numpy"

        if self.device == "mlx" or self.device == "auto":
            try:
                import mlx.core as mx  # noqa: F401
                if self.device == "mlx":
                    return "mlx"
                # For auto, MLX is secondary to CuPy
            except ImportError:
                pass

        if self.device == "auto":
            try:
                import cupy  # noqa: F401
                return "cupy"
            except ImportError:
                try:
                    import mlx.core  # noqa: F401
                    return "mlx"
                except ImportError:
                    return "numpy"

        return "numpy"

    def resize_cupy(self, image: np.ndarray, size: tuple, interpolation: str = "bilinear") -> np.ndarray:
        """Resize using CuPy (NVIDIA GPU)."""
        try:
            import cupy as cp
            from cupyx.scipy import ndimage

            gpu_image = cp.asarray(image)

            if interpolation == "bilinear":
                order = 1
            elif interpolation == "nearest":
                order = 0
            else:
                order = 1

            scale_h = size[0] / image.shape[0]
            scale_w = size[1] / image.shape[1]

            # Use zoom for resizing
            resized = ndimage.zoom(gpu_image, (scale_h, scale_w, 1), order=order)
            return cp.asnumpy(resized).astype(np.uint8)
        except Exception as e:
            warnings.warn(f"CuPy resize failed: {e}, falling back to NumPy")
            return self.resize_numpy(image, size, interpolation)

    def resize_mlx(self, image: np.ndarray, size: tuple, interpolation: str = "bilinear") -> np.ndarray:
        """Resize using MLX (Apple Silicon GPU)."""
        try:
            import mlx.core as mx

            gpu_image = mx.array(image.astype(np.float32))
            h, w = image.shape[:2]

            # Simple bilinear interpolation via transpose and scaling
            scale_h = mx.array([size[0] / h])
            scale_w = mx.array([size[1] / w])

            # Nearest neighbor fallback for simplicity
            indices_h = mx.floor(mx.arange(size[0]) / scale_h).astype(mx.int32)
            indices_w = mx.floor(mx.arange(size[1]) / scale_w).astype(mx.int32)

            # Gather operation
            output = mx.zeros((size[0], size[1], image.shape[2]), dtype=mx.uint8)
            for i in range(size[0]):
                for j in range(size[1]):
                    h_idx = mx.minimum(indices_h[i], h - 1)
                    w_idx = mx.minimum(indices_w[j], w - 1)
                    output[i, j] = gpu_image[h_idx, w_idx]

            return np.array(output, dtype=np.uint8)
        except Exception as e:
            warnings.warn(f"MLX resize failed: {e}, falling back to NumPy")
            return self.resize_numpy(image, size, interpolation)

    def resize_numpy(self, image: np.ndarray, size: tuple, interpolation: str = "bilinear") -> np.ndarray:
        """Resize using NumPy (CPU fallback)."""
        from scipy import ndimage

        if interpolation == "bilinear":
            order = 1
        elif interpolation == "nearest":
            order = 0
        else:
            order = 1

        scale_h = size[0] / image.shape[0]
        scale_w = size[1] / image.shape[1]

        resized = ndimage.zoom(image, (scale_h, scale_w, 1), order=order)
        return resized.astype(np.uint8)

    def resize(self, image: np.ndarray, size: tuple, interpolation: str = "bilinear") -> np.ndarray:
        """Resize image to target size."""
        if self.backend == "cupy":
            return self.resize_cupy(image, size, interpolation)
        elif self.backend == "mlx":
            return self.resize_mlx(image, size, interpolation)
        else:
            return self.resize_numpy(image, size, interpolation)

    def normalize_cupy(self, image: np.ndarray, mean: list, std: list) -> np.ndarray:
        """Normalize using CuPy."""
        try:
            import cupy as cp

            gpu_image = cp.asarray(image, dtype=cp.float32) / 255.0
            gpu_mean = cp.array(mean, dtype=cp.float32).reshape(1, 1, 3)
            gpu_std = cp.array(std, dtype=cp.float32).reshape(1, 1, 3)

            normalized = (gpu_image - gpu_mean) / (gpu_std + 1e-8)
            return cp.asnumpy(normalized).astype(np.float32)
        except Exception as e:
            warnings.warn(f"CuPy normalize failed: {e}, falling back to NumPy")
            return self.normalize_numpy(image, mean, std)

    def normalize_mlx(self, image: np.ndarray, mean: list, std: list) -> np.ndarray:
        """Normalize using MLX."""
        try:
            import mlx.core as mx

            gpu_image = mx.array(image.astype(np.float32)) / 255.0
            gpu_mean = mx.array(mean, dtype=mx.float32).reshape(1, 1, 3)
            gpu_std = mx.array(std, dtype=mx.float32).reshape(1, 1, 3)

            normalized = (gpu_image - gpu_mean) / (gpu_std + 1e-8)
            return np.array(normalized, dtype=np.float32)
        except Exception as e:
            warnings.warn(f"MLX normalize failed: {e}, falling back to NumPy")
            return self.normalize_numpy(image, mean, std)

    def normalize_numpy(self, image: np.ndarray, mean: list, std: list) -> np.ndarray:
        """Normalize using NumPy."""
        image = image.astype(np.float32) / 255.0
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

        return (image - mean) / (std + 1e-8)

    def normalize(self, image: np.ndarray, mean: list, std: list) -> np.ndarray:
        """Normalize image."""
        if self.backend == "cupy":
            return self.normalize_cupy(image, mean, std)
        elif self.backend == "mlx":
            return self.normalize_mlx(image, mean, std)
        else:
            return self.normalize_numpy(image, mean, std)


class OpticalFlowEstimator:
    """Optical flow estimation for temporal consistency."""

    @staticmethod
    def estimate_lucas_kanade(frame1: np.ndarray, frame2: np.ndarray, window_size: int = 15) -> np.ndarray:
        """Estimate optical flow using Lucas-Kanade method."""
        try:
            import cv2
        except ImportError:
            warnings.warn("OpenCV not available, using gradient-based fallback")
            return OpticalFlowEstimator._estimate_gradient_flow(frame1, frame2)

        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray1, gray2 = frame1, frame2

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    @staticmethod
    def _estimate_gradient_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Simple gradient-based optical flow fallback."""
        if len(frame1.shape) == 3:
            gray1 = np.mean(frame1, axis=2).astype(np.float32)
            gray2 = np.mean(frame2, axis=2).astype(np.float32)
        else:
            gray1 = frame1.astype(np.float32)
            gray2 = frame2.astype(np.float32)

        # Compute gradients
        from scipy import ndimage

        Ix = ndimage.sobel(gray1, axis=1)
        Iy = ndimage.sobel(gray1, axis=0)
        It = gray2 - gray1

        # Initialize flow
        h, w = gray1.shape
        flow = np.zeros((h, w, 2), dtype=np.float32)

        # Simple motion estimation
        motion_mag = np.abs(It).mean()
        if motion_mag > 1.0:
            flow[:, :, 0] = -It / (np.abs(Ix) + 1e-8)
            flow[:, :, 1] = -It / (np.abs(Iy) + 1e-8)

        return flow


class TemporalFilter:
    """Temporal filtering for video stitching consistency."""

    @staticmethod
    def apply_temporal_smoothing(frames: list, alpha: float = 0.7) -> np.ndarray:
        """Apply exponential moving average for temporal smoothing.

        Args:
            frames: List of [H, W, 3] frames
            alpha: Smoothing factor (0-1)

        Returns:
            Smoothed frame stack [T, H, W, 3]
        """
        if not frames:
            return np.array([])

        frames = np.array(frames, dtype=np.float32)
        smoothed = np.zeros_like(frames)
        smoothed[0] = frames[0]

        for t in range(1, len(frames)):
            smoothed[t] = alpha * frames[t] + (1 - alpha) * smoothed[t - 1]

        return smoothed.astype(np.uint8)

    @staticmethod
    def apply_median_filter(frames: list, kernel_size: int = 3) -> np.ndarray:
        """Apply median filtering across time.

        Args:
            frames: List of [H, W, 3] frames
            kernel_size: Temporal window size

        Returns:
            Filtered frame stack [T, H, W, 3]
        """
        from scipy import ndimage

        frames = np.array(frames, dtype=np.uint8)

        if kernel_size == 1:
            return frames

        # Apply median filter along time axis
        pad = kernel_size // 2
        padded = np.pad(frames, ((pad, pad), (0, 0), (0, 0), (0, 0)), mode='edge')

        output = np.zeros_like(frames)
        for t in range(len(frames)):
            window = padded[t:t + kernel_size]
            output[t] = np.median(window, axis=0).astype(np.uint8)

        return output


class MLXTransforms:
    """Native on-device MLX transforms for Apple Silicon without host copies.

    Keeps data in unified memory (no GPU ↔ CPU transfers for intermediate results).
    """

    def __init__(self):
        """Initialize MLX transforms."""
        try:
            import mlx.core as mx
            self.mx = mx
            self.available = True
        except ImportError:
            self.available = False

    def is_available(self) -> bool:
        """Check if MLX is available."""
        return self.available

    def resize(
        self,
        image: Any,
        size: tuple[int, int],
        interpolation: str = "nearest",
    ) -> Any:
        """Resize image to target size on device.

        Args:
            image: MLX array [H, W, C] or NumPy array
            size: Target (height, width)
            interpolation: "nearest" or "linear"

        Returns:
            MLX array with new shape
        """
        if not self.available:
            raise RuntimeError("MLX not available")

        import mlx.core as mx

        # Convert to MLX if needed
        if isinstance(image, np.ndarray):
            image = mx.array(image.astype(np.float32))
        else:
            image = mx.astype(image, mx.float32)

        h, w = image.shape[0], image.shape[1]
        target_h, target_w = size

        scale_h = mx.array(h / target_h, dtype=mx.float32)
        scale_w = mx.array(w / target_w, dtype=mx.float32)

        # Compute source indices for each target pixel
        target_indices_h = mx.arange(target_h, dtype=mx.float32)
        target_indices_w = mx.arange(target_w, dtype=mx.float32)

        src_h = (target_indices_h * scale_h).astype(mx.int32)
        src_w = (target_indices_w * scale_w).astype(mx.int32)

        # Clamp to valid range
        src_h = mx.clip(src_h, 0, h - 1)
        src_w = mx.clip(src_w, 0, w - 1)

        # Gather operation: for each target location, copy from nearest source
        output = mx.zeros((target_h, target_w) + image.shape[2:], dtype=mx.float32)
        for i in range(target_h):
            for j in range(target_w):
                output[i, j] = image[src_h[i], src_w[j]]

        return output

    def normalize(
        self,
        image: Any,
        mean: list[float] | np.ndarray = None,
        std: list[float] | np.ndarray = None,
    ) -> Any:
        """Normalize image (subtract mean, divide by std) on device.

        Args:
            image: MLX array [H, W, C] or [B, H, W, C]
            mean: Per-channel mean (default: [0.5, 0.5, 0.5])
            std: Per-channel std (default: [0.5, 0.5, 0.5])

        Returns:
            Normalized MLX array
        """
        if not self.available:
            raise RuntimeError("MLX not available")

        import mlx.core as mx

        if mean is None:
            mean = [0.5, 0.5, 0.5]
        if std is None:
            std = [0.5, 0.5, 0.5]

        mean = mx.array(mean, dtype=mx.float32)
        std = mx.array(std, dtype=mx.float32)

        # Reshape for broadcasting
        if image.ndim == 4:  # [B, H, W, C]
            mean = mean.reshape(1, 1, 1, -1)
            std = std.reshape(1, 1, 1, -1)
        else:  # [H, W, C]
            mean = mean.reshape(1, 1, -1)
            std = std.reshape(1, 1, -1)

        image = mx.astype(image, mx.float32)
        return (image - mean) / std

    def compose(self, image: Any, transforms: list[dict]) -> Any:
        """Apply a sequence of transforms in-device.

        Args:
            image: MLX array
            transforms: List of {"op": name, "args": {...}} dicts

        Returns:
            Transformed MLX array

        Example:
            >>> image = mx.array(...)
            >>> transforms = [
            ...     {"op": "normalize", "args": {"mean": [0.5]*3, "std": [0.5]*3}},
            ...     {"op": "resize", "args": {"size": (224, 224)}},
            ... ]
            >>> result = mlx_transforms.compose(image, transforms)
        """
        result = image
        for transform in transforms:
            op = transform.get("op")
            args = transform.get("args", {})

            if op == "normalize":
                result = self.normalize(result, **args)
            elif op == "resize":
                result = self.resize(result, **args)
            elif op == "cast":
                dtype = args.get("dtype", self.mx.float32)
                result = self.mx.astype(result, dtype)
            else:
                raise ValueError(f"Unknown transform: {op}")

        return result


class MPSTransforms:
    """On-device PyTorch transforms for Apple Silicon via MPS backend."""

    def __init__(self):
        """Initialize MPS transforms."""
        try:
            import torch
            self.torch = torch
            self.available = torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            self.available = False

    def is_available(self) -> bool:
        """Check if MPS is available."""
        return self.available

    def resize(
        self,
        image: Any,
        size: tuple[int, int],
        interpolation: str = "bilinear",
    ) -> Any:
        """Resize image on MPS device.

        Args:
            image: Torch tensor [H, W, C] or [B, H, W, C], or NumPy array
            size: Target (height, width)
            interpolation: "nearest", "bilinear", "bicubic"

        Returns:
            Torch tensor on MPS
        """
        if not self.available:
            raise RuntimeError("MPS not available")

        import torch.nn.functional as F

        # Convert to torch tensor
        if isinstance(image, np.ndarray):
            tensor = self.torch.from_numpy(image).float()
        else:
            tensor = image.float()

        # Add batch dim if needed
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # [B, H, W, C]

        # Reorder to [B, C, H, W] for PyTorch
        tensor = tensor.permute(0, 3, 1, 2).contiguous()

        # Move to MPS
        device = self.torch.device("mps")
        tensor = tensor.to(device)

        # Resize
        mode = "nearest" if interpolation == "nearest" else "bilinear"
        resized = F.interpolate(tensor, size=size, mode=mode, align_corners=False if mode == "bilinear" else None)

        # Reorder back to [B, H, W, C]
        resized = resized.permute(0, 2, 3, 1).contiguous()

        return resized

    def normalize(
        self,
        image: Any,
        mean: list[float] | np.ndarray = None,
        std: list[float] | np.ndarray = None,
    ) -> Any:
        """Normalize image on MPS device.

        Args:
            image: Torch tensor or NumPy array
            mean: Per-channel mean
            std: Per-channel std

        Returns:
            Normalized Torch tensor on MPS
        """
        if not self.available:
            raise RuntimeError("MPS not available")

        if mean is None:
            mean = [0.5, 0.5, 0.5]
        if std is None:
            std = [0.5, 0.5, 0.5]

        # Convert to torch
        if isinstance(image, np.ndarray):
            tensor = self.torch.from_numpy(image).float()
        else:
            tensor = image.float()

        mean = self.torch.tensor(mean, dtype=self.torch.float32).to("mps")
        std = self.torch.tensor(std, dtype=self.torch.float32).to("mps")

        # Reshape for broadcasting
        if tensor.ndim == 4:  # [B, H, W, C]
            mean = mean.view(1, 1, 1, -1)
            std = std.view(1, 1, 1, -1)
        else:  # [H, W, C]
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)

        tensor = tensor.to("mps")
        return (tensor - mean) / std
