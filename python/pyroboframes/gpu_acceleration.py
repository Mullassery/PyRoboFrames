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
