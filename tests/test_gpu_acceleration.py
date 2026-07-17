"""Tests for GPU acceleration module."""

import numpy as np
import pytest
from pyroboframes.gpu_acceleration import GPUTransforms, OpticalFlowEstimator, TemporalFilter


class TestGPUTransforms:
    """Test GPU transform operations."""

    def test_resize_numpy(self):
        """Test NumPy-based resize."""
        transforms = GPUTransforms(device="cpu")
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        resized = transforms.resize(image, (240, 320))

        assert resized.shape == (240, 320, 3)
        assert resized.dtype == np.uint8

    def test_normalize_numpy(self):
        """Test NumPy-based normalization."""
        transforms = GPUTransforms(device="cpu")
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        normalized = transforms.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        assert normalized.shape == (480, 640, 3)
        assert normalized.dtype == np.float32
        # Check roughly normalized range
        assert normalized.min() < 0
        assert normalized.max() > 0

    def test_resize_maintains_aspect(self):
        """Test that resize produces correct output shape."""
        transforms = GPUTransforms(device="cpu")
        image = np.ones((100, 200, 3), dtype=np.uint8)

        resized = transforms.resize(image, (50, 100))

        assert resized.shape == (50, 100, 3)

    def test_normalize_range(self):
        """Test that normalization produces valid range."""
        transforms = GPUTransforms(device="cpu")
        image = np.full((100, 100, 3), 128, dtype=np.uint8)

        normalized = transforms.normalize(
            image,
            mean=[0.5, 0.5, 0.5],
            std=[0.2, 0.2, 0.2]
        )

        # 128/255 ≈ 0.502, normalized ≈ (0.502 - 0.5) / 0.2 ≈ 0.1
        assert np.isfinite(normalized).all()


class TestOpticalFlow:
    """Test optical flow estimation."""

    def test_lucas_kanade_fallback(self):
        """Test Lucas-Kanade with fallback."""
        frame1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        flow = OpticalFlowEstimator.estimate_lucas_kanade(frame1, frame2)

        assert flow.shape == (100, 100, 2)
        assert flow.dtype in [np.float32, np.float64]

    def test_gradient_flow_consistency(self):
        """Test gradient flow produces valid output."""
        # Create synthetic motion
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame1[40:60, 40:60] = 255

        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[42:62, 40:60] = 255  # Shifted down

        flow = OpticalFlowEstimator._estimate_gradient_flow(frame1, frame2)

        assert flow.shape == (100, 100, 2)
        # Motion should be detected in center
        center_flow = flow[50, 50]
        assert np.isfinite(center_flow).all()


class TestTemporalFilter:
    """Test temporal filtering."""

    def test_exponential_smoothing(self):
        """Test exponential moving average smoothing."""
        # Create noisy temporal sequence
        frames = [
            np.full((10, 10, 3), i * 50, dtype=np.uint8)
            for i in range(5)
        ]

        smoothed = TemporalFilter.apply_temporal_smoothing(frames, alpha=0.7)

        assert smoothed.shape == (5, 10, 10, 3)
        assert smoothed.dtype == np.uint8
        # First frame should match
        assert np.allclose(smoothed[0], frames[0])

    def test_median_filtering(self):
        """Test median filtering over time."""
        frames = [
            np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
            for _ in range(5)
        ]

        filtered = TemporalFilter.apply_median_filter(frames, kernel_size=3)

        assert filtered.shape == (5, 10, 10, 3)
        assert filtered.dtype == np.uint8

    def test_smoothing_reduces_variance(self):
        """Test that smoothing reduces temporal variance."""
        # Create noisy frames
        base = 128
        frames = [
            np.full((10, 10, 3), base + np.random.randint(-30, 30), dtype=np.uint8)
            for _ in range(10)
        ]

        smoothed = TemporalFilter.apply_temporal_smoothing(frames, alpha=0.9)

        # Variance should be reduced
        original_var = np.var([f.mean() for f in frames])
        smoothed_var = np.var([f.mean() for f in smoothed])

        assert smoothed_var < original_var

    def test_single_frame_filtering(self):
        """Test filtering with single frame."""
        frames = [np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)]

        filtered = TemporalFilter.apply_median_filter(frames, kernel_size=1)

        assert np.array_equal(filtered[0], frames[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
