"""Tests for Phase 4a (GPU Acceleration) and Phase 4b (Temporal Consistency)."""

import numpy as np
import pytest

from pyroboframes.automotive import CylindricalStitcher, get_waymo_layout
from pyroboframes.automotive.gpu_backend import (
    get_gpu_backend,
    NumPyBackend,
    CuPyBackend,
    MLXBackend,
)
from pyroboframes.automotive.optical_flow import (
    OpticalFlow,
    SeamTracker,
    temporal_blend,
)


class TestPhase4aGPUBackends:
    """Test Phase 4a: GPU acceleration."""

    def test_numpy_backend_available(self):
        """Test NumPy backend always available."""
        backend = get_gpu_backend("cpu")
        assert isinstance(backend, NumPyBackend)
        assert backend.name == "cpu"

    def test_gpu_backend_auto_detect(self):
        """Test GPU backend auto-detection."""
        backend = get_gpu_backend(None)
        assert backend is not None
        assert backend.name in ["cuda", "mlx", "cpu"]

    def test_numpy_gaussian_blur(self):
        """Test Gaussian blur on NumPy backend."""
        backend = NumPyBackend()
        image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

        blurred = backend.gaussian_blur(image, sigma=1.0)

        assert blurred.shape == image.shape
        assert blurred.dtype == np.float32

    def test_numpy_downsample(self):
        """Test downsampling on NumPy backend."""
        backend = NumPyBackend()
        image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

        downsampled = backend.downsample(image)

        assert downsampled.shape == (50, 50, 3)

    def test_numpy_upsample(self):
        """Test upsampling on NumPy backend."""
        backend = NumPyBackend()
        image = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)

        upsampled = backend.upsample(image, (100, 100, 3))

        assert upsampled.shape == (100, 100, 3)

    def test_backend_switching(self):
        """Test switching between backends."""
        backends = [NumPyBackend()]

        # Try to add GPU backends if available
        try:
            backends.append(CuPyBackend())
        except ImportError:
            pass

        try:
            backends.append(MLXBackend())
        except ImportError:
            pass

        for backend in backends:
            assert backend.name in ["cpu", "cuda", "mlx"]


class TestPhase4bTemporalConsistency:
    """Test Phase 4b: Temporal consistency."""

    def test_optical_flow_farneback(self):
        """Test Farneback optical flow computation."""
        pytest.importorskip("cv2")

        flow_model = OpticalFlow("farneback")

        frame0 = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        frame1 = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

        flow = flow_model.compute(frame0, frame1)

        assert flow.shape == (100, 100, 2)
        assert flow.dtype == np.float32

    def test_seam_tracker_initialization(self):
        """Test seam tracker initialization."""
        tracker = SeamTracker(history_size=5)

        assert tracker.seam_estimate is None
        assert len(tracker.history) == 0

    def test_seam_tracker_update(self):
        """Test seam tracker update with Kalman filtering."""
        tracker = SeamTracker(history_size=5)

        # Create test seam
        seam = np.array([100, 110, 120, 130, 140])

        # First update (initialize)
        tracked = tracker.track_seam(seam, flow_t=None)
        assert tracked.shape == seam.shape

        # Second update (should smooth)
        seam2 = np.array([105, 115, 125, 135, 145])
        tracked2 = tracker.track_seam(seam2, flow_t=None)

        # Tracked should be between seam and seam2 (smoothed)
        assert np.all(tracked2 >= np.minimum(seam, seam2) - 10)
        assert np.all(tracked2 <= np.maximum(seam, seam2) + 10)

    def test_seam_tracker_reset(self):
        """Test seam tracker reset."""
        tracker = SeamTracker()

        seam = np.array([100, 110, 120])
        tracker.track_seam(seam)

        assert len(tracker.history) == 1

        tracker.reset()

        assert len(tracker.history) == 0
        assert tracker.seam_estimate is None

    def test_temporal_blend(self):
        """Test temporal blending between frames."""
        pan0 = (np.ones((100, 100, 3)) * 100).astype(np.uint8)
        pan1 = (np.ones((100, 100, 3)) * 200).astype(np.uint8)

        blended = temporal_blend(pan0, pan1, alpha=0.5)

        assert blended.shape == pan0.shape
        assert blended.dtype == np.uint8
        # Should be roughly in between
        assert np.all(blended > 100)
        assert np.all(blended < 200)

    def test_temporal_blend_alpha_values(self):
        """Test temporal blending with different alpha values."""
        pan0 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        pan1 = np.ones((10, 10, 3), dtype=np.uint8) * 200

        # Alpha=0 should return pan0
        blended_0 = temporal_blend(pan0, pan1, alpha=0.0)
        assert np.allclose(blended_0, pan0, atol=1)

        # Alpha=1 should return pan1
        blended_1 = temporal_blend(pan0, pan1, alpha=1.0)
        assert np.allclose(blended_1, pan1, atol=1)

        # Alpha=0.5 should be roughly midpoint
        blended_half = temporal_blend(pan0, pan1, alpha=0.5)
        assert np.all(blended_half > 100)
        assert np.all(blended_half < 200)


class TestCylindricalStitcherWithGPU:
    """Test CylindricalStitcher with GPU support (Phase 4a)."""

    def test_stitcher_with_cpu_backend(self):
        """Test stitcher explicitly using CPU backend."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout, device="cpu")

        assert stitcher.gpu_backend.name == "cpu"

    def test_stitcher_auto_gpu_detection(self):
        """Test stitcher auto-detects GPU backend."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout, device=None)

        assert stitcher.gpu_backend.name in ["cuda", "mlx", "cpu"]

    def test_stitcher_repr_with_device(self):
        """Test stitcher repr includes device info."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout, device="cpu")

        repr_str = repr(stitcher)
        assert "device='cpu'" in repr_str

    def test_stitcher_basic_stitch(self):
        """Test basic stitching works on GPU backend."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout, device="cpu")

        frames = {
            cam: (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
            for cam in layout.cameras.keys()
        }

        panorama = stitcher.stitch(frames)

        assert panorama.shape == (1, 480, 1728, 3)
        assert panorama.dtype == np.uint8


class TestCylindricalStitcherTemporal:
    """Test CylindricalStitcher with temporal consistency (Phase 4b)."""

    def test_stitcher_with_temporal(self):
        """Test stitcher with temporal consistency enabled."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(
            layout, device="cpu", use_temporal_consistency=True
        )

        assert stitcher.use_temporal is True
        assert stitcher.seam_tracker is not None
        assert stitcher.optical_flow is not None

    def test_temporal_repr(self):
        """Test stitcher repr includes temporal flag."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(
            layout, use_temporal_consistency=True
        )

        repr_str = repr(stitcher)
        assert "temporal=True" in repr_str

    def test_stitch_with_temporal_blending(self):
        """Test stitching with temporal blending."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(
            layout, device="cpu", use_temporal_consistency=True, temporal_alpha=0.3
        )

        frames1 = {
            cam: (np.ones((720, 1280, 3)) * 100).astype(np.uint8)
            for cam in layout.cameras.keys()
        }

        frames2 = {
            cam: (np.ones((720, 1280, 3)) * 150).astype(np.uint8)
            for cam in layout.cameras.keys()
        }

        # Stitch first frame
        pan1 = stitcher.stitch(frames1)
        assert pan1.dtype == np.uint8

        # Stitch second frame (should apply temporal blending)
        pan2 = stitcher.stitch(frames2)
        assert pan2.dtype == np.uint8

        # Second should be different due to blending
        assert not np.allclose(pan1, pan2)

    def test_stitch_temporal_sequence(self):
        """Test stitching full temporal sequence."""
        layout = get_waymo_layout()

        # Test without optical flow (will gracefully degrade)
        stitcher = CylindricalStitcher(
            layout, device="cpu", use_temporal_consistency=True
        )

        # Create 5-frame sequence
        num_frames = 5
        frames_seq = {
            cam: (np.random.rand(num_frames, 720, 1280, 3) * 255).astype(np.uint8)
            for cam in layout.cameras.keys()
        }

        # This should work even without OpenCV
        try:
            panoramas = stitcher.stitch_temporal_sequence(frames_seq)
            assert panoramas.shape == (5, 480, 1728, 3)
            assert panoramas.dtype == np.uint8
        except ImportError:
            pytest.skip("OpenCV not installed for optical flow")

    def test_temporal_alpha_parameter(self):
        """Test different temporal alpha values."""
        layout = get_waymo_layout()

        for alpha in [0.1, 0.3, 0.5]:
            stitcher = CylindricalStitcher(
                layout,
                device="cpu",
                use_temporal_consistency=True,
                temporal_alpha=alpha,
            )
            assert stitcher.temporal_alpha == alpha


class TestGPUBenchmark:
    """Performance tests for Phase 4a GPU acceleration."""

    def test_backend_speed_comparison(self):
        """Compare speeds of different backends (qualitative)."""
        import time

        image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

        backends = [NumPyBackend()]

        try:
            backends.append(CuPyBackend())
        except ImportError:
            pass

        for backend in backends:
            start = time.time()
            for _ in range(10):
                backend.gaussian_blur(image, sigma=1.0)
            elapsed = time.time() - start

            print(f"{backend.name}: {elapsed:.3f}s for 10 iterations")

    def test_panorama_stitching_speed(self):
        """Benchmark panorama stitching (not for CI, informational only)."""
        import time

        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout, device="cpu")

        frames = {
            cam: (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
            for cam in layout.cameras.keys()
        }

        start = time.time()
        for _ in range(3):
            _ = stitcher.stitch(frames)
        elapsed = time.time() - start

        fps = 3 / elapsed
        print(f"M3 CPU stitching: {fps:.1f} FPS")
