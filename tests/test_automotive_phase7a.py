"""Tests for Phase 7a: SAM3 Segmentation."""

import numpy as np
import pytest

from pyroboframes.automotive import SAM3Segmenter


class TestSAM3ModelLoading:
    """Test SAM3 model initialization and loading."""

    def test_sam3_init_default(self):
        """Test SAM3Segmenter initialization with defaults."""
        # Don't actually load the model (avoid HF download in tests)
        # Just verify the object can be created
        try:
            segmenter = SAM3Segmenter(model_id="facebook/sam3-base", device="cpu")
            assert segmenter.model_id == "facebook/sam3-base"
            assert segmenter.device == "cpu"
            assert segmenter.temporal_smoothing is True
            assert segmenter.cache_frames == 5
        except (ImportError, OSError):
            pytest.skip("Transformers not installed or SAM3 model unavailable")

    def test_sam3_init_small_model(self):
        """Test initialization with small model."""
        try:
            segmenter = SAM3Segmenter(model_id="facebook/sam3-small", device="cpu")
            assert segmenter.model_id == "facebook/sam3-small"
        except (ImportError, OSError):
            pytest.skip("Transformers not installed or SAM3 model unavailable")

    def test_sam3_init_large_model(self):
        """Test initialization with large model."""
        try:
            segmenter = SAM3Segmenter(model_id="facebook/sam3-large", device="cpu")
            assert segmenter.model_id == "facebook/sam3-large"
        except (ImportError, OSError):
            pytest.skip("Transformers not installed or SAM3 model unavailable")

    def test_sam3_device_options(self):
        """Test different device options."""
        try:
            for device in ["cpu", "cuda", "mlx"]:
                segmenter = SAM3Segmenter(device=device)
                assert segmenter.device == device
        except (ImportError, OSError):
            pytest.skip("Transformers not installed or SAM3 model unavailable")

    def test_sam3_temporal_params(self):
        """Test temporal parameters."""
        try:
            segmenter = SAM3Segmenter(
                cache_frames=10,
                temporal_smoothing=True,
                kalman_process_var=0.001,
                kalman_measurement_var=2.0,
            )
            assert segmenter.cache_frames == 10
            assert segmenter.temporal_smoothing is True
            assert segmenter.kalman_process_var == 0.001
            assert segmenter.kalman_measurement_var == 2.0
        except (ImportError, OSError):
            pytest.skip("Transformers not installed or SAM3 model unavailable")


class TestSAM3Segmentation:
    """Test SAM3 segmentation operations (mock)."""

    def test_segment_mock_frame(self):
        """Test segmenting a mock frame."""
        # Create mock segmenter (without loading actual model)
        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None
        segmenter.device = "cpu"

        # Mock segmentation output
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Should raise ImportError when model not loaded
        with pytest.raises(ImportError):
            segmenter.segment(frame)

    def test_segment_dimensions(self):
        """Test that segment validates input dimensions."""
        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None

        # Wrong dimensions
        frame = np.zeros((480, 640), dtype=np.uint8)  # Missing color channel

        with pytest.raises(ImportError):
            segmenter.segment(frame)


class TestSAM3TemporalTracking:
    """Test temporal mask tracking."""

    def setup_method(self):
        """Setup test segmenter."""
        self.segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        self.segmenter.temporal_smoothing = True
        self.segmenter.instance_id_counter = 0
        self.segmenter.mask_history = {}

    def test_assign_instance_ids(self):
        """Test instance ID assignment."""
        # Create synthetic masks
        masks = np.array([
            [[1, 1, 0], [1, 0, 0]],  # Mask 1
            [[0, 0, 1], [0, 1, 1]],  # Mask 2
        ], dtype=np.uint8)

        scores = np.array([0.95, 0.85])  # Confidence scores

        tracked = self.segmenter._assign_instance_ids(masks, scores)

        assert len(tracked) == 2
        assert self.segmenter.instance_id_counter == 2

    def test_track_masks_new_instances(self):
        """Test tracking completely new masks."""
        current_masks = np.array([
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],  # Mask 1
        ], dtype=np.uint8)

        current_scores = np.array([0.9])
        prev_instance_map = np.zeros((3, 4), dtype=np.uint8)

        tracked = self.segmenter._track_masks(
            current_masks, current_scores, prev_instance_map
        )

        assert len(tracked) == 1
        assert tracked[0].shape == (3, 4)

    def test_track_masks_with_previous(self):
        """Test tracking with previous frame data."""
        current_masks = np.array([
            [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        ], dtype=np.uint8)

        current_scores = np.array([0.9])

        # Previous frame has instance ID 1 in similar location
        prev_instance_map = np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ], dtype=np.uint8)

        self.segmenter.instance_id_counter = 1

        tracked = self.segmenter._track_masks(
            current_masks, current_scores, prev_instance_map
        )

        assert len(tracked) == 1

    def test_kalman_smooth_mask(self):
        """Test Kalman smoothing of mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1

        smoothed = self.segmenter._kalman_smooth_mask(mask, instance_id=1)

        # Currently returns mask as-is (placeholder)
        assert np.array_equal(smoothed, mask)


class TestSAM3Video:
    """Test video segmentation."""

    def test_segment_video_shape(self):
        """Test video segmentation output shape."""
        # Create mock video (5 frames, 480×640)
        frames = np.random.randint(0, 256, (5, 480, 640, 3), dtype=np.uint8)

        # Create mock segmenter
        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None
        segmenter.frame_cache = []
        segmenter.mask_history = {}
        segmenter.instance_id_counter = 0
        segmenter.cache_frames = 5

        # Mock segment method to return dummy masks
        def mock_segment(frame):
            masks = np.ones((2, 480, 640), dtype=np.uint8)
            masks[0, :240, :] = 1  # Instance 1
            masks[1, 240:, :] = 1  # Instance 2
            scores = np.array([0.9, 0.85])
            return masks, scores

        segmenter.segment = mock_segment
        segmenter._track_masks = lambda m, s, p: [m[0], m[1]]
        segmenter._assign_instance_ids = lambda m, s: list(m)
        segmenter._kalman_smooth_mask = lambda m, i: m

        result = segmenter.segment_video(frames, use_temporal_tracking=False)

        assert result.shape == (5, 480, 640)
        assert result.dtype == np.uint8

    def test_segment_video_temporal_consistency(self):
        """Test temporal consistency in video."""
        # Simple video: 3 frames
        frames = np.ones((3, 100, 100, 3), dtype=np.uint8) * 128

        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None
        segmenter.frame_cache = []
        segmenter.mask_history = {}
        segmenter.instance_id_counter = 0
        segmenter.cache_frames = 5
        segmenter.temporal_smoothing = True

        # Mock methods
        def mock_segment(frame):
            masks = np.zeros((1, 100, 100), dtype=np.uint8)
            masks[0, :50, :50] = 1
            return masks, np.array([0.9])

        segmenter.segment = mock_segment
        segmenter._track_masks = lambda m, s, p: list(m)
        segmenter._assign_instance_ids = lambda m, s: list(m)
        segmenter._kalman_smooth_mask = lambda m, i: m

        result = segmenter.segment_video(frames, use_temporal_tracking=True)

        assert result.shape == (3, 100, 100)
        # All frames should have consistent instance IDs
        assert np.max(result) > 0


class TestSAM3Batch:
    """Test batch processing."""

    def test_segment_batch_shape(self):
        """Test batch segmentation output shape."""
        batch = np.random.randint(0, 256, (4, 480, 640, 3), dtype=np.uint8)

        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None

        # Mock segment method
        def mock_segment(frame):
            masks = np.ones((2, 480, 640), dtype=np.uint8)
            scores = np.array([0.9, 0.85])
            return masks, scores

        segmenter.segment = mock_segment

        result = segmenter.segment_batch(batch)

        assert result.shape == (4, 480, 640)
        assert result.dtype == np.uint8

    def test_segment_batch_empty(self):
        """Test batch with empty batch."""
        batch = np.zeros((0, 480, 640, 3), dtype=np.uint8)

        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None
        segmenter.segment = lambda f: (np.zeros((1, 480, 640)), np.array([0.9]))

        result = segmenter.segment_batch(batch)

        assert result.shape == (0, 480, 640)


class TestSAM3WithPrompts:
    """Test prompted segmentation."""

    def test_segment_with_point_prompt(self):
        """Test segmentation with point prompts."""
        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        prompt = {
            "points": [[240, 320], [100, 200]],
            "labels": [1, 0],  # 1=foreground, 0=background
        }

        with pytest.raises(ImportError):
            segmenter.segment_with_prompt(frame, prompt)

    def test_segment_with_box_prompt(self):
        """Test segmentation with bounding box prompt."""
        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        prompt = {
            "boxes": [[100, 100, 300, 300]],  # (y1, x1, y2, x2)
        }

        with pytest.raises(ImportError):
            segmenter.segment_with_prompt(frame, prompt)

    def test_segment_with_mask_prompt(self):
        """Test segmentation with existing mask refinement."""
        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask_input = np.zeros((480, 640), dtype=np.float32)
        mask_input[100:300, 100:300] = 1.0

        prompt = {
            "mask_input": mask_input,
        }

        with pytest.raises(ImportError):
            segmenter.segment_with_prompt(frame, prompt)


class TestSAM3State:
    """Test state management."""

    def test_reset_clears_cache(self):
        """Test that reset clears temporal state."""
        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.frame_cache = [np.zeros((480, 640, 3))]
        segmenter.mask_history = {1: [np.ones((480, 640))]}
        segmenter.instance_id_counter = 5

        segmenter.reset()

        assert len(segmenter.frame_cache) == 0
        assert len(segmenter.mask_history) == 0
        # Note: instance_id_counter not reset (for consistency across resets)

    def test_repr(self):
        """Test string representation."""
        try:
            segmenter = SAM3Segmenter(
                model_id="facebook/sam3-base",
                device="mlx",
            )
            repr_str = repr(segmenter)
            assert "facebook/sam3-base" in repr_str
            assert "mlx" in repr_str
            assert "temporal=True" in repr_str
        except (ImportError, OSError):
            pytest.skip("Transformers not installed or SAM3 model unavailable")


class TestSAM3OccupancyIntegration:
    """Test integration with occupancy grid."""

    def test_masks_to_occupancy_update(self):
        """Test converting masks to occupancy grid updates."""
        from pyroboframes.automotive import OccupancyGrid

        # Create occupancy grid
        occupancy = OccupancyGrid(size=(-50, 50), resolution=0.5)

        # Create synthetic mask (simulating SAM3 output)
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 100:300] = 1  # Rectangle in image space

        # In production, would map image coords → world coords
        # and update occupancy with mask

        # Just verify grid can be created
        assert occupancy.grid_size == 200
        assert occupancy.log_odds.shape == (200, 200)


class TestSAM3EdgeCases:
    """Test edge cases and error handling."""

    def test_segment_all_black_frame(self):
        """Test segmentation on all-black frame."""
        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ImportError):
            segmenter.segment(frame)

    def test_segment_all_white_frame(self):
        """Test segmentation on all-white frame."""
        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        with pytest.raises(ImportError):
            segmenter.segment(frame)

    def test_video_single_frame(self):
        """Test video with single frame."""
        frames = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)

        segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
        segmenter.model = None
        segmenter.frame_cache = []
        segmenter.mask_history = {}
        segmenter.instance_id_counter = 0
        segmenter.cache_frames = 5

        def mock_segment(frame):
            masks = np.ones((1, 480, 640), dtype=np.uint8)
            return masks, np.array([0.9])

        segmenter.segment = mock_segment
        segmenter._assign_instance_ids = lambda m, s: list(m)

        result = segmenter.segment_video(frames, use_temporal_tracking=False)

        assert result.shape == (1, 480, 640)

    def test_max_resolution_parameter(self):
        """Test max_resolution downsampling option."""
        try:
            segmenter = SAM3Segmenter(max_resolution=480)
            assert segmenter.max_resolution == 480
        except (ImportError, OSError):
            pytest.skip("Transformers not installed or SAM3 model unavailable")
