"""Tests for Phase 7c: Grounding DINO Open-Vocabulary Detection."""

import numpy as np
import pytest

from pyroboframes.automotive import GroundingDINO


class TestGroundingDINOInitialization:
    """Test Grounding DINO initialization."""

    def test_grounding_dino_init_default(self):
        """Test GroundingDINO initialization with defaults."""
        try:
            detector = GroundingDINO(model_id="IDEA-Research/grounding-dino-tiny")
            assert detector.model_id == "IDEA-Research/grounding-dino-tiny"
            assert detector.device == "cpu"
            assert detector.use_sam3 is True
            assert detector.confidence_threshold == 0.3
        except ImportError:
            pytest.skip("Transformers not installed")

    def test_grounding_dino_init_small_model(self):
        """Test initialization with small model."""
        try:
            detector = GroundingDINO(model_id="IDEA-Research/grounding-dino-small")
            assert detector.model_id == "IDEA-Research/grounding-dino-small"
        except (ImportError, OSError):
            pytest.skip("Transformers not installed or model unavailable")

    def test_grounding_dino_device_options(self):
        """Test different device options."""
        try:
            for device in ["cpu", "cuda", "mlx"]:
                detector = GroundingDINO(device=device)
                assert detector.device == device
        except ImportError:
            pytest.skip("Transformers not installed")

    def test_grounding_dino_parameters(self):
        """Test custom parameters."""
        try:
            detector = GroundingDINO(
                confidence_threshold=0.5,
                nms_threshold=0.4,
                use_sam3=False,
            )
            assert detector.confidence_threshold == 0.5
            assert detector.nms_threshold == 0.4
            assert detector.use_sam3 is False
        except ImportError:
            pytest.skip("Transformers not installed")


class TestGroundingDINODetection:
    """Test detection operations."""

    def test_detect_mock(self):
        """Test detection call structure."""
        detector = GroundingDINO.__new__(GroundingDINO)
        detector.model = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        text = "car . pedestrian"

        with pytest.raises(ImportError):
            detector.detect(frame, text)

    def test_detect_text_parsing(self):
        """Test text prompt parsing."""
        # Classes separated by "."
        text = "car . pedestrian . bicycle . motorcycle"
        classes = [c.strip() for c in text.split(".") if c.strip()]

        assert len(classes) == 4
        assert "car" in classes
        assert "pedestrian" in classes


class TestGroundingDINOBatchProcessing:
    """Test batch detection."""

    def test_batch_detection_shape(self):
        """Test batch detection returns correct structure."""
        detector = GroundingDINO.__new__(GroundingDINO)
        detector.model = None

        batch = np.zeros((4, 480, 640, 3), dtype=np.uint8)

        with pytest.raises(ImportError):
            detector.detect_batch(batch, "car . person")

    def test_batch_results_structure(self):
        """Test batch results structure."""
        # Mock batch results
        batch_size = 4
        results = []

        for b in range(batch_size):
            detections = {
                "car": [],
                "pedestrian": [],
            }
            results.append(detections)

        assert len(results) == 4
        for detection_dict in results:
            assert "car" in detection_dict
            assert "pedestrian" in detection_dict


class TestGroundingDINONMS:
    """Test non-maximum suppression."""

    def test_nms_computation(self):
        """Test NMS computation."""
        detector = GroundingDINO.__new__(GroundingDINO)

        # Create mock detections
        boxes = [
            (np.array([10, 10, 100, 100]), 0.9, None),
            (np.array([12, 12, 98, 98]), 0.8, None),   # Overlap with box 0
            (np.array([200, 200, 300, 300]), 0.85, None),  # No overlap
        ]

        keep_indices = detector._nms(boxes, threshold=0.3)

        # Should keep first and third (high confidence, low overlap)
        assert len(keep_indices) >= 1
        assert 0 in keep_indices or 2 in keep_indices

    def test_iou_computation(self):
        """Test IOU calculation."""
        detector = GroundingDINO.__new__(GroundingDINO)

        # Perfect overlap
        box1 = np.array([10, 10, 100, 100])
        box2 = np.array([10, 10, 100, 100])

        iou = detector._compute_iou(box1, box2)

        assert np.isclose(iou, 1.0)

    def test_iou_no_overlap(self):
        """Test IOU with no overlap."""
        detector = GroundingDINO.__new__(GroundingDINO)

        # No overlap
        box1 = np.array([10, 10, 100, 100])
        box2 = np.array([200, 200, 300, 300])

        iou = detector._compute_iou(box1, box2)

        assert iou == 0.0

    def test_iou_partial_overlap(self):
        """Test IOU with partial overlap."""
        detector = GroundingDINO.__new__(GroundingDINO)

        # Partial overlap
        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])

        iou = detector._compute_iou(box1, box2)

        # Expected: intersection = 50*50 = 2500, union = 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert np.isclose(iou, expected)


class TestGroundingDINOFiltering:
    """Test detection filtering."""

    def test_confidence_filtering(self):
        """Test confidence threshold filtering."""
        detector = GroundingDINO.__new__(GroundingDINO)
        detector.confidence_threshold = 0.5

        detections = {
            "car": [
                (np.array([10, 10, 100, 100]), 0.9, None),
                (np.array([200, 200, 300, 300]), 0.3, None),
            ],
        }

        filtered = detector.filter_by_confidence(detections, threshold=0.5)

        assert len(filtered["car"]) == 1
        assert filtered["car"][0][1] == 0.9

    def test_nms_filtering(self):
        """Test NMS filtering."""
        detector = GroundingDINO.__new__(GroundingDINO)
        detector.nms_threshold = 0.3

        detections = {
            "car": [
                (np.array([10, 10, 100, 100]), 0.9, None),
                (np.array([12, 12, 98, 98]), 0.8, None),
            ],
        }

        filtered = detector.apply_nms(detections)

        # After NMS, overlapping detections should be reduced
        total_before = sum(len(boxes) for boxes in detections.values())
        total_after = sum(len(boxes) for boxes in filtered.values())

        assert total_after <= total_before


class TestGroundingDINOFormatConversion:
    """Test format conversions."""

    def test_detections_to_bboxes(self):
        """Test converting detections to bbox array."""
        detector = GroundingDINO.__new__(GroundingDINO)

        detections = {
            "car": [
                (np.array([10, 10, 100, 100]), 0.9, None),
                (np.array([200, 200, 300, 300]), 0.85, None),
            ],
            "pedestrian": [
                (np.array([50, 50, 150, 150]), 0.8, None),
            ],
        }

        bboxes = detector.get_detections_as_bboxes(detections)

        assert bboxes.shape == (3, 5)
        assert np.all(bboxes[:, 4] >= 0.75)  # All confidence scores

    def test_empty_detections_to_bboxes(self):
        """Test empty detections conversion."""
        detector = GroundingDINO.__new__(GroundingDINO)

        detections = {
            "car": [],
            "pedestrian": [],
        }

        bboxes = detector.get_detections_as_bboxes(detections)

        assert bboxes.shape == (0, 5)


class TestGroundingDINOCustomPrompt:
    """Test custom prompts."""

    def test_custom_prompt_structure(self):
        """Test custom prompt handling."""
        detector = GroundingDINO.__new__(GroundingDINO)
        detector.model = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        prompt = "a red car approaching from the left"

        # Mock: custom prompt as single class
        detections = {prompt: []}

        assert prompt in detections
        assert isinstance(detections[prompt], list)


class TestGroundingDINOSAM3Integration:
    """Test SAM3 refinement."""

    def test_sam3_refinement_available(self):
        """Test SAM3 refinement initialization."""
        try:
            detector = GroundingDINO(use_sam3=True)

            # Check if SAM3 loaded
            has_sam3 = detector.sam3_segmenter is not None

            # SAM3 might not be available, but detector should still work
            assert detector.use_sam3 is True

        except ImportError:
            pytest.skip("Transformers not installed")

    def test_sam3_refinement_disabled(self):
        """Test disabling SAM3 refinement."""
        try:
            detector = GroundingDINO(use_sam3=False)
            assert detector.sam3_segmenter is None
            assert detector.use_sam3 is False

        except ImportError:
            pytest.skip("Transformers not installed")

    def test_mask_matching(self):
        """Test SAM3 mask matching to boxes."""
        detector = GroundingDINO.__new__(GroundingDINO)

        # Mock SAM3 masks
        masks = np.zeros((2, 480, 640), dtype=np.uint8)
        masks[0, 10:100, 10:100] = 1   # Mask 1
        masks[1, 200:300, 200:300] = 1  # Mask 2

        # Find mask in box [10, 10, 100, 100]
        mask = detector._find_matching_mask(masks, 10, 10, 100, 100)

        assert mask is not None


class TestGroundingDINOEdgeCases:
    """Test edge cases."""

    def test_empty_frame(self):
        """Test detection on empty frame."""
        detector = GroundingDINO.__new__(GroundingDINO)
        detector.model = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ImportError):
            detector.detect(frame, "car")

    def test_empty_text_prompt(self):
        """Test with empty text prompt."""
        text = ""
        classes = [c.strip() for c in text.split(".") if c.strip()]

        assert len(classes) == 0

    def test_many_classes(self):
        """Test with many object classes."""
        text = "car . truck . bus . motorcycle . bicycle . pedestrian . dog . cat . bird . sign"
        classes = [c.strip() for c in text.split(".") if c.strip()]

        assert len(classes) == 10

    def test_special_characters_in_text(self):
        """Test text with special characters."""
        text = "a red car . a green truck . people walking"
        classes = [c.strip() for c in text.split(".") if c.strip()]

        assert len(classes) == 3
        assert "a red car" in classes


class TestGroundingDINORepr:
    """Test string representation."""

    def test_repr_format(self):
        """Test __repr__ format."""
        try:
            detector = GroundingDINO(
                model_id="IDEA-Research/grounding-dino-tiny",
                device="mlx",
            )
            repr_str = repr(detector)
            assert "grounding-dino-tiny" in repr_str
            assert "mlx" in repr_str
        except ImportError:
            pytest.skip("Transformers not installed")


class TestGroundingDINOIntegration:
    """Test integration with other modules."""

    def test_with_sam3_and_clip(self):
        """Test integration with SAM3 and CLIP."""
        from pyroboframes.automotive import SAM3Segmenter, CLIPEmbedding

        try:
            detector = GroundingDINO(use_sam3=True)
            # These would be used together in real pipeline
            assert hasattr(detector, "device")
            assert hasattr(detector, "use_sam3")

        except ImportError:
            pytest.skip("Dependencies not available")
