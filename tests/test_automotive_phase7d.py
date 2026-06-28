"""Tests for Phase 7d: Multi-modal fusion (SAM3 + CLIP + Grounding DINO).

Tests SceneUnderstanding, DetectedObject, and MultiModalFusion classes.
"""

import numpy as np
import pytest

from pyroboframes.automotive import (
    MultiModalFusion,
    SceneUnderstanding,
    DetectedObject,
)


class TestDetectedObject:
    """Tests for DetectedObject dataclass."""

    def test_basic_creation(self):
        """Test creating a detected object."""
        bbox = np.array([100, 50, 200, 150])
        obj = DetectedObject(
            object_class="car",
            bbox=bbox,
            confidence=0.95,
        )

        assert obj.object_class == "car"
        assert np.array_equal(obj.bbox, bbox)
        assert obj.confidence == 0.95
        assert obj.mask is None
        assert obj.semantic_label is None

    def test_with_mask_and_label(self):
        """Test detected object with mask and semantic label."""
        mask = np.ones((480, 640), dtype=np.uint8)
        embedding = np.random.randn(512).astype(np.float32)

        obj = DetectedObject(
            object_class="pedestrian",
            bbox=np.array([50, 100, 150, 200]),
            confidence=0.87,
            mask=mask,
            semantic_label="person walking",
            embedding=embedding,
        )

        assert obj.semantic_label == "person walking"
        assert obj.mask.shape == (480, 640)
        assert obj.embedding.shape == (512,)

    def test_repr(self):
        """Test string representation."""
        obj = DetectedObject(
            object_class="car",
            bbox=np.array([0, 0, 100, 100]),
            confidence=0.9,
        )
        repr_str = repr(obj)
        assert "car" in repr_str
        assert "0.90" in repr_str


class TestSceneUnderstanding:
    """Tests for SceneUnderstanding dataclass."""

    def test_basic_creation(self):
        """Test creating scene understanding."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        objects = [
            DetectedObject(
                object_class="car",
                bbox=np.array([50, 50, 150, 150]),
                confidence=0.9,
            ),
            DetectedObject(
                object_class="pedestrian",
                bbox=np.array([200, 200, 300, 300]),
                confidence=0.85,
            ),
        ]

        scene = SceneUnderstanding(
            objects=objects,
            scene_type="highway",
            scene_scores={"highway": 0.95, "city": 0.03},
            weather="clear",
            image=image,
        )

        assert len(scene.objects) == 2
        assert scene.scene_type == "highway"
        assert scene.weather == "clear"

    def test_get_object_masks_no_masks(self):
        """Test getting masks when objects have no masks."""
        scene = SceneUnderstanding(
            objects=[
                DetectedObject(
                    object_class="car",
                    bbox=np.array([0, 0, 100, 100]),
                    confidence=0.9,
                )
            ],
            scene_type="city",
            scene_scores={},
            weather="clear",
            image=np.zeros((480, 640, 3), dtype=np.uint8),
        )

        masks = scene.get_object_masks()
        assert masks is None

    def test_get_object_masks_with_masks(self):
        """Test getting stacked masks."""
        mask1 = np.ones((480, 640), dtype=np.uint8)
        mask2 = np.zeros((480, 640), dtype=np.uint8)

        scene = SceneUnderstanding(
            objects=[
                DetectedObject(
                    object_class="car",
                    bbox=np.array([0, 0, 100, 100]),
                    confidence=0.9,
                    mask=mask1,
                ),
                DetectedObject(
                    object_class="pedestrian",
                    bbox=np.array([200, 200, 300, 300]),
                    confidence=0.85,
                    mask=mask2,
                ),
            ],
            scene_type="city",
            scene_scores={},
            weather="clear",
            image=np.zeros((480, 640, 3), dtype=np.uint8),
        )

        masks = scene.get_object_masks()
        assert masks is not None
        assert masks.shape == (2, 480, 640)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        scene = SceneUnderstanding(
            objects=[
                DetectedObject(
                    object_class="car",
                    bbox=np.array([50, 50, 150, 150]),
                    confidence=0.9,
                    semantic_label="sedan",
                )
            ],
            scene_type="highway",
            scene_scores={"highway": 0.95},
            weather="clear",
            image=np.zeros((480, 640, 3), dtype=np.uint8),
        )

        scene_dict = scene.to_dict()
        assert scene_dict["scene_type"] == "highway"
        assert scene_dict["weather"] == "clear"
        assert len(scene_dict["objects"]) == 1
        assert scene_dict["objects"][0]["class"] == "car"
        assert scene_dict["objects"][0]["semantic_label"] == "sedan"


class TestMultiModalFusion:
    """Tests for MultiModalFusion pipeline."""

    def test_initialization(self):
        """Test initializing fusion pipeline."""
        fusion = MultiModalFusion(
            detection_prompt="car . pedestrian",
            device="cpu",
        )

        assert fusion.device == "cpu"
        assert fusion.detection_prompt == "car . pedestrian"

    def test_initialization_with_modules(self):
        """Test that modules attempt to load."""
        # Models may not be available, so we just check initialization
        fusion = MultiModalFusion(
            use_sam3=True,
            use_clip=True,
        )

        assert fusion.use_sam3
        assert fusion.use_clip

    def test_understand_without_grounding_dino(self):
        """Test that understand fails gracefully without Grounding DINO."""
        fusion = MultiModalFusion()
        fusion.grounding_dino = None

        image = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ImportError):
            fusion.understand(image)

    def test_find_best_mask(self):
        """Test finding mask matching bbox."""
        fusion = MultiModalFusion()

        # Create mock masks
        mask1 = np.zeros((480, 640), dtype=bool)
        mask1[100:200, 100:200] = True

        mask2 = np.zeros((480, 640), dtype=bool)
        mask2[300:400, 300:400] = True

        masks = np.stack([mask1, mask2])

        # Find mask for bbox in mask1 region
        bbox = np.array([100, 100, 200, 200])
        best_mask = fusion._find_best_mask(masks, bbox, (480, 640, 3))

        assert best_mask is not None
        assert np.array_equal(best_mask, mask1)

    def test_find_best_mask_no_overlap(self):
        """Test when no mask overlaps bbox."""
        fusion = MultiModalFusion()

        mask = np.zeros((480, 640), dtype=bool)
        mask[300:400, 300:400] = True
        masks = np.array([mask])

        # BBox far from mask
        bbox = np.array([0, 0, 50, 50])
        best_mask = fusion._find_best_mask(masks, bbox, (480, 640, 3))

        assert best_mask is None

    def test_compute_iou(self):
        """Test IoU computation."""
        fusion = MultiModalFusion()

        box1 = np.array([0, 0, 100, 100])
        box2 = np.array([50, 50, 150, 150])

        iou = fusion._compute_iou(box1, box2)

        # Expected: intersection area = 50*50 = 2500
        # Union area = 100*100 + 100*100 - 2500 = 17500
        # IoU = 2500/17500 = 1/7 ≈ 0.143
        assert iou == pytest.approx(1.0 / 7.0, abs=0.01)

    def test_compute_iou_no_overlap(self):
        """Test IoU when boxes don't overlap."""
        fusion = MultiModalFusion()

        box1 = np.array([0, 0, 50, 50])
        box2 = np.array([100, 100, 150, 150])

        iou = fusion._compute_iou(box1, box2)

        assert iou == 0.0

    def test_compute_iou_identical(self):
        """Test IoU of identical boxes."""
        fusion = MultiModalFusion()

        box = np.array([0, 0, 100, 100])
        iou = fusion._compute_iou(box, box)

        assert iou == pytest.approx(1.0)

    def test_match_objects(self):
        """Test object matching between frames."""
        fusion = MultiModalFusion()

        prev_objects = [
            DetectedObject(
                object_class="car",
                bbox=np.array([100, 100, 200, 200]),
                confidence=0.9,
            ),
            DetectedObject(
                object_class="pedestrian",
                bbox=np.array([300, 300, 400, 400]),
                confidence=0.85,
            ),
        ]

        # Similar positions in next frame
        curr_objects = [
            DetectedObject(
                object_class="car",
                bbox=np.array([105, 105, 205, 205]),
                confidence=0.91,
            ),
            DetectedObject(
                object_class="pedestrian",
                bbox=np.array([305, 305, 405, 405]),
                confidence=0.84,
            ),
        ]

        matches = fusion._match_objects(prev_objects, curr_objects)

        assert len(matches) == 2
        assert (0, 0) in matches
        assert (1, 1) in matches

    def test_repr(self):
        """Test string representation."""
        fusion = MultiModalFusion(device="cuda")
        repr_str = repr(fusion)

        assert "cuda" in repr_str
        assert "MultiModalFusion" in repr_str


class TestMultiModalFusionIntegration:
    """Integration tests for complete pipeline."""

    @pytest.mark.skip(reason="Requires foundation models")
    def test_understand_single_frame(self):
        """Test understanding a single frame."""
        fusion = MultiModalFusion()

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        scene = fusion.understand(frame)

        assert isinstance(scene, SceneUnderstanding)
        assert scene.image.shape == (480, 640, 3)

    @pytest.mark.skip(reason="Requires foundation models")
    def test_understand_batch(self):
        """Test understanding batch of frames."""
        fusion = MultiModalFusion()

        frames = np.random.randint(0, 256, (5, 480, 640, 3), dtype=np.uint8)
        scenes = fusion.understand_batch(frames)

        assert len(scenes) == 5
        assert all(isinstance(s, SceneUnderstanding) for s in scenes)

    @pytest.mark.skip(reason="Requires foundation models")
    def test_understand_video(self):
        """Test understanding video sequence."""
        fusion = MultiModalFusion()

        frames = np.random.randint(0, 256, (10, 480, 640, 3), dtype=np.uint8)
        scenes = fusion.understand_video(frames, temporal_consistency=True)

        assert len(scenes) == 10


# Test skipped by default (requires models)
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
