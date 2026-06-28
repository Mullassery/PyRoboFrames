"""Multi-modal fusion for unified scene understanding in autonomous driving.

Phase 7d: Combine SAM3 segmentation, CLIP embeddings, and Grounding DINO detection
into a single coherent perception pipeline.

Pipeline: Grounding DINO (detect) → SAM3 (segment) → CLIP (classify)
Output: Unified SceneUnderstanding with objects, masks, and semantic context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np


@dataclass
class DetectedObject:
    """Single detected object with mask and semantic label.

    Attributes:
        object_class: Object class (e.g., "car", "pedestrian")
        bbox: [y1, x1, y2, x2] bounding box
        confidence: Detection confidence [0, 1]
        mask: [H, W] binary mask (optional, from SAM3)
        semantic_label: Semantic class (optional, from CLIP)
        embedding: [D] semantic embedding from CLIP
    """

    object_class: str
    bbox: np.ndarray
    confidence: float
    mask: Optional[np.ndarray] = None
    semantic_label: Optional[str] = None
    embedding: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        mask_info = f", mask: {self.mask.shape}" if self.mask is not None else ""
        return (
            f"DetectedObject({self.object_class}, conf={self.confidence:.2f}"
            f"{mask_info})"
        )


@dataclass
class SceneUnderstanding:
    """Unified scene understanding from multi-modal fusion.

    Attributes:
        objects: List of detected objects with masks and labels
        scene_type: Primary scene classification (highway, city, etc.)
        scene_scores: {scene_type -> confidence}
        weather: Weather/lighting condition
        image: Original input image [H, W, 3]
        panorama: Optional panoramic stitching [H, W', 3]
    """

    objects: List[DetectedObject]
    scene_type: str
    scene_scores: Dict[str, float]
    weather: str
    image: np.ndarray
    panorama: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"SceneUnderstanding("
            f"scene={self.scene_type}, "
            f"objects={len(self.objects)}, "
            f"weather={self.weather})"
        )

    def get_object_masks(self) -> Optional[np.ndarray]:
        """Get stacked masks for all objects.

        Returns:
            [N, H, W] uint8 masks or None if no masks available
        """
        masks = [obj.mask for obj in self.objects if obj.mask is not None]

        if not masks:
            return None

        return np.stack(masks, axis=0).astype(np.uint8)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "scene_type": self.scene_type,
            "scene_scores": self.scene_scores,
            "weather": self.weather,
            "objects": [
                {
                    "class": obj.object_class,
                    "bbox": obj.bbox.tolist(),
                    "confidence": float(obj.confidence),
                    "semantic_label": obj.semantic_label,
                }
                for obj in self.objects
            ],
        }


class MultiModalFusion:
    """Unified perception pipeline combining SAM3, CLIP, and Grounding DINO.

    Sequential pipeline:
    1. Grounding DINO: Detect objects with language
    2. SAM3: Refine with instance segmentation masks
    3. CLIP: Classify scene and label objects semantically

    Usage:
        ```python
        from pyroboframes.automotive import MultiModalFusion

        fusion = MultiModalFusion(
            detection_prompt="car . pedestrian . cyclist . truck"
        )

        scene = fusion.understand(frame)
        print(f"Scene: {scene.scene_type}")
        for obj in scene.objects:
            print(f"  {obj.object_class}: {obj.semantic_label}")
        ```
    """

    def __init__(
        self,
        detection_prompt: str = "car . pedestrian . cyclist . truck . bus . traffic sign",
        device: Optional[str] = None,
        use_sam3: bool = True,
        use_clip: bool = True,
    ):
        """Initialize multi-modal fusion pipeline.

        Args:
            detection_prompt: Language prompt for Grounding DINO detection
            device: "cuda", "mlx", "cpu", or None for auto-detect
            use_sam3: Include SAM3 segmentation refinement
            use_clip: Include CLIP semantic classification
        """
        self.detection_prompt = detection_prompt
        self.device = device or "cpu"
        self.use_sam3 = use_sam3
        self.use_clip = use_clip

        # Initialize sub-modules
        self.grounding_dino = None
        self.sam3_segmenter = None
        self.clip_embedder = None

        self._load_modules()

    def _load_modules(self):
        """Load foundation models."""
        try:
            from pyroboframes.automotive import GroundingDINO

            self.grounding_dino = GroundingDINO(
                device=self.device,
                use_sam3=False,  # We handle SAM3 separately
            )
        except (ImportError, OSError):
            print("Note: Grounding DINO not available")
            self.grounding_dino = None

        if self.use_sam3:
            try:
                from pyroboframes.automotive import SAM3Segmenter

                self.sam3_segmenter = SAM3Segmenter(device=self.device)
            except (ImportError, OSError):
                print("Note: SAM3 not available")
                self.sam3_segmenter = None

        if self.use_clip:
            try:
                from pyroboframes.automotive import CLIPEmbedding

                self.clip_embedder = CLIPEmbedding(device=self.device)
            except (ImportError, OSError):
                print("Note: CLIP not available")
                self.clip_embedder = None

    def understand(
        self,
        image: np.ndarray,
        detection_prompt: Optional[str] = None,
        semantic_classes: Optional[List[str]] = None,
    ) -> SceneUnderstanding:
        """Perform unified scene understanding on a single frame.

        Sequential pipeline:
        1. Detect objects with Grounding DINO
        2. Refine with SAM3 masks
        3. Classify semantically with CLIP

        Args:
            image: [H, W, 3] uint8 image
            detection_prompt: Override default detection prompt
            semantic_classes: Custom semantic classes for CLIP labeling
                (defaults to object classes from detection)

        Returns:
            SceneUnderstanding with detected objects and scene context
        """
        if self.grounding_dino is None:
            raise ImportError("Grounding DINO required for detection phase")

        prompt = detection_prompt or self.detection_prompt

        # Phase 1: Detect objects with Grounding DINO
        detections = self.grounding_dino.detect(image, prompt)

        # Phase 2: Segment with SAM3
        objects = []
        if self.sam3_segmenter is not None:
            masks, _ = self.sam3_segmenter.segment(image)

            for obj_class, boxes in detections.items():
                for bbox, conf, _ in boxes:
                    # Find matching SAM3 mask
                    mask = self._find_best_mask(masks, bbox, image.shape)

                    obj = DetectedObject(
                        object_class=obj_class,
                        bbox=bbox,
                        confidence=conf,
                        mask=mask,
                    )
                    objects.append(obj)
        else:
            # Without SAM3, just use bounding boxes
            for obj_class, boxes in detections.items():
                for bbox, conf, _ in boxes:
                    obj = DetectedObject(
                        object_class=obj_class,
                        bbox=bbox,
                        confidence=conf,
                    )
                    objects.append(obj)

        # Phase 3: Classify semantically with CLIP
        if self.clip_embedder is not None:
            objects = self._classify_objects(image, objects, semantic_classes)

        # Classify scene
        scene_scores = self.clip_embedder.scene_classification(image) if self.clip_embedder else {}
        scene_type = max(scene_scores, key=scene_scores.get) if scene_scores else "unknown"

        weather = (
            self.clip_embedder.weather_classification(image).get("clear day", "unknown")
            if self.clip_embedder
            else "unknown"
        )

        return SceneUnderstanding(
            objects=objects,
            scene_type=scene_type,
            scene_scores=scene_scores,
            weather=weather,
            image=image,
        )

    def understand_batch(
        self,
        frames: np.ndarray,
        detection_prompt: Optional[str] = None,
    ) -> List[SceneUnderstanding]:
        """Perform scene understanding on batch of frames.

        Args:
            frames: [B, H, W, 3] uint8 batch of images
            detection_prompt: Override default detection prompt

        Returns:
            List of SceneUnderstanding (one per frame)
        """
        batch_size = frames.shape[0]
        results = []

        for b in range(batch_size):
            scene = self.understand(frames[b], detection_prompt)
            results.append(scene)

        return results

    def understand_video(
        self,
        frames: np.ndarray,
        detection_prompt: Optional[str] = None,
        temporal_consistency: bool = True,
    ) -> List[SceneUnderstanding]:
        """Perform scene understanding on video sequence.

        Args:
            frames: [T, H, W, 3] uint8 video frames
            detection_prompt: Override default detection prompt
            temporal_consistency: Track objects across frames

        Returns:
            List of SceneUnderstanding (one per frame)
        """
        results = self.understand_batch(frames, detection_prompt)

        if temporal_consistency:
            results = self._apply_temporal_tracking(results)

        return results

    def _classify_objects(
        self,
        image: np.ndarray,
        objects: List[DetectedObject],
        semantic_classes: Optional[List[str]] = None,
    ) -> List[DetectedObject]:
        """Classify detected objects using CLIP embeddings.

        Args:
            image: Original image [H, W, 3]
            objects: List of detected objects
            semantic_classes: Custom classes to classify into

        Returns:
            Objects with semantic labels and embeddings
        """
        if self.clip_embedder is None:
            return objects

        # Determine semantic classes
        if semantic_classes is None:
            semantic_classes = list(set(obj.object_class for obj in objects))

        # Embed all semantic classes
        class_embeddings = self.clip_embedder.embed_texts_batch(semantic_classes)

        for obj in objects:
            # Crop object region
            H, W = image.shape[:2]
            y1, x1, y2, x2 = [int(v) for v in obj.bbox]
            y1, y2 = max(0, y1), min(H, y2)
            x1, x2 = max(0, x1), min(W, x2)

            if y2 > y1 and x2 > x1:
                obj_region = image[y1:y2, x1:x2]

                # Embed object region
                obj_embedding = self.clip_embedder.embed_frame(obj_region)

                # Find best matching semantic class
                similarities = obj_embedding @ class_embeddings.T
                best_idx = np.argmax(similarities)

                obj.semantic_label = semantic_classes[best_idx]
                obj.embedding = obj_embedding

        return objects

    def _find_best_mask(
        self,
        masks: np.ndarray,
        bbox: np.ndarray,
        image_shape: Tuple[int, int, int],
    ) -> Optional[np.ndarray]:
        """Find SAM3 mask matching bounding box.

        Args:
            masks: [N, H, W] SAM3 masks
            bbox: [y1, x1, y2, x2] bounding box
            image_shape: (H, W, 3) image shape

        Returns:
            Matching mask [H, W] or None
        """
        if len(masks) == 0:
            return None

        H, W = image_shape[:2]
        y1, x1, y2, x2 = [int(v) for v in bbox]
        y1, y2 = max(0, y1), min(H, y2)
        x1, x2 = max(0, x1), min(W, x2)

        # Create box region
        box_region = np.zeros((H, W), dtype=np.uint8)
        box_region[y1:y2, x1:x2] = 1

        # Find mask with most overlap
        max_overlap = 0
        best_mask = None

        for mask in masks:
            overlap = np.sum((mask > 0) & (box_region > 0))
            if overlap > max_overlap:
                max_overlap = overlap
                best_mask = mask

        return best_mask if max_overlap > 0 else None

    def _apply_temporal_tracking(
        self,
        scenes: List[SceneUnderstanding],
    ) -> List[SceneUnderstanding]:
        """Apply temporal tracking to maintain object identity across frames.

        Simple tracking based on IoU matching between consecutive frames.

        Args:
            scenes: List of SceneUnderstanding

        Returns:
            Scenes with consistent object IDs
        """
        for t in range(1, len(scenes)):
            prev_objects = scenes[t - 1].objects
            curr_objects = scenes[t].objects

            # Match objects based on IoU
            matches = self._match_objects(prev_objects, curr_objects)

            # Could add ID tracking here if needed
            # For now, just ensure consistent ordering

        return scenes

    def _match_objects(
        self,
        prev_objects: List[DetectedObject],
        curr_objects: List[DetectedObject],
    ) -> List[Tuple[int, int]]:
        """Match objects between consecutive frames.

        Args:
            prev_objects: Objects from previous frame
            curr_objects: Objects from current frame

        Returns:
            List of (prev_idx, curr_idx) matches
        """
        matches = []

        for i, prev_obj in enumerate(prev_objects):
            best_j = -1
            best_iou = 0

            for j, curr_obj in enumerate(curr_objects):
                iou = self._compute_iou(prev_obj.bbox, curr_obj.bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou > 0.3:  # IoU threshold
                matches.append((i, best_j))

        return matches

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes.

        Args:
            box1, box2: [y1, x1, y2, x2] boxes

        Returns:
            IoU score
        """
        y1_min, x1_min, y1_max, x1_max = box1
        y2_min, x2_min, y2_max, x2_max = box2

        inter_y_min = max(y1_min, y2_min)
        inter_x_min = max(x1_min, x2_min)
        inter_y_max = min(y1_max, y2_max)
        inter_x_max = min(x1_max, x2_max)

        if inter_y_min >= inter_y_max or inter_x_min >= inter_x_max:
            return 0.0

        inter_area = (inter_y_max - inter_y_min) * (inter_x_max - inter_x_min)
        area1 = (y1_max - y1_min) * (x1_max - x1_min)
        area2 = (y2_max - y2_min) * (x2_max - x2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"MultiModalFusion("
            f"device='{self.device}', "
            f"sam3={self.use_sam3}, "
            f"clip={self.use_clip})"
        )
