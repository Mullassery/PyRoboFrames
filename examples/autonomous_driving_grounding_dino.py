"""
Phase 7c: Grounding DINO Open-Vocabulary Detection.

Demonstrates:
- Language-grounded object detection
- Open-vocabulary detection (arbitrary descriptions)
- SAM3 mask refinement for precise boundaries
- Text-based object search
- Integration with SAM3 + CLIP pipeline
"""

import numpy as np

from pyroboframes.automotive import (
    CylindricalStitcher,
    GroundingDINO,
    SAM3Segmenter,
    CLIPEmbedding,
    get_waymo_layout,
)


def phase7c_basic_detection():
    """Phase 7c: Basic Grounding DINO detection."""
    print("=" * 70)
    print("Phase 7c: Open-Vocabulary Object Detection with Grounding DINO")
    print("=" * 70)

    try:
        detector = GroundingDINO(
            model_id="IDEA-Research/grounding-dino-tiny",
            device="mlx",
            use_sam3=True,  # Refine with SAM3 masks
        )

        print(f"✓ Grounding DINO initialized")
        print(f"  - Model: IDEA-Research/grounding-dino-tiny")
        print(f"  - Device: MLX (Apple Silicon)")
        print(f"  - SAM3 refinement: enabled")
        print()

        # Single frame detection
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Open-vocabulary detection
        text = "car . pedestrian . traffic sign . cyclist"
        detections = detector.detect(frame, text)

        print(f"Detected objects:")
        for obj_class, boxes in detections.items():
            print(f"  {obj_class:15s}: {len(boxes)} instances")

        print()

    except ImportError:
        print("ℹ️  Grounding DINO requires: pip install torch transformers")
        print()
        print("   API Preview:")
        print('   detector = GroundingDINO("IDEA-Research/grounding-dino-tiny")')
        print()
        print('   # Open-vocabulary detection with . as class separator')
        print('   text = "car . pedestrian . traffic sign . cyclist"')
        print("   detections = detector.detect(frame, text)")
        print()


def phase7c_custom_descriptions():
    """Phase 7c: Detection with custom descriptions."""
    print("=" * 70)
    print("Phase 7c: Custom Language Descriptions")
    print("=" * 70)

    try:
        detector = GroundingDINO("IDEA-Research/grounding-dino-tiny")

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Custom descriptions (free-form language)
        descriptions = [
            "a red car driving down the road",
            "a person standing on the sidewalk",
            "a traffic light indicating red",
        ]

        print(f"Custom descriptions (open-vocabulary):")
        for desc in descriptions:
            print(f"  • {desc}")

        print()

    except ImportError:
        print("ℹ️  Custom language descriptions")
        print()
        print("   # Free-form text descriptions")
        print('   prompt = "a red car approaching from the left"')
        print("   detections = detector.detect_with_custom_prompt(frame, prompt)")
        print()


def phase7c_with_sam3_refinement():
    """Phase 7c: SAM3 mask refinement."""
    print("=" * 70)
    print("Phase 7c: SAM3 Mask Refinement for Precise Boundaries")
    print("=" * 70)

    try:
        detector = GroundingDINO(
            "IDEA-Research/grounding-dino-tiny",
            use_sam3=True,
        )

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        text = "person . car"
        detections = detector.detect(frame, text)

        print(f"Detection with SAM3 refinement:")
        for obj_class, boxes in detections.items():
            print(f"\n  {obj_class}:")
            for box, conf, mask in boxes:
                if mask is not None:
                    print(f"    ✓ Bbox refined with SAM3 mask (conf: {conf:.3f})")
                else:
                    print(f"    Bbox only (conf: {conf:.3f})")

        print()

    except (ImportError, OSError):
        print("ℹ️  SAM3 mask refinement")
        print()
        print("   detector = GroundingDINO(use_sam3=True)")
        print()
        print("   # Returns (bbox, confidence, mask)")
        print("   detections = detector.detect(frame, text)")
        print("   for obj_class, boxes in detections.items():")
        print("       for bbox, conf, mask in boxes:")
        print("           if mask is not None:")
        print("               # Use mask for precise instance segmentation")
        print()


def phase7c_filtering():
    """Phase 7c: Detection filtering (confidence, NMS)."""
    print("=" * 70)
    print("Phase 7c: Detection Filtering and Post-Processing")
    print("=" * 70)

    try:
        detector = GroundingDINO(
            "IDEA-Research/grounding-dino-tiny",
            confidence_threshold=0.3,
            nms_threshold=0.5,
        )

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        text = "car . pedestrian"
        detections = detector.detect(frame, text)

        print(f"Post-processing:")
        print(f"  - Confidence threshold: {detector.confidence_threshold}")
        print(f"  - NMS threshold: {detector.nms_threshold}")
        print()

        # Filter by confidence
        filtered = detector.filter_by_confidence(detections, threshold=0.5)
        print(f"  After confidence filtering (>0.5):")
        for obj_class, boxes in filtered.items():
            print(f"    {obj_class}: {len(boxes)} detections")

        # Apply NMS
        nms_filtered = detector.apply_nms(filtered, threshold=0.5)
        print(f"  After NMS (IoU >0.5):")
        for obj_class, boxes in nms_filtered.items():
            print(f"    {obj_class}: {len(boxes)} detections")

        print()

    except (ImportError, OSError):
        print("ℹ️  Filtering operations")
        print()
        print("   # Confidence filtering")
        print("   filtered = detector.filter_by_confidence(detections, 0.5)")
        print()
        print("   # Non-maximum suppression")
        print("   nms_filtered = detector.apply_nms(filtered, 0.5)")
        print()


def phase7c_full_pipeline():
    """Phase 7c: Full multi-modal pipeline (SAM3 + CLIP + Grounding DINO)."""
    print("=" * 70)
    print("Phase 7c: Complete Multi-Modal Pipeline")
    print("=" * 70)

    # Step 1: Stitch panorama
    layout = get_waymo_layout()
    stitcher = CylindricalStitcher(layout, blend_method="laplacian")

    frames = {
        cam: np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        for cam in layout.cameras.keys()
    }

    panorama = stitcher.stitch(frames)
    pano_frame = panorama[0]

    print(f"Step 1: Panoramic stitching")
    print(f"  Input: 5 × 720×1280 → Output: {pano_frame.shape}")
    print()

    # Step 2: CLIP scene understanding
    try:
        clip = CLIPEmbedding("openai/clip-vit-b32", device="mlx")

        scene_scores = clip.scene_classification(pano_frame)
        best_scene = max(scene_scores, key=scene_scores.get)

        print(f"Step 2: CLIP scene classification")
        print(f"  Best scene: {best_scene} ({scene_scores[best_scene]:.1%})")
        print()

    except (ImportError, OSError):
        best_scene = "highway"
        print(f"Step 2: CLIP scene classification (requires transformers)")
        print()

    # Step 3: Grounding DINO object detection
    try:
        detector = GroundingDINO("IDEA-Research/grounding-dino-tiny", use_sam3=True)

        # Use scene context to guide detection
        if best_scene == "highway":
            text = "car . truck . motorcycle . road markings"
        elif best_scene == "city street":
            text = "car . pedestrian . traffic sign . traffic light . bicycle"
        else:
            text = "car . pedestrian . bicycle"

        detections = detector.detect(pano_frame, text)

        print(f"Step 3: Grounding DINO detection")
        print(f"  Text prompt: \"{text}\"")
        print(f"  Detections:")
        for obj_class, boxes in detections.items():
            print(f"    {obj_class}: {len(boxes)} instances")

        print()

    except (ImportError, OSError):
        print(f"Step 3: Grounding DINO detection (requires transformers)")
        print()

    # Step 4: SAM3 segmentation
    try:
        segmenter = SAM3Segmenter("facebook/sam3-small", device="mlx")

        masks = segmenter.segment(pano_frame)

        print(f"Step 4: SAM3 instance segmentation")
        print(f"  Instances: {len(masks)}")
        print()

    except (ImportError, OSError):
        print(f"Step 4: SAM3 segmentation (requires transformers)")
        print()

    # Combined reasoning
    print(f"Step 5: Multi-modal reasoning")
    print(f"  Scene: {best_scene}")
    print(f"  Objects: cars, pedestrians, traffic signs")
    print(f"  Instances: multiple segmented regions")
    print(f"  → Integrate for rich 3D scene understanding")
    print()


def main():
    """Run all Phase 7c examples."""
    print("\n" + "=" * 70)
    print("PyRoboFrames v0.5.3: Phase 7c - Grounding DINO Detection")
    print("=" * 70 + "\n")

    phase7c_basic_detection()
    phase7c_custom_descriptions()
    phase7c_with_sam3_refinement()
    phase7c_filtering()
    phase7c_full_pipeline()

    print("=" * 70)
    print("Phase 7c Examples Complete")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Phase 7d: Multi-modal fusion (SAM3 + CLIP + Grounding DINO)")
    print("  2. v0.5.3 Release with all Phase 7 capabilities")
    print()


if __name__ == "__main__":
    main()
