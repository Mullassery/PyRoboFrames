"""Multi-modal fusion example: Unified scene understanding.

Phase 7d: Combine SAM3 segmentation, CLIP embeddings, and Grounding DINO detection
into a single coherent perception pipeline for autonomous driving.

Sequential pipeline:
1. Grounding DINO: Detect objects with language (e.g., "car . pedestrian . cyclist")
2. SAM3: Refine with instance segmentation masks
3. CLIP: Classify scene and label objects semantically

This example demonstrates:
- Single frame understanding
- Batch processing
- Video processing with temporal consistency
- Scene serialization for downstream use
"""

import numpy as np
from pyroboframes.automotive import MultiModalFusion


def example_single_frame():
    """Understand a single driving frame with multi-modal fusion."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Frame Understanding")
    print("=" * 70)

    # Initialize fusion pipeline
    fusion = MultiModalFusion(
        detection_prompt="car . pedestrian . cyclist . truck . bus",
        device="cpu",
        use_sam3=True,
        use_clip=True,
    )

    # Simulate a frame from autonomous driving scenario
    # In practice, this would come from a camera or dataset
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    try:
        # Phase 1: Detect objects with Grounding DINO
        # Phase 2: Refine with SAM3 masks
        # Phase 3: Classify with CLIP
        scene = fusion.understand(frame)

        print(f"\nScene Classification:")
        print(f"  Primary scene: {scene.scene_type}")
        print(f"  Weather: {scene.weather}")
        print(f"\n  Scene confidence scores:")
        for scene_type, score in sorted(
            scene.scene_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]:
            print(f"    {scene_type}: {score:.2%}")

        print(f"\nDetected Objects ({len(scene.objects)}):")
        for i, obj in enumerate(scene.objects, 1):
            print(f"\n  [{i}] {obj.object_class}")
            print(f"      Confidence: {obj.confidence:.2%}")
            print(f"      Bbox: {obj.bbox}")
            if obj.semantic_label:
                print(f"      Semantic label: {obj.semantic_label}")
            if obj.mask is not None:
                print(f"      Mask shape: {obj.mask.shape}")

    except ImportError as e:
        print(f"\nNote: {e}")
        print(
            "Models not available yet. Structure is ready for when "
            "SAM3, CLIP, and Grounding DINO are released."
        )


def example_batch_processing():
    """Understand multiple frames in batch."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 70)

    fusion = MultiModalFusion(
        detection_prompt="car . pedestrian . cyclist",
        device="cpu",
    )

    # Simulate a batch of frames
    batch_frames = np.random.randint(0, 256, (4, 480, 640, 3), dtype=np.uint8)

    try:
        scenes = fusion.understand_batch(batch_frames)

        print(f"\nProcessed {len(scenes)} frames")
        for i, scene in enumerate(scenes):
            print(f"\n  Frame {i + 1}:")
            print(f"    Scene: {scene.scene_type}")
            print(f"    Objects: {len(scene.objects)}")

    except ImportError as e:
        print(f"\nNote: {e}")


def example_video_processing():
    """Understand video sequence with temporal consistency."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Video Processing with Temporal Consistency")
    print("=" * 70)

    fusion = MultiModalFusion(
        detection_prompt="car . pedestrian . traffic sign",
        device="cpu",
    )

    # Simulate a short video (10 frames)
    video_frames = np.random.randint(0, 256, (10, 480, 640, 3), dtype=np.uint8)

    try:
        # Process with temporal tracking to maintain object consistency
        scenes = fusion.understand_video(
            video_frames,
            temporal_consistency=True,
        )

        print(f"\nProcessed {len(scenes)} video frames")
        print(f"Temporal consistency tracking enabled")

        # Analyze temporal changes
        for t in range(min(3, len(scenes))):
            scene = scenes[t]
            print(f"\n  Frame {t}:")
            print(f"    Scene type: {scene.scene_type}")
            print(f"    Objects detected: {len(scene.objects)}")

    except ImportError as e:
        print(f"\nNote: {e}")


def example_scene_serialization():
    """Demonstrate scene serialization for downstream processing."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Scene Serialization")
    print("=" * 70)

    from pyroboframes.automotive import DetectedObject, SceneUnderstanding

    # Manually create a scene understanding for demo
    objects = [
        DetectedObject(
            object_class="car",
            bbox=np.array([100, 50, 250, 200]),
            confidence=0.94,
            semantic_label="sedan moving forward",
        ),
        DetectedObject(
            object_class="pedestrian",
            bbox=np.array([300, 100, 380, 400]),
            confidence=0.87,
            semantic_label="person crossing street",
        ),
        DetectedObject(
            object_class="traffic_sign",
            bbox=np.array([50, 20, 100, 80]),
            confidence=0.91,
            semantic_label="stop sign",
        ),
    ]

    scene = SceneUnderstanding(
        objects=objects,
        scene_type="intersection",
        scene_scores={
            "intersection": 0.92,
            "city street": 0.06,
            "parking lot": 0.02,
        },
        weather="clear day",
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )

    # Serialize to dictionary
    scene_dict = scene.to_dict()

    print("\nSerialized Scene Understanding:")
    import json

    print(json.dumps(scene_dict, indent=2))

    # Get object masks if available
    masks = scene.get_object_masks()
    if masks is None:
        print("\nNote: Objects don't have masks (masks would be [N, H, W])")


def example_custom_semantic_classes():
    """Demonstrate using custom semantic classes for labeling."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Custom Semantic Classes")
    print("=" * 70)

    fusion = MultiModalFusion(
        detection_prompt="vehicle . person . object",
        device="cpu",
    )

    # Custom semantic classes instead of just object class
    custom_classes = [
        "stationary vehicle",
        "moving vehicle",
        "pedestrian walking",
        "pedestrian standing",
        "parked car",
        "emergency vehicle",
    ]

    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    try:
        scene = fusion.understand(
            frame,
            semantic_classes=custom_classes,
        )

        print(f"\nUsing custom semantic classes:")
        for cls in custom_classes:
            print(f"  - {cls}")

        print(f"\nDetected and classified objects:")
        for obj in scene.objects:
            print(f"  {obj.object_class} → {obj.semantic_label}")

    except ImportError as e:
        print(f"\nNote: {e}")


def example_multi_view_fusion():
    """Demonstrate multi-view fusion (multiple cameras)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Multi-View Understanding (Multiple Cameras)")
    print("=" * 70)

    fusion = MultiModalFusion(
        detection_prompt="car . pedestrian . cyclist",
        device="cpu",
    )

    # Simulate multi-camera setup (Waymo layout)
    camera_frames = {
        "FRONT": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "FRONT_LEFT": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "FRONT_RIGHT": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "SIDE_LEFT": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "SIDE_RIGHT": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    }

    try:
        # Process each camera view
        scenes = {}
        for camera_name, frame in camera_frames.items():
            scene = fusion.understand(frame)
            scenes[camera_name] = scene

        print(f"\nProcessed {len(scenes)} camera views:")
        for camera_name, scene in scenes.items():
            print(
                f"  {camera_name:15s}: {len(scene.objects)} objects, "
                f"scene={scene.scene_type}"
            )

        # Could aggregate detections across views for 3D understanding
        total_objects = sum(len(s.objects) for s in scenes.values())
        print(f"\nTotal detections across all views: {total_objects}")

    except ImportError as e:
        print(f"\nNote: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PyRoboFrames: Phase 7d Multi-Modal Fusion Examples")
    print("=" * 70)
    print("\nSequential pipeline: Grounding DINO → SAM3 → CLIP")
    print("Output: Unified scene understanding with detected objects,")
    print("        instance masks, and semantic labels")

    example_single_frame()
    example_batch_processing()
    example_video_processing()
    example_scene_serialization()
    example_custom_semantic_classes()
    example_multi_view_fusion()

    print("\n" + "=" * 70)
    print("Examples Complete")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run on real autonomous driving data (Waymo/nuScenes)")
    print("2. Benchmark GPU performance (benchmark_gpu_performance.py)")
    print("3. Integrate with planning and control systems")
    print("4. Add temporal object tracking across video")
    print("5. Evaluate on safety-critical scenarios\n")
