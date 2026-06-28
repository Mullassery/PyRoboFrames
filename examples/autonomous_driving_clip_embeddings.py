"""
Phase 7b: CLIP Scene Understanding for Autonomous Driving.

Demonstrates:
- Scene classification (highway, city, parking, etc.)
- Weather/lighting condition detection
- Object presence scoring
- Text-based frame search
- Integration with SAM3 segmentation
- Multi-modal reasoning
"""

import numpy as np

from pyroboframes.automotive import (
    CylindricalStitcher,
    CLIPEmbedding,
    SAM3Segmenter,
    get_waymo_layout,
)


def phase7b_scene_classification():
    """Phase 7b: Classify driving scenes."""
    print("=" * 70)
    print("Phase 7b: Scene Classification with CLIP")
    print("=" * 70)

    try:
        # Initialize CLIP
        clip = CLIPEmbedding(model_id="openai/clip-vit-b32", device="mlx")

        print(f"✓ CLIP Embedder initialized")
        print(f"  - Model: openai/clip-vit-b32 (512D embeddings)")
        print(f"  - Device: MLX (Apple Silicon)")
        print(f"  - Embedding cache: enabled")
        print()

        # Single frame classification
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        scene_scores = clip.scene_classification(frame)

        print(f"Scene classification results:")
        for scene, score in sorted(scene_scores.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 40)
            print(f"  {scene:20s} {score:.3f}  {bar}")
        print()

    except ImportError:
        print("ℹ️  CLIP requires: pip install torch transformers")
        print()
        print("   API Preview:")
        print('   clip = CLIPEmbedding("openai/clip-vit-b32")')
        print()
        print("   # Pre-defined scene classification")
        print("   scores = clip.scene_classification(frame)")
        print("   # Returns: {scene_type -> confidence_score}")
        print()


def phase7b_weather_detection():
    """Phase 7b: Detect weather and lighting conditions."""
    print("=" * 70)
    print("Phase 7b: Weather/Lighting Condition Detection")
    print("=" * 70)

    try:
        clip = CLIPEmbedding(model_id="openai/clip-vit-b32", device="cpu")

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        weather_scores = clip.weather_classification(frame)

        print(f"Weather/lighting conditions:")
        for condition, score in sorted(weather_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {condition:15s} {score:.3f}")
        print()

        # Determine dominant condition
        dominant = max(weather_scores, key=weather_scores.get)
        print(f"Dominant condition: {dominant} ({weather_scores[dominant]:.1%})")
        print()

    except (ImportError, OSError):
        print("ℹ️  Weather classification")
        print()
        print("   # Detects: clear, cloudy, rainy, snowy, foggy, night, dawn/dusk")
        print("   scores = clip.weather_classification(frame)")
        print()


def phase7b_object_detection():
    """Phase 7b: Detect presence of driving objects."""
    print("=" * 70)
    print("Phase 7b: Object Presence Detection")
    print("=" * 70)

    try:
        clip = CLIPEmbedding(model_id="openai/clip-vit-b32")

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        object_scores = clip.object_presence(frame)

        print(f"Object presence scores:")
        for obj_type, score in sorted(object_scores.items(), key=lambda x: x[1], reverse=True):
            confidence = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
            print(f"  {confidence} {obj_type:15s} {score:.3f}")
        print()

        # Risk assessment
        pedestrian_score = object_scores["pedestrians"]
        vehicle_score = object_scores["cars"]

        risk_level = "HIGH" if pedestrian_score > 0.7 else "MEDIUM" if vehicle_score > 0.5 else "LOW"
        print(f"Risk level: {risk_level}")
        print()

    except (ImportError, OSError):
        print("ℹ️  Object presence detection")
        print()
        print("   # Detects: cars, pedestrians, bicycles, motorcycles")
        print("   #           trucks, buses, traffic signs, traffic lights")
        print("   scores = clip.object_presence(frame)")
        print()


def phase7b_text_search():
    """Phase 7b: Search frames by text query."""
    print("=" * 70)
    print("Phase 7b: Text-Based Frame Search")
    print("=" * 70)

    try:
        clip = CLIPEmbedding(model_id="openai/clip-vit-b32", device="mlx")

        # Simulate video: 30 frames
        T = 30
        video = np.random.randint(0, 256, (T, 480, 640, 3), dtype=np.uint8)

        print(f"Embedding {T}-frame sequence...")
        frame_embeddings = clip.embed_frames_batch(video)
        print(f"  ✓ Created {frame_embeddings.shape[0]} × {frame_embeddings.shape[1]}D embeddings")
        print()

        # Search queries
        queries = [
            "approaching intersection",
            "heavy traffic",
            "parked cars",
        ]

        print(f"Searching for key moments...")
        results = clip.search_by_text(frame_embeddings, queries, top_k=3)

        for query, matches in results.items():
            print(f"\n  Query: \"{query}\"")
            print(f"  Top matches:")
            for frame_idx, similarity in matches:
                bar = "▓" * int(similarity * 40) if similarity > 0 else ""
                print(f"    Frame {frame_idx:2d}  {similarity:.3f}  {bar}")
        print()

    except (ImportError, OSError):
        print("ℹ️  Text-based search")
        print()
        print("   # Embed entire video")
        print("   embeddings = clip.embed_frames_batch(video)  # [T, 512]")
        print()
        print("   # Search for key moments")
        print("   queries = ['approaching intersection', 'heavy traffic']")
        print("   matches = clip.search_by_text(embeddings, queries, top_k=5)")
        print()


def phase7b_with_sam3():
    """Phase 7b: Integrate CLIP with SAM3 segmentation."""
    print("=" * 70)
    print("Phase 7b: CLIP + SAM3 Multi-Modal Understanding")
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
    print(f"  - Input: 5 × 720×1280 → Output: {pano_frame.shape}")
    print()

    # Step 2: Segment with SAM3
    try:
        segmenter = SAM3Segmenter("facebook/sam3-small", device="mlx")
        masks = segmenter.segment(pano_frame)

        print(f"Step 2: SAM3 segmentation (Phase 7a)")
        print(f"  - Instances detected: {len(masks)}")
        print()

    except (ImportError, OSError):
        masks = None
        print(f"Step 2: SAM3 segmentation (requires torch transformers)")
        print()

    # Step 3: Classify scene with CLIP
    try:
        clip = CLIPEmbedding("openai/clip-vit-b32", device="mlx")

        scene_scores = clip.scene_classification(pano_frame)
        best_scene = max(scene_scores, key=scene_scores.get)

        print(f"Step 3: CLIP scene classification (Phase 7b)")
        print(f"  - Best match: {best_scene} ({scene_scores[best_scene]:.1%})")
        print()

        # Combined reasoning
        print(f"Step 4: Multi-modal reasoning")
        print(f"  - Scene: {best_scene}")
        if masks is not None:
            print(f"  - Instances: {len(masks)} detected")
        print(f"  - Integration: Apply scene context to instance understanding")
        print()

        # Example: if highway, expect vehicles and high speed
        if best_scene == "highway":
            print(f"  ✓ Highway context:")
            print(f"    - Expect: vehicles, lane markings, high speed")
            print(f"    - Strategy: track vehicles for collision avoidance")
        elif best_scene == "city street":
            print(f"  ✓ City context:")
            print(f"    - Expect: pedestrians, traffic lights, parked cars")
            print(f"    - Strategy: monitor pedestrians, obey signals")
        print()

    except (ImportError, OSError):
        print(f"Step 3: CLIP scene classification (requires torch transformers)")
        print()


def phase7b_batch_analysis():
    """Phase 7b: Batch analyze video sequence."""
    print("=" * 70)
    print("Phase 7b: Batch Video Analysis")
    print("=" * 70)

    try:
        clip = CLIPEmbedding("openai/clip-vit-b32", device="mlx")

        # Simulate video: 10 frames
        T = 10
        video = np.random.randint(0, 256, (T, 480, 640, 3), dtype=np.uint8)

        print(f"Analyzing {T}-frame sequence...")
        print()

        # Classify each frame
        print(f"Frame-by-frame scene classification:")
        scene_classes = ["highway", "city street", "parking lot"]

        for t in range(T):
            scores = clip.classify(video[t], scene_classes)
            best = scene_classes[np.argmax(scores)]
            conf = scores[np.argmax(scores)]
            print(f"  Frame {t:2d}: {best:15s} ({conf:.2f})")

        print()

    except (ImportError, OSError):
        print("ℹ️  Batch analysis")
        print()
        print("   for t, frame in enumerate(video):")
        print('       scores = clip.classify(frame, ["highway", "city", "parking"])')
        print("       scene = scene_classes[np.argmax(scores)]")
        print()


def main():
    """Run all Phase 7b examples."""
    print("\n" + "=" * 70)
    print("PyRoboFrames v0.5.3: Phase 7b - CLIP Scene Understanding")
    print("=" * 70 + "\n")

    phase7b_scene_classification()
    phase7b_weather_detection()
    phase7b_object_detection()
    phase7b_text_search()
    phase7b_with_sam3()
    phase7b_batch_analysis()

    print("=" * 70)
    print("Phase 7b Examples Complete")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Phase 7c: Grounding DINO (open-vocabulary detection)")
    print("  2. Phase 7d: Multi-modal fusion (SAM3 + CLIP + Grounding DINO)")
    print()


if __name__ == "__main__":
    main()
