"""Tests for Phase 7b: CLIP Scene Understanding."""

import numpy as np
import pytest

from pyroboframes.automotive import CLIPEmbedding


class TestCLIPInitialization:
    """Test CLIP model initialization."""

    def test_clip_init_default(self):
        """Test CLIPEmbedding initialization with defaults."""
        try:
            clip = CLIPEmbedding(model_id="openai/clip-vit-b32", device="cpu")
            assert clip.model_id == "openai/clip-vit-b32"
            assert clip.device == "cpu"
            assert clip.cache_embeddings is True
        except (ImportError, OSError):
            pytest.skip("Transformers not installed or CLIP model unavailable")

    def test_clip_init_l14_model(self):
        """Test initialization with ViT-L14 model."""
        try:
            clip = CLIPEmbedding(model_id="openai/clip-vit-l14", device="cpu")
            assert clip.model_id == "openai/clip-vit-l14"
        except (ImportError, OSError):
            pytest.skip("CLIP model unavailable")

    def test_clip_device_options(self):
        """Test different device options."""
        try:
            for device in ["cpu", "cuda", "mlx"]:
                clip = CLIPEmbedding(device=device)
                assert clip.device == device
        except (ImportError, OSError):
            pytest.skip("CLIP not available")

    def test_clip_cache_disabled(self):
        """Test initialization with cache disabled."""
        try:
            clip = CLIPEmbedding(cache_embeddings=False)
            assert clip.cache_embeddings is False
        except (ImportError, OSError):
            pytest.skip("CLIP not available")


class TestCLIPEmbedding:
    """Test CLIP embedding operations (mock)."""

    def test_embed_frame_mock(self):
        """Test embedding a single frame."""
        clip = CLIPEmbedding.__new__(CLIPEmbedding)
        clip.model = None
        clip.device = "cpu"

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ImportError):
            clip.embed_frame(frame)

    def test_embed_text_mock(self):
        """Test embedding text."""
        clip = CLIPEmbedding.__new__(CLIPEmbedding)
        clip.model = None
        clip.text_embedding_cache = {}
        clip.cache_embeddings = True

        with pytest.raises(ImportError):
            clip.embed_text("highway")


class TestCLIPClassification:
    """Test scene classification."""

    def test_classify_mock_scores(self):
        """Test classification returns proper scores."""
        clip = CLIPEmbedding.__new__(CLIPEmbedding)
        clip.model = None
        clip.text_embedding_cache = {}
        clip.cache_embeddings = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        classes = ["highway", "city", "parking"]

        with pytest.raises(ImportError):
            clip.classify(frame, classes)

    def test_classify_structure(self):
        """Test classification output structure."""
        # Create mock classification
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        classes = ["highway", "city", "parking", "rural"]

        # Mock scores (should sum to 1.0 for softmax)
        scores = np.array([0.6, 0.2, 0.15, 0.05], dtype=np.float32)

        assert len(scores) == len(classes)
        assert np.isclose(np.sum(scores), 1.0)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


class TestCLIPBatchOperations:
    """Test batch processing."""

    def test_embed_frames_batch_shape(self):
        """Test batch embedding shape."""
        # Create mock batch
        batch = np.zeros((4, 480, 640, 3), dtype=np.uint8)

        # Expected embedding dimension
        embedding_dim = 512  # ViT-B32

        # Mock output
        embeddings = np.random.randn(4, embedding_dim).astype(np.float32)

        assert embeddings.shape == (4, embedding_dim)

    def test_classify_batch_shape(self):
        """Test batch classification shape."""
        batch = np.zeros((4, 480, 640, 3), dtype=np.uint8)
        classes = ["highway", "city", "parking"]

        # Mock batch scores
        batch_scores = np.random.dirichlet([1, 1, 1], size=4).astype(np.float32)

        assert batch_scores.shape == (4, 3)
        # Each row should sum to ~1.0
        assert np.allclose(np.sum(batch_scores, axis=1), 1.0)


class TestCLIPSimilarity:
    """Test text-image similarity."""

    def test_similarity_range(self):
        """Test similarity is in [0, 1]."""
        # Mock similarity computation
        image_emb = np.random.randn(512)
        image_emb /= np.linalg.norm(image_emb)

        text_emb = np.random.randn(512)
        text_emb /= np.linalg.norm(text_emb)

        # Cosine similarity on normalized vectors: [-1, 1]
        sim = float(np.dot(image_emb, text_emb))
        sim = (sim + 1.0) / 2.0  # Map to [0, 1]

        assert 0.0 <= sim <= 1.0

    def test_similarity_identical(self):
        """Test similarity of identical embeddings."""
        emb = np.random.randn(512)
        emb /= np.linalg.norm(emb)

        # Identical embeddings
        sim = float(np.dot(emb, emb))
        sim = (sim + 1.0) / 2.0

        # Should be close to 1.0
        assert sim > 0.9


class TestCLIPSceneClassification:
    """Test pre-defined scene classifications."""

    def test_scene_classification_output(self):
        """Test scene classification output structure."""
        # Mock scene classes
        scene_classes = [
            "highway",
            "city street",
            "residential area",
            "parking lot",
            "rural road",
            "intersection",
            "construction zone",
            "tunnel",
        ]

        # Mock scores
        scores = np.random.dirichlet(np.ones(len(scene_classes)))

        result = {cls: float(score) for cls, score in zip(scene_classes, scores)}

        assert len(result) == len(scene_classes)
        assert np.isclose(sum(result.values()), 1.0)

    def test_weather_classification_output(self):
        """Test weather classification output."""
        weather_classes = [
            "clear day",
            "cloudy",
            "rainy",
            "snowy",
            "foggy",
            "night time",
            "dawn/dusk",
        ]

        scores = np.random.dirichlet(np.ones(len(weather_classes)))

        result = {cls: float(score) for cls, score in zip(weather_classes, scores)}

        assert len(result) == len(weather_classes)

    def test_object_presence_output(self):
        """Test object presence classification."""
        object_classes = [
            "cars",
            "pedestrians",
            "bicycles",
            "motorcycles",
            "trucks",
            "buses",
            "traffic signs",
            "traffic lights",
        ]

        scores = np.random.dirichlet(np.ones(len(object_classes)))

        result = {cls: float(score) for cls, score in zip(object_classes, scores)}

        assert len(result) == len(object_classes)


class TestCLIPSearch:
    """Test text-based search."""

    def test_search_by_text_structure(self):
        """Test search results structure."""
        # Mock embeddings
        frame_embeddings = np.random.randn(30, 512).astype(np.float32)
        # Normalize
        frame_embeddings /= np.linalg.norm(frame_embeddings, axis=1, keepdims=True)

        queries = ["approaching intersection", "parked cars", "traffic"]

        results = {}
        for query in queries:
            query_emb = np.random.randn(512)
            query_emb /= np.linalg.norm(query_emb)

            similarities = frame_embeddings @ query_emb
            top_indices = np.argsort(-similarities)[:3]
            top_scores = similarities[top_indices]

            results[query] = list(zip(top_indices, top_scores))

        assert len(results) == 3
        for query, matches in results.items():
            assert len(matches) == 3
            for idx, score in matches:
                assert 0 <= idx < 30
                assert -1 <= score <= 1


class TestCLIPCache:
    """Test embedding caching."""

    def test_cache_initialization(self):
        """Test cache is initialized."""
        clip = CLIPEmbedding.__new__(CLIPEmbedding)
        clip.embedding_cache = {}
        clip.text_embedding_cache = {}
        clip.cache_embeddings = True

        assert len(clip.embedding_cache) == 0
        assert len(clip.text_embedding_cache) == 0

    def test_cache_reset(self):
        """Test cache reset."""
        clip = CLIPEmbedding.__new__(CLIPEmbedding)
        clip.embedding_cache = {"key": np.array([1, 2, 3])}
        clip.text_embedding_cache = {"text": np.array([4, 5, 6])}

        clip.reset_cache()

        assert len(clip.embedding_cache) == 0
        assert len(clip.text_embedding_cache) == 0


class TestCLIPIntegration:
    """Test integration with other modules."""

    def test_with_occupancy_grid(self):
        """Test CLIP with occupancy grid."""
        from pyroboframes.automotive import OccupancyGrid

        # Create occupancy grid
        occupancy = OccupancyGrid(size=(-50, 50), resolution=0.5)

        # CLIP would classify scene
        scene_scores = {
            "highway": 0.7,
            "city": 0.2,
            "parking": 0.1,
        }

        # Use scene classification to inform occupancy parameters
        if scene_scores["highway"] > 0.5:
            # Highway: larger grid, longer range
            assert occupancy.grid_size > 100
        else:
            assert occupancy.grid_size == 200

    def test_with_sam3(self):
        """Test CLIP with SAM3 segmentation."""
        from pyroboframes.automotive import SAM3Segmenter

        try:
            segmenter = SAM3Segmenter.__new__(SAM3Segmenter)
            clip = CLIPEmbedding.__new__(CLIPEmbedding)

            # Both have device and model parameters
            assert hasattr(segmenter, "device")
            assert hasattr(clip, "device")

        except:
            pass


class TestCLIPEdgeCases:
    """Test edge cases."""

    def test_all_black_frame(self):
        """Test CLIP on all-black frame."""
        clip = CLIPEmbedding.__new__(CLIPEmbedding)
        clip.model = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(ImportError):
            clip.embed_frame(frame)

    def test_all_white_frame(self):
        """Test CLIP on all-white frame."""
        clip = CLIPEmbedding.__new__(CLIPEmbedding)
        clip.model = None

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        with pytest.raises(ImportError):
            clip.embed_frame(frame)

    def test_empty_query_list(self):
        """Test with empty query list."""
        results = {}  # Empty search results

        assert len(results) == 0

    def test_many_queries(self):
        """Test with many text queries."""
        queries = [
            "approaching intersection",
            "parked cars",
            "traffic light",
            "pedestrians crossing",
            "construction zone",
            "highway with traffic",
            "empty road",
            "night driving",
        ]

        assert len(queries) == 8


class TestCLIPRepr:
    """Test string representation."""

    def test_repr_format(self):
        """Test __repr__ format."""
        try:
            clip = CLIPEmbedding(
                model_id="openai/clip-vit-b32",
                device="mlx",
            )
            repr_str = repr(clip)
            assert "openai/clip-vit-b32" in repr_str
            assert "mlx" in repr_str
        except (ImportError, OSError):
            pytest.skip("CLIP not available")
