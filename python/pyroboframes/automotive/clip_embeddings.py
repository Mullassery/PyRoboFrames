"""CLIP embeddings for scene understanding in autonomous driving.

Phase 7b: Multi-modal scene classification and understanding.
- Text-image similarity scoring
- Closed-set scene classification
- Open-vocabulary scene search
- Semantic understanding of driving scenarios
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple

import numpy as np


class CLIPEmbedding:
    """CLIP model for text-image similarity and scene understanding.

    Supports:
    - Scene classification (highway, city, parking, etc.)
    - Open-vocabulary scene search
    - Text-image similarity scoring
    - Batch processing of images and texts
    - Multi-modal embeddings for downstream tasks

    Usage:
        ```python
        from pyroboframes.automotive import CLIPEmbedding

        clip = CLIPEmbedding(model_id="openai/clip-vit-b32")

        # Classify a single frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        scene_types = ["highway", "city", "parking", "rural"]
        scores = clip.classify(frame, scene_types)
        # Returns [4] softmax scores

        # Open-vocabulary search across video
        panorama_seq = np.zeros((30, 480, 1728, 3), dtype=np.uint8)
        embeddings = clip.embed_frames_batch(panorama_seq)
        matches = clip.search_by_text(embeddings, ["approaching intersection"])
        # Returns top matches with scores
        ```
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-b32",
        device: Optional[str] = None,
        cache_embeddings: bool = True,
    ):
        """Initialize CLIP model.

        Args:
            model_id: CLIP model identifier from HF Hub
                - "openai/clip-vit-b32" (fast, good for real-time)
                - "openai/clip-vit-l14" (accurate, slower)
                - "openai/clip-vit-l14@336" (higher resolution)
            device: "cuda", "mlx", "cpu", or None for auto-detect
            cache_embeddings: Cache computed embeddings for repeated queries

        Raises:
            ImportError: If transformers or torch not available
        """
        self.model_id = model_id
        self.device = device or "cpu"
        self.cache_embeddings = cache_embeddings

        # Model & processor
        self.model = None
        self.processor = None
        self.tokenizer = None

        # Embedding cache
        self.embedding_cache = {}
        self.text_embedding_cache = {}

        self._load_model()

    def _load_model(self):
        """Load CLIP model from HF Hub.

        Downloads and caches model from HuggingFace Hub.
        Supports lazy loading - only loads on first inference if available.

        Raises:
            ImportError: If transformers or torch not available
        """
        self.model = None
        self.processor = None

        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch

            self.torch = torch

            try:
                # Download and load model
                self.processor = CLIPProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=None,  # Use default HF cache
                )

                self.model = CLIPModel.from_pretrained(
                    self.model_id,
                    device_map=self._get_device_map(),
                )

                # Move to device
                if self.device == "cuda":
                    self.model = self.model.cuda()
                elif self.device == "mlx":
                    self.model = self.model.to("cpu")
                else:
                    self.model = self.model.to("cpu")

                self.model.eval()

            except Exception as model_error:
                print(f"Note: CLIP model {self.model_id} load issue: {model_error}")
                self.model = None
                self.processor = None

        except ImportError as e:
            raise ImportError(
                f"CLIP requires: pip install torch transformers. Error: {e}"
            )

    def _get_device_map(self) -> str:
        """Get device map for model loading.

        Returns:
            Device specification for transformers
        """
        if self.device == "cuda":
            return "cuda"
        else:
            return "cpu"

    def embed_frame(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Embed a single frame.

        Args:
            image: [H, W, 3] uint8 image

        Returns:
            [D] float32 embedding (D=512 for ViT-B32, D=768 for ViT-L14)
        """
        if self.model is None:
            raise ImportError("CLIP model failed to load")

        # Normalize to [0, 1]
        image_norm = image.astype(np.float32) / 255.0

        # Process image
        inputs = self.processor(
            images=image_norm,
            return_tensors="pt",
        )

        # Move to device
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Embed
        with self.torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]

    def embed_frames_batch(
        self,
        frames: np.ndarray,
    ) -> np.ndarray:
        """Embed batch of frames.

        Args:
            frames: [B, H, W, 3] uint8 batch of images

        Returns:
            [B, D] float32 embeddings (normalized)
        """
        if self.model is None:
            raise ImportError("CLIP model failed to load")

        batch_size = frames.shape[0]
        embeddings = []

        for b in range(batch_size):
            embedding = self.embed_frame(frames[b])
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        """Embed a text description.

        Args:
            text: Text description of scene/object

        Returns:
            [D] float32 embedding (normalized)
        """
        if self.model is None:
            raise ImportError("CLIP model failed to load")

        # Check cache
        cache_key = text.lower()
        if cache_key in self.text_embedding_cache:
            return self.text_embedding_cache[cache_key]

        # Tokenize & embed text
        inputs = self.processor(text=text, return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with self.torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        embedding = text_features.cpu().numpy()[0]

        # Cache if enabled
        if self.cache_embeddings:
            self.text_embedding_cache[cache_key] = embedding

        return embedding

    def embed_texts_batch(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """Embed batch of text descriptions.

        Args:
            texts: List of text descriptions

        Returns:
            [N, D] float32 embeddings (normalized)
        """
        embeddings = [self.embed_text(text) for text in texts]
        return np.array(embeddings, dtype=np.float32)

    def classify(
        self,
        image: np.ndarray,
        classes: List[str],
    ) -> np.ndarray:
        """Classify image into one of provided classes.

        Args:
            image: [H, W, 3] uint8 image
            classes: List of class names

        Returns:
            [N] softmax scores (sum to 1.0)
        """
        # Embed image and class texts
        image_embedding = self.embed_frame(image)
        text_embeddings = self.embed_texts_batch(classes)

        # Compute similarity (dot product on normalized embeddings)
        similarities = image_embedding @ text_embeddings.T  # [N]

        # Temperature scaling (CLIP default)
        temperature = 100.0
        logits = similarities * temperature

        # Softmax
        scores = np.exp(logits) / np.sum(np.exp(logits))

        return scores.astype(np.float32)

    def classify_batch(
        self,
        frames: np.ndarray,
        classes: List[str],
    ) -> np.ndarray:
        """Classify batch of frames.

        Args:
            frames: [B, H, W, 3] uint8 batch
            classes: List of class names

        Returns:
            [B, N] softmax scores
        """
        batch_size = frames.shape[0]
        scores_list = []

        for b in range(batch_size):
            scores = self.classify(frames[b], classes)
            scores_list.append(scores)

        return np.array(scores_list, dtype=np.float32)

    def similarity(
        self,
        image: np.ndarray,
        text: str,
    ) -> float:
        """Compute similarity between image and text.

        Args:
            image: [H, W, 3] uint8 image
            text: Text description

        Returns:
            float in [0, 1] (after softmax normalization)
        """
        image_embedding = self.embed_frame(image)
        text_embedding = self.embed_text(text)

        # Cosine similarity (already normalized)
        sim = float(np.dot(image_embedding, text_embedding))

        # Clamp to [0, 1]
        sim = (sim + 1.0) / 2.0  # Map [-1, 1] to [0, 1]

        return sim

    def search_by_text(
        self,
        frame_embeddings: np.ndarray,
        queries: List[str],
        top_k: int = 3,
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Search frames by text query.

        Args:
            frame_embeddings: [B, D] embeddings from embed_frames_batch
            queries: List of text queries
            top_k: Return top-k matches per query

        Returns:
            Dict mapping query → [(frame_idx, similarity_score)]
        """
        results = {}

        for query in queries:
            query_embedding = self.embed_text(query)

            # Compute similarity to all frames
            similarities = frame_embeddings @ query_embedding  # [B]

            # Get top-k
            top_indices = np.argsort(-similarities)[:top_k]
            top_scores = similarities[top_indices]

            results[query] = list(zip(top_indices, top_scores))

        return results

    def scene_classification(
        self,
        image: np.ndarray,
    ) -> Dict[str, float]:
        """Classify driving scene into standard categories.

        Pre-defined scene types for autonomous driving.

        Args:
            image: [H, W, 3] uint8 image

        Returns:
            Dict mapping scene type → confidence score
        """
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

        scores = self.classify(image, scene_classes)

        return {cls: float(score) for cls, score in zip(scene_classes, scores)}

    def weather_classification(
        self,
        image: np.ndarray,
    ) -> Dict[str, float]:
        """Classify weather/lighting conditions.

        Args:
            image: [H, W, 3] uint8 image

        Returns:
            Dict mapping condition → confidence score
        """
        weather_classes = [
            "clear day",
            "cloudy",
            "rainy",
            "snowy",
            "foggy",
            "night time",
            "dawn/dusk",
        ]

        scores = self.classify(image, weather_classes)

        return {cls: float(score) for cls, score in zip(weather_classes, scores)}

    def object_presence(
        self,
        image: np.ndarray,
    ) -> Dict[str, float]:
        """Detect presence of common driving objects.

        Args:
            image: [H, W, 3] uint8 image

        Returns:
            Dict mapping object type → confidence score
        """
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

        scores = self.classify(image, object_classes)

        return {cls: float(score) for cls, score in zip(object_classes, scores)}

    def reset_cache(self):
        """Clear embedding caches."""
        self.embedding_cache.clear()
        self.text_embedding_cache.clear()

    def __repr__(self) -> str:
        return (
            f"CLIPEmbedding("
            f"model='{self.model_id}', "
            f"device='{self.device}', "
            f"cache={self.cache_embeddings}"
            f")"
        )
