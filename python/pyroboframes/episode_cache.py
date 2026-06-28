"""Episode-level RAM cache for repeated-pass training.

Pre-decodes entire episodes (all frames, all cameras) into numpy arrays and keeps them in an
LRU cache at episode granularity. Useful for behavior cloning with multiple epochs over small
datasets where re-decoding the same videos is the bottleneck.

```python
from pyroboframes.episode_cache import EpisodeCache

cache = EpisodeCache(ds, max_episodes=8)

for epoch in range(10):
    for ep_idx in range(ds.num_episodes()):
        ep = cache.get_episode(ep_idx)
        states = ep["observation.state"]   # [T, D]
        frames = ep.get("observation.images.top")  # [T, H, W, 3] or None
        # ... training step
```
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from ._core import RoboFrameDataset


class EpisodeCache:
    """LRU cache of decoded episodes at episode granularity.

    Each entry holds all tabular features and (optionally) decoded video frames for one episode.
    When the cache is full the least-recently-used episode is evicted.

    Args:
        dataset: RoboFrameDataset to load from.
        max_episodes: Maximum number of episodes to keep in RAM simultaneously.
        cameras: Camera keys to decode. None = no video (tabular only, much faster).
        output: Output format for arrays — ``"numpy"`` (default), ``"torch"``, or ``"mlx"``.
    """

    def __init__(
        self,
        dataset: "RoboFrameDataset",
        max_episodes: int = 4,
        cameras: Optional[list[str]] = None,
        output: str = "numpy",
    ) -> None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be >= 1")
        self.dataset = dataset
        self.max_episodes = max_episodes
        self.cameras = cameras or []
        self.output = output
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None

    def get_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        """Return all arrays for this episode, decoding and caching if needed.

        Args:
            episode_index: Episode to fetch.

        Returns:
            Dict mapping feature name → ``[T, ...]`` array. Camera keys map to
            ``[T, H, W, 3]`` uint8 arrays when cameras are configured.
        """
        with self._lock:
            if episode_index in self._cache:
                self._cache.move_to_end(episode_index)
                return self._cache[episode_index]

        # Decode outside the lock so other threads can still hit the cache.
        data = self._decode_episode(episode_index)

        with self._lock:
            # Another thread may have decoded the same episode concurrently; prefer the cached copy.
            if episode_index not in self._cache:
                if len(self._cache) >= self.max_episodes:
                    self._cache.popitem(last=False)  # evict LRU
                self._cache[episode_index] = data
            else:
                self._cache.move_to_end(episode_index)
            return self._cache[episode_index]

    def prefetch(self, episode_indices: list[int]) -> None:
        """Start background decoding of the given episodes (non-blocking).

        Already-cached episodes are skipped. Only one prefetch thread runs at a time;
        calling prefetch() again while one is running has no effect.
        """
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            return

        to_fetch = [i for i in episode_indices if i not in self._cache]
        if not to_fetch:
            return

        def _worker() -> None:
            for ep_idx in to_fetch:
                with self._lock:
                    if ep_idx in self._cache:
                        continue
                data = self._decode_episode(ep_idx)
                with self._lock:
                    if ep_idx not in self._cache:
                        if len(self._cache) >= self.max_episodes:
                            self._cache.popitem(last=False)
                        self._cache[ep_idx] = data

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def clear(self) -> None:
        """Evict all cached episodes."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def cached_episodes(self) -> list[int]:
        """Return a list of currently cached episode indices (MRU order)."""
        with self._lock:
            return list(reversed(list(self._cache.keys())))

    def _decode_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        """Load all frames of one episode via the dataset loader."""
        episodes = self.dataset.episodes()
        if episode_index >= len(episodes):
            raise IndexError(
                f"episode_index {episode_index} out of range "
                f"(dataset has {len(episodes)} episodes)"
            )
        ep = episodes[episode_index]
        ep_length = ep.get("length", 0)
        if ep_length == 0:
            return {}

        loader = self.dataset.loader(
            batch_size=ep_length,
            shuffle=False,
            episodes=[episode_index],
            cameras=self.cameras,
            output=self.output,
            num_workers=0,
        )

        batch = next(iter(loader), None)
        if batch is None:
            return {}

        # Ensure everything is numpy for consistent cache semantics.
        result: dict[str, np.ndarray] = {}
        for key, val in batch.items():
            if hasattr(val, "numpy"):
                result[key] = val.numpy()
            elif hasattr(val, "__array__"):
                result[key] = np.asarray(val)
            elif isinstance(val, np.ndarray):
                result[key] = val
            # Skip non-array values (e.g. scalars, metadata)
        return result
