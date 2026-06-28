"""Tests for intelligent episode caching (EpisodeCache and configurable FrameCache)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import pytest
from test_loader import make_dataset


def test_episode_cache_returns_consistent_arrays(tmp_path):
    from pyroboframes.episode_cache import EpisodeCache
    import pyroboframes as prf

    ds = prf.RoboFrameDataset.from_path(str(make_dataset(tmp_path, episodes=3, length=10)))
    cache = EpisodeCache(ds, max_episodes=3, cameras=[], output="numpy")

    ep0_first = cache.get_episode(0)
    ep0_second = cache.get_episode(0)
    # Same object returned from cache on second call.
    assert ep0_first is ep0_second


def test_episode_cache_lru_eviction(tmp_path):
    from pyroboframes.episode_cache import EpisodeCache
    import pyroboframes as prf

    ds = prf.RoboFrameDataset.from_path(str(make_dataset(tmp_path, episodes=4, length=10)))
    cache = EpisodeCache(ds, max_episodes=2, cameras=[], output="numpy")

    cache.get_episode(0)
    cache.get_episode(1)
    assert len(cache) == 2

    # Accessing episode 2 should evict episode 0 (LRU).
    cache.get_episode(2)
    assert len(cache) == 2
    assert 0 not in cache.cached_episodes()
    assert 2 in cache.cached_episodes()


def test_episode_cache_arrays_have_correct_shape(tmp_path):
    from pyroboframes.episode_cache import EpisodeCache
    import pyroboframes as prf

    root = make_dataset(tmp_path, episodes=2, length=15)
    ds = prf.RoboFrameDataset.from_path(str(root))
    cache = EpisodeCache(ds, max_episodes=4, cameras=[], output="numpy")

    ep = cache.get_episode(0)
    for key, arr in ep.items():
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == 15, f"Expected T=15 for {key}, got {arr.shape}"


def test_episode_cache_clear(tmp_path):
    from pyroboframes.episode_cache import EpisodeCache
    import pyroboframes as prf

    ds = prf.RoboFrameDataset.from_path(str(make_dataset(tmp_path, episodes=3, length=5)))
    cache = EpisodeCache(ds, max_episodes=3, cameras=[], output="numpy")
    cache.get_episode(0)
    cache.get_episode(1)
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


def test_episode_cache_prefetch_nonblocking(tmp_path):
    from pyroboframes.episode_cache import EpisodeCache
    import pyroboframes as prf
    import time

    ds = prf.RoboFrameDataset.from_path(str(make_dataset(tmp_path, episodes=3, length=5)))
    cache = EpisodeCache(ds, max_episodes=3, cameras=[], output="numpy")

    # prefetch should return immediately.
    t0 = time.monotonic()
    cache.prefetch([0, 1, 2])
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, "prefetch() blocked for too long"
