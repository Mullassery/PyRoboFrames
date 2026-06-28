"""Tests for distributed loading (shard_episodes, RemoteDataset, RayDistributedLoader)."""

import pytest
from pyroboframes.distributed import shard_episodes


# ---------------------------------------------------------------------------
# shard_episodes — pure function, no deps
# ---------------------------------------------------------------------------


def test_shard_episodes_round_robin():
    assert shard_episodes(10, 3, 0) == [0, 3, 6, 9]
    assert shard_episodes(10, 3, 1) == [1, 4, 7]
    assert shard_episodes(10, 3, 2) == [2, 5, 8]


def test_shard_episodes_no_overlap():
    world_size = 4
    all_assigned = []
    for rank in range(world_size):
        all_assigned.extend(shard_episodes(12, world_size, rank))
    assert sorted(all_assigned) == list(range(12))


def test_shard_episodes_single_worker():
    result = shard_episodes(5, 1, 0)
    assert result == [0, 1, 2, 3, 4]


def test_shard_episodes_more_workers_than_episodes():
    # 3 episodes, 5 workers — some workers get 0, some get 1.
    combined = []
    for rank in range(5):
        combined.extend(shard_episodes(3, 5, rank))
    assert sorted(combined) == [0, 1, 2]


def test_shard_episodes_invalid_world_size():
    with pytest.raises(ValueError, match="world_size"):
        shard_episodes(10, 0, 0)


def test_shard_episodes_invalid_rank():
    with pytest.raises(ValueError, match="rank"):
        shard_episodes(10, 3, 3)  # rank must be < world_size

    with pytest.raises(ValueError, match="rank"):
        shard_episodes(10, 3, -1)


def test_shard_episodes_returns_sorted():
    result = shard_episodes(20, 4, 2)
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# RemoteDataset — requires fsspec; skip if not available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not pytest.importorskip("fsspec", reason="fsspec not installed"),
    reason="fsspec not installed",
)
def test_remote_dataset_from_s3_constructs():
    try:
        from pyroboframes.distributed import RemoteDataset
        ds = RemoteDataset.from_s3("s3://my-bucket/dataset", cache_dir="/tmp/test_cache")
        assert ds.remote_uri == "s3://my-bucket/dataset"
        assert "test_cache" in ds.cache_dir
    except ImportError:
        pytest.skip("fsspec not available")


def test_remote_dataset_import_error_without_fsspec(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _block_fsspec(name, *args, **kwargs):
        if name == "fsspec":
            raise ImportError("fsspec not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_fsspec)

    from pyroboframes.distributed import RemoteDataset
    with pytest.raises(ImportError, match="fsspec"):
        RemoteDataset("s3://bucket/path")


# ---------------------------------------------------------------------------
# RayDistributedLoader — requires ray; skip if not available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not pytest.importorskip("ray", reason="ray not installed"),
    reason="ray not installed",
)
def test_ray_distributed_loader_import(tmp_path):
    from pyroboframes.distributed import RayDistributedLoader  # noqa: F401


def test_ray_import_error_without_ray(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _block_ray(name, *args, **kwargs):
        if name == "ray":
            raise ImportError("ray not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_ray)

    from pyroboframes.distributed import RayDistributedLoader
    with pytest.raises(ImportError, match="ray"):
        # The import error is raised at __init__ time.
        RayDistributedLoader.__new__(RayDistributedLoader).__init__(
            "/nonexistent", 1, 0, 1
        )
