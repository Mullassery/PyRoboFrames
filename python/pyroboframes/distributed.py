"""Distributed data loading for multi-GPU and multi-machine training.

Supports PyTorch distributed sampler for synchronized loading across workers.

```python
import torch.distributed as dist
from pyroboframes.distributed import DistributedLoader

# Initialize distributed backend (via torch.distributed.launch)
dist.init_process_group("nccl")

# Create distributed loader
ds = prf.RoboFrameDataset.from_path("…")
loader = DistributedLoader(
    ds,
    batch_size=32,
    world_size=dist.get_world_size(),
    rank=dist.get_rank(),
    num_workers=4,
)

# Each worker loads different episodes; synchronized across ranks
for batch in loader:
    # All workers yield batches in lockstep
    ...
```
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from ._core import RoboFrameDataset, Loader


def _uri_to_safe_name(uri: str) -> str:
    """Convert a remote URI to a filesystem-safe cache directory name."""
    return uri.replace("://", "_").replace("/", "_").strip("_")


class DistributedSampler:
    """Episode sampler for distributed training.

    Divides episodes evenly across workers (no overlap) and supports
    synchronized shuffling and epoch-based re-seeding.
    """

    def __init__(
        self,
        num_episodes: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """Initialize sampler.

        Args:
            num_episodes: Total number of episodes
            num_replicas: Number of distributed processes (world_size)
            rank: Rank of current process (0 to num_replicas-1)
            shuffle: Whether to shuffle episodes
            seed: Random seed (for reproducibility)
            drop_last: Drop last incomplete batch
        """
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, should be in the interval [0, {num_replicas})"
            )

        self.num_episodes = num_episodes
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate how many episodes per worker
        if self.drop_last and self.num_episodes % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (self.num_episodes - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(self.num_episodes / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """Yield episode indices for this worker."""
        import numpy as np

        if self.shuffle:
            # Seeded shuffle for reproducibility
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = np.arange(self.num_episodes)
            rng.shuffle(indices)
            indices = indices.tolist()
        else:
            indices = list(range(self.num_episodes))

        # Pad or trim to make divisible by num_replicas
        if len(indices) < self.total_size:
            # Pad with random indices
            padding = self.total_size - len(indices)
            indices += [indices[i % len(indices)] for i in range(padding)]
        elif len(indices) > self.total_size:
            indices = indices[: self.total_size]

        # Partition by rank
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]

        return iter(indices)

    def __len__(self) -> int:
        """Number of episodes for this worker."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible shuffling.

        Call this at the start of each epoch to ensure different shuffle
        order across epochs while maintaining reproducibility.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch


class DistributedLoader:
    """Wrapper around RoboFrameDataset for distributed training.

    Automatically handles episode partitioning and synchronized loading
    across multiple workers/machines.
    """

    def __init__(
        self,
        dataset: RoboFrameDataset,
        batch_size: int,
        world_size: int,
        rank: int,
        num_workers: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        **loader_kwargs: Any,
    ):
        """Initialize distributed loader.

        Args:
            dataset: RoboFrameDataset to load from
            batch_size: Batch size per worker
            world_size: Total number of distributed processes
            rank: Rank of current process
            num_workers: Number of prefetch workers
            shuffle: Whether to shuffle episodes
            seed: Random seed
            drop_last: Drop last incomplete batch
            **loader_kwargs: Additional kwargs passed to dataset.loader()
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.loader_kwargs = loader_kwargs
        self.epoch = 0

        # Create sampler
        self.sampler = DistributedSampler(
            num_episodes=dataset.num_episodes(),
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def __iter__(self):
        """Iterate batches for this worker."""
        # Set sampler epoch for reproducible shuffling
        self.sampler.set_epoch(self.epoch)

        # Get episodes for this worker
        episodes = list(self.sampler)

        # Create loader with these episodes
        loader = self.dataset.loader(
            episodes=episodes,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **self.loader_kwargs,
        )

        return iter(loader)

    def __len__(self) -> int:
        """Number of batches for this worker."""
        num_episodes = len(self.sampler)
        # Approximate batches (true count depends on episode length)
        avg_episode_length = self.dataset.total_frames() / max(1, self.dataset.num_episodes())
        total_frames = num_episodes * avg_episode_length
        return int(math.ceil(total_frames / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible shuffling.

        Call at the start of each epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
        self.sampler.set_epoch(epoch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def shard_episodes(total_episodes: int, world_size: int, rank: int) -> list[int]:
    """Return episode indices assigned to ``rank`` using round-robin sharding.

    Args:
        total_episodes: Total number of episodes in the dataset.
        world_size: Number of distributed processes.
        rank: Rank of the current process (0 to world_size-1).

    Returns:
        Sorted list of episode indices for this rank.

    Example:
        >>> shard_episodes(10, 3, 0)
        [0, 3, 6, 9]
        >>> shard_episodes(10, 3, 1)
        [1, 4, 7]
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return list(range(rank, total_episodes, world_size))


# ---------------------------------------------------------------------------
# S3 / GCS Remote Dataset
# ---------------------------------------------------------------------------


class RemoteDataset:
    """RoboFrameDataset wrapper that streams data from S3 or GCS.

    Files are downloaded to a local ``cache_dir`` on demand and then read by the
    standard Rust-backed loader. Episode prefetch runs in background threads to
    overlap network I/O with training.

    Requires ``fsspec``, and either ``s3fs`` (for S3) or ``gcsfs`` (for GCS).

    Args:
        remote_uri: S3 URI (``s3://bucket/prefix``) or GCS URI (``gs://bucket/prefix``).
        cache_dir: Local directory to download files into. Uses ``~/.cache/pyroboframes``
            if not specified.
        storage_options: Extra kwargs passed to ``fsspec.filesystem()``.
    """

    def __init__(
        self,
        remote_uri: str,
        cache_dir: Optional[str] = None,
        storage_options: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            import fsspec  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "fsspec is required for remote datasets: pip install fsspec s3fs gcsfs"
            ) from exc
        self.remote_uri = remote_uri.rstrip("/")
        self.cache_dir = cache_dir or os.path.expanduser(
            f"~/.cache/pyroboframes/{_uri_to_safe_name(remote_uri)}"
        )
        self.storage_options = storage_options or {}
        self._dataset: Optional[Any] = None

    @classmethod
    def from_s3(
        cls,
        s3_uri: str,
        *,
        cache_dir: Optional[str] = None,
        aws_profile: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> "RemoteDataset":
        """Create a RemoteDataset backed by Amazon S3.

        Args:
            s3_uri: S3 URI, e.g. ``s3://my-bucket/datasets/robot_lerobot``.
            cache_dir: Local cache directory.
            aws_profile: AWS credentials profile name.
            region_name: AWS region override.
        """
        opts: dict[str, Any] = {}
        if aws_profile:
            opts["profile"] = aws_profile
        if region_name:
            opts["client_kwargs"] = {"region_name": region_name}
        return cls(s3_uri, cache_dir=cache_dir, storage_options=opts)

    @classmethod
    def from_gcs(
        cls,
        gcs_uri: str,
        *,
        cache_dir: Optional[str] = None,
        project: Optional[str] = None,
        token: Optional[str] = None,
    ) -> "RemoteDataset":
        """Create a RemoteDataset backed by Google Cloud Storage.

        Args:
            gcs_uri: GCS URI, e.g. ``gs://my-bucket/datasets/robot_lerobot``.
            cache_dir: Local cache directory.
            project: GCP project ID.
            token: GCS credentials token path or ``"anon"`` for public buckets.
        """
        opts: dict[str, Any] = {}
        if project:
            opts["project"] = project
        if token:
            opts["token"] = token
        return cls(gcs_uri, cache_dir=cache_dir, storage_options=opts)

    def _sync(self) -> None:
        """Download the dataset metadata and index to the local cache directory."""
        import fsspec

        protocol = self.remote_uri.split("://")[0]
        fs = fsspec.filesystem(protocol, **self.storage_options)
        remote_path = self.remote_uri.split("://", 1)[1]

        # Download meta/ and episodes index first (small files, required to open dataset).
        for subdir in ("meta",):
            remote_subdir = f"{remote_path}/{subdir}"
            local_subdir = os.path.join(self.cache_dir, subdir)
            if fs.exists(remote_subdir):
                fs.get(remote_subdir, local_subdir, recursive=True)

    def open(self) -> Any:
        """Download metadata and return a :class:`~pyroboframes._core.RoboFrameDataset`.

        The data shards are downloaded lazily when episodes are prefetched or a loader
        is created with ``prefetch_episodes()`` called first.
        """
        from ._core import RoboFrameDataset

        os.makedirs(self.cache_dir, exist_ok=True)
        self._sync()
        self._dataset = RoboFrameDataset.from_path(self.cache_dir)
        return self._dataset

    def prefetch_episodes(self, episode_indices: list[int]) -> None:
        """Download data shards for the specified episodes to the local cache.

        Downloads run in background threads so this call returns immediately.
        Call :meth:`open` first to retrieve the dataset object.

        Args:
            episode_indices: Episode indices to download.
        """
        import threading

        import fsspec

        def _download() -> None:
            protocol = self.remote_uri.split("://")[0]
            fs = fsspec.filesystem(protocol, **self.storage_options)
            remote_path = self.remote_uri.split("://", 1)[1]
            for ep_idx in episode_indices:
                # Download data shard for this episode (chunk-000/file-NNN.parquet).
                chunk = ep_idx // 1000
                file_idx = ep_idx % 1000
                for subpath in (
                    f"data/chunk-{chunk:03d}/file-{file_idx:03d}.parquet",
                    f"meta/episodes/chunk-000/file-000.parquet",
                ):
                    remote_file = f"{remote_path}/{subpath}"
                    local_file = os.path.join(self.cache_dir, subpath)
                    if not os.path.exists(local_file) and fs.exists(remote_file):
                        os.makedirs(os.path.dirname(local_file), exist_ok=True)
                        fs.get(remote_file, local_file)

        t = threading.Thread(target=_download, daemon=True)
        t.start()

    def loader(self, **kwargs: Any) -> Any:
        """Open the dataset and return a standard :class:`~pyroboframes.dataloader.DataLoader`.

        Args:
            **kwargs: Passed directly to :meth:`~pyroboframes._core.RoboFrameDataset.loader`.
        """
        if self._dataset is None:
            self.open()
        return self._dataset.loader(**kwargs)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Ray Distributed Loader
# ---------------------------------------------------------------------------


class RayDistributedLoader:
    """Distributed loader that shards episodes across Ray workers.

    Each Ray worker holds a :class:`~pyroboframes._core.RoboFrameDataset` and a
    :class:`~pyroboframes.dataloader.DataLoader` for its assigned episodes.
    Workers are assigned episodes via round-robin sharding using :func:`shard_episodes`.

    Requires ``ray>=2.0``: ``pip install ray``.

    Args:
        dataset_path: Local path to the LeRobot dataset directory.
        num_workers: Number of Ray workers (actors) to spawn.
        rank: Rank of this worker within the pool (0 to num_workers-1).
        world_size: Total number of workers.
        **loader_kwargs: Passed to :meth:`~pyroboframes._core.RoboFrameDataset.loader`.
    """

    def __init__(
        self,
        dataset_path: str,
        num_workers: int,
        rank: int,
        world_size: int,
        **loader_kwargs: Any,
    ) -> None:
        try:
            import ray  # noqa: F401
        except ImportError as exc:
            raise ImportError("ray is required: pip install ray") from exc

        from ._core import RoboFrameDataset

        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.loader_kwargs = loader_kwargs

        self._dataset = RoboFrameDataset.from_path(dataset_path)
        self._episodes = shard_episodes(self._dataset.num_episodes(), world_size, rank)
        self._loader_kwargs = {**loader_kwargs, "episodes": self._episodes, "shuffle": False}

    @staticmethod
    def from_ray_actor(dataset_path: str, **loader_kwargs: Any) -> "RayDistributedLoader":
        """Construct inside a Ray actor using ``ray.get_runtime_context()``.

        Rank and world_size are read from the Ray runtime context.

        Args:
            dataset_path: Path to the dataset.
            **loader_kwargs: Passed to the loader.
        """
        import ray

        ctx = ray.get_runtime_context()
        rank = ctx.get_actor_id() or 0
        world_size = loader_kwargs.pop("world_size", 1)
        num_workers = loader_kwargs.pop("num_workers", 1)
        return RayDistributedLoader(
            dataset_path, num_workers, rank, world_size, **loader_kwargs
        )

    def __iter__(self) -> Iterator[Any]:
        loader = self._dataset.loader(**self._loader_kwargs)
        return iter(loader)

    def __len__(self) -> int:
        avg = self._dataset.total_frames() / max(1, self._dataset.num_episodes())
        total = len(self._episodes) * avg
        batch_size = self.loader_kwargs.get("batch_size", 32)
        return max(1, int(math.ceil(total / batch_size)))

    @property
    def assigned_episodes(self) -> list[int]:
        """Episode indices assigned to this worker."""
        return self._episodes
