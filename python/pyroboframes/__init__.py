"""PyRoboFrames — zero-copy, hardware-accelerated robot-learning dataloader for Apple Silicon.

The heavy lifting (dataset reading, VideoToolbox decode, zero-copy windowing) lives in the
compiled Rust extension ``pyroboframes._core``. This package is the ergonomic Python surface:
the ``RoboFrameDataset`` / ``Loader`` API and the MLX / NumPy / PyTorch adapters.

The public API is still being built (see the project ROADMAP); today the package exposes the
engine version so wheels and the build can be smoke-tested end to end.
"""

from __future__ import annotations

from . import _core, backend, transforms
from ._core import Loader, PointCloudPy, RoboFrameDataset, convert_mcap, convert_ros2_bag
from .augmentation import (
    AugmentationPipeline,
    RandomBrightness,
    RandomCrop,
    RandomFlip,
    RandomNoise,
    RandomRotate,
)
from .backend import available_backends, default_framework, resolve_device, to_backend
from .compression import CompressionPipeline, DeltaEncoder, SparseArray
from .dataframe import AlignedFrame, RoboticsDataFrame, TopicFrame
from .dataloader import DataLoader
from .distributed import DistributedLoader, DistributedSampler
from .filtering import EpisodeFilter, EpisodeFilterBuilder
from .hub import download_lerobot_dataset
from .lazy_parquet import LazyDataFrameShards, LazyParquetReader
from .lerobot import write_lerobot_dataset
from .masking import MaskedDataFrame, SensorHealthMonitor, interpolate_missing
from .quality import EpisodeScorer, quality_percentile_filter
from .streaming import KafkaStreamer, MQTTStreamer, StreamingRoboticsDataset
from .tensorflow_support import KerasDataAdapter, create_keras_model_for_robotics, to_tf_dataset
from .versioning import DatasetManifest, DatasetVersion

# Public alias for the point cloud class
PointCloud = PointCloudPy

__all__ = [
    "__version__",
    "engine_version",
    "RoboFrameDataset",
    "Loader",
    "PointCloud",
    "DataLoader",
    "DistributedLoader",
    "DistributedSampler",
    "convert_mcap",
    "convert_ros2_bag",
    "RoboticsDataFrame",
    "TopicFrame",
    "AlignedFrame",
    "write_lerobot_dataset",
    "download_lerobot_dataset",
    "LazyParquetReader",
    "LazyDataFrameShards",
    "EpisodeScorer",
    "quality_percentile_filter",
    "EpisodeFilter",
    "EpisodeFilterBuilder",
    "DatasetVersion",
    "DatasetManifest",
    "DeltaEncoder",
    "SparseArray",
    "CompressionPipeline",
    "MaskedDataFrame",
    "SensorHealthMonitor",
    "interpolate_missing",
    "AugmentationPipeline",
    "RandomRotate",
    "RandomBrightness",
    "RandomNoise",
    "RandomCrop",
    "RandomFlip",
    "to_tf_dataset",
    "KerasDataAdapter",
    "create_keras_model_for_robotics",
    "MQTTStreamer",
    "KafkaStreamer",
    "StreamingRoboticsDataset",
    "transforms",
    "backend",
    "resolve_device",
    "available_backends",
    "default_framework",
    "to_backend",
]

__version__: str = _core.__version__


def engine_version() -> str:
    """Return the version of the underlying Rust engine."""
    return _core.engine_version()
