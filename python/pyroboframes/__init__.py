"""PyRoboFrames v1.0 — Fast ML dataloader for robot learning on Apple Silicon.

Focus: LeRobot datasets, video decode (GPU-accelerated), temporal windows, multi-output formats.

The heavy lifting (dataset reading, VideoToolbox decode, zero-copy windowing) lives in the
compiled Rust extension ``pyroboframes._core``. This package is the ergonomic Python surface:
the ``RoboFrameDataset`` / ``ProprioceptiveLoader`` / ``DataLoader`` API and device adapters
(MLX, NumPy, PyTorch, JAX).

For autonomous driving perception and foundation models, see PyRoboVision:
https://github.com/Mullassery/PyRoboVision
"""

from __future__ import annotations

from . import _core, backend, depth_io, sensor_fusion, transforms
from ._core import (
    CameraCalibrationPy,
    CameraIntrinsicsPy,
    Loader,
    PointCloudPy,
    RoboFrameDataset,
    convert_mcap,
    convert_ros2_bag,
)
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
from .proprioceptive_loader import ProprioceptiveLoader, ProprioceptiveDataFrame
from .distributed import DistributedLoader, DistributedSampler, RemoteDataset, RayDistributedLoader, shard_episodes
from .episode_cache import EpisodeCache
from .filtering import EpisodeFilter, EpisodeFilterBuilder
from .hdf5 import HDF5Dataset, convert_hdf5
from .hub import download_lerobot_dataset
from .lazy_parquet import LazyDataFrameShards, LazyParquetReader
from .lerobot import encode_video_frames, write_lerobot_dataset
from .masking import MaskedDataFrame, SensorHealthMonitor, interpolate_missing
from .netcdf import NetCDFDataset, convert_netcdf
from .quality import CrossDatasetComparator, DatasetQualityProfile, EpisodeScorer, compare_datasets, quality_percentile_filter
from .rlds import RLDSDataset, convert_rlds
from .streaming import KafkaStreamer, MQTTStreamer, StreamingRoboticsDataset
from .tensorflow_support import KerasDataAdapter, create_keras_model_for_robotics, to_tf_dataset
from .validation import DatasetValidator, FullValidationReport, ValidationIssue
from .versioning import DatasetManifest, DatasetVersion

# CRITICAL: Multi-format dataset support (unblocks non-LeRobot users)
from ._format_registry import (
    FormatRegistry,
    DatasetFormat,
    FormatSpec,
    DatasetLoader,
    LeRobotLoader,
    RLDSLoader,
    HDF5Loader,
    get_registry,
    load_dataset,
)

# Public aliases for depth and calibration classes
PointCloud = PointCloudPy
CameraIntrinsics = CameraIntrinsicsPy
CameraCalibration = CameraCalibrationPy

__all__ = [
    "__version__",
    "engine_version",
    "RoboFrameDataset",
    "Loader",
    "PointCloud",
    "CameraIntrinsics",
    "CameraCalibration",
    "DataLoader",
    "ProprioceptiveLoader",
    "ProprioceptiveDataFrame",
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
    "depth_io",
    "sensor_fusion",
    "resolve_device",
    "available_backends",
    "default_framework",
    "to_backend",
    # New in v1.1
    "encode_video_frames",
    "DatasetValidator",
    "FullValidationReport",
    "ValidationIssue",
    "EpisodeCache",
    "DatasetQualityProfile",
    "CrossDatasetComparator",
    "compare_datasets",
    "HDF5Dataset",
    "convert_hdf5",
    "NetCDFDataset",
    "convert_netcdf",
    "RLDSDataset",
    "convert_rlds",
    "RemoteDataset",
    "RayDistributedLoader",
    "shard_episodes",
    # Multi-format support (v1.3.0+)
    "FormatRegistry",
    "DatasetFormat",
    "FormatSpec",
    "DatasetLoader",
    "LeRobotLoader",
    "RLDSLoader",
    "HDF5Loader",
    "get_registry",
    "load_dataset",
]

__version__: str = _core.__version__


def engine_version() -> str:
    """Return the version of the underlying Rust engine."""
    return _core.engine_version()
