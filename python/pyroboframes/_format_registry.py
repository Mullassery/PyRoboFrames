"""
Format registry and loader for multiple robot learning dataset formats.

Supports:
- LeRobot (HuggingFace JSONL format)
- RLDS (TF Records, Open X-Embodiment)
- HDF5 (hierarchical data)
- Custom formats via plugin system

This CRITICAL feature unblocks teams using non-LeRobot datasets.
"""

from typing import Dict, List, Optional, Callable, Any, Type
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatasetFormat(Enum):
    """Supported dataset formats."""
    LEROBOT = "lerobot"
    RLDS = "rlds"
    HDF5 = "hdf5"
    NETCDF = "netcdf"
    CUSTOM = "custom"


@dataclass
class FormatSpec:
    """Specification for a dataset format."""
    name: str
    format_enum: DatasetFormat
    file_extensions: List[str]
    loader_class: Type
    supports_streaming: bool
    supports_random_access: bool
    description: str


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, path: str):
        self.path = Path(path)

    def load_episode(self, episode_id: int) -> Dict[str, Any]:
        """Load a single episode."""
        raise NotImplementedError

    def load_frame(self, episode_id: int, frame_idx: int) -> Dict[str, Any]:
        """Load a single frame."""
        raise NotImplementedError

    def get_episode_count(self) -> int:
        """Get total number of episodes."""
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        raise NotImplementedError


class LeRobotLoader(DatasetLoader):
    """Loader for LeRobot JSONL format."""

    def load_episode(self, episode_id: int) -> Dict[str, Any]:
        """Load LeRobot episode."""
        # Implementation delegates to existing lerobot.py
        from .lerobot import load_episode as le_load
        return le_load(str(self.path), episode_id)

    def load_frame(self, episode_id: int, frame_idx: int) -> Dict[str, Any]:
        """Load single frame from LeRobot episode."""
        from .lerobot import load_frame as le_load_frame
        return le_load_frame(str(self.path), episode_id, frame_idx)

    def get_episode_count(self) -> int:
        """Get LeRobot episode count."""
        from .lerobot import get_episode_count
        return get_episode_count(str(self.path))

    def get_metadata(self) -> Dict[str, Any]:
        """Get LeRobot metadata."""
        return {
            "format": "lerobot",
            "path": str(self.path),
            "description": "LeRobot JSONL format"
        }


class RLDSLoader(DatasetLoader):
    """Loader for RLDS (TF Records) format."""

    def load_episode(self, episode_id: int) -> Dict[str, Any]:
        """Load RLDS episode from TF Records."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("RLDS support requires tensorflow. Install: pip install tensorflow")

        # Read TF Record dataset
        raw_dataset = tf.data.TFRecordDataset(str(self.path / f"episode_{episode_id:06d}.tfrecord"))
        episode_data = {"frames": []}

        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            frame = self._parse_tf_example(example)
            episode_data["frames"].append(frame)

        episode_data["episode_id"] = episode_id
        episode_data["n_frames"] = len(episode_data["frames"])
        return episode_data

    def load_frame(self, episode_id: int, frame_idx: int) -> Dict[str, Any]:
        """Load single frame from RLDS."""
        episode = self.load_episode(episode_id)
        if frame_idx >= len(episode["frames"]):
            raise IndexError(f"Frame {frame_idx} out of range (max {len(episode['frames'])-1})")
        return episode["frames"][frame_idx]

    def get_episode_count(self) -> int:
        """Count RLDS episodes by finding TF Record files."""
        tfrecord_files = list(self.path.glob("episode_*.tfrecord"))
        return len(tfrecord_files)

    def get_metadata(self) -> Dict[str, Any]:
        """Get RLDS metadata."""
        return {
            "format": "rlds",
            "path": str(self.path),
            "description": "RLDS/Open X-Embodiment TF Records format",
            "n_episodes": self.get_episode_count()
        }

    @staticmethod
    def _parse_tf_example(example: Any) -> Dict[str, Any]:
        """Parse TensorFlow Example proto."""
        feature_dict = example.features.feature
        parsed = {}
        for key, feature in feature_dict.items():
            if feature.HasField("int64_list"):
                parsed[key] = feature.int64_list.value
            elif feature.HasField("float_list"):
                parsed[key] = feature.float_list.value
            elif feature.HasField("bytes_list"):
                parsed[key] = feature.bytes_list.value
        return parsed


class HDF5Loader(DatasetLoader):
    """Loader for HDF5 hierarchical format."""

    def load_episode(self, episode_id: int) -> Dict[str, Any]:
        """Load HDF5 episode."""
        try:
            import h5py
        except ImportError:
            raise ImportError("HDF5 support requires h5py. Install: pip install h5py")

        with h5py.File(self.path, "r") as f:
            episode_key = f"episode_{episode_id:06d}"
            if episode_key not in f:
                raise KeyError(f"Episode {episode_key} not found in HDF5")

            episode_group = f[episode_key]
            episode_data = {}

            # Recursively load all datasets
            def load_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    episode_data[name] = obj[()].tolist()

            episode_group.visititems(load_dataset)
            episode_data["episode_id"] = episode_id
            return episode_data

    def load_frame(self, episode_id: int, frame_idx: int) -> Dict[str, Any]:
        """Load single frame from HDF5."""
        try:
            import h5py
        except ImportError:
            raise ImportError("HDF5 support requires h5py. Install: pip install h5py")

        with h5py.File(self.path, "r") as f:
            episode_key = f"episode_{episode_id:06d}"
            frame_path = f"{episode_key}/frame_{frame_idx:06d}"

            if frame_path not in f:
                raise KeyError(f"Frame {frame_path} not found in HDF5")

            frame_group = f[frame_path]
            frame_data = {}

            def load_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    frame_data[name] = obj[()].tolist()

            frame_group.visititems(load_dataset)
            return frame_data

    def get_episode_count(self) -> int:
        """Count HDF5 episodes."""
        try:
            import h5py
        except ImportError:
            raise ImportError("HDF5 support requires h5py. Install: pip install h5py")

        with h5py.File(self.path, "r") as f:
            episodes = [k for k in f.keys() if k.startswith("episode_")]
            return len(episodes)

    def get_metadata(self) -> Dict[str, Any]:
        """Get HDF5 metadata."""
        return {
            "format": "hdf5",
            "path": str(self.path),
            "description": "HDF5 hierarchical format",
            "n_episodes": self.get_episode_count()
        }


class FormatRegistry:
    """
    Registry for dataset formats and their loaders.

    Allows plugging in custom format loaders.
    """

    def __init__(self):
        self.formats: Dict[DatasetFormat, FormatSpec] = {}
        self._register_builtin_formats()

    def _register_builtin_formats(self):
        """Register built-in formats."""
        self.register(
            DatasetFormat.LEROBOT,
            FormatSpec(
                name="LeRobot",
                format_enum=DatasetFormat.LEROBOT,
                file_extensions=[".jsonl", ".parquet"],
                loader_class=LeRobotLoader,
                supports_streaming=True,
                supports_random_access=False,
                description="LeRobot JSONL/Parquet format from HuggingFace"
            )
        )

        self.register(
            DatasetFormat.RLDS,
            FormatSpec(
                name="RLDS",
                format_enum=DatasetFormat.RLDS,
                file_extensions=[".tfrecord"],
                loader_class=RLDSLoader,
                supports_streaming=True,
                supports_random_access=False,
                description="Open X-Embodiment RLDS TF Records format"
            )
        )

        self.register(
            DatasetFormat.HDF5,
            FormatSpec(
                name="HDF5",
                format_enum=DatasetFormat.HDF5,
                file_extensions=[".h5", ".hdf5"],
                loader_class=HDF5Loader,
                supports_streaming=False,
                supports_random_access=True,
                description="Hierarchical HDF5 format"
            )
        )

    def register(self, format_enum: DatasetFormat, spec: FormatSpec):
        """Register a format loader."""
        self.formats[format_enum] = spec
        logger.info(f"Registered format: {spec.name} ({', '.join(spec.file_extensions)})")

    def register_custom(self, name: str, file_extensions: List[str], loader_class: Type,
                       description: str = ""):
        """Register a custom format loader."""
        spec = FormatSpec(
            name=name,
            format_enum=DatasetFormat.CUSTOM,
            file_extensions=file_extensions,
            loader_class=loader_class,
            supports_streaming=False,
            supports_random_access=False,
            description=description
        )
        self.register(DatasetFormat.CUSTOM, spec)

    def get_loader(self, dataset_path: str, format_hint: Optional[str] = None) -> DatasetLoader:
        """
        Get appropriate loader for dataset path.

        Args:
            dataset_path: Path to dataset
            format_hint: Optional format hint (LeRobot, RLDS, HDF5, etc.)

        Returns:
            Instantiated loader
        """
        path = Path(dataset_path)

        # Try format hint first
        if format_hint:
            for fmt, spec in self.formats.items():
                if spec.name.lower() == format_hint.lower():
                    logger.info(f"Using {spec.name} loader (hint)")
                    return spec.loader_class(dataset_path)

        # Auto-detect by file extension
        for fmt, spec in self.formats.items():
            for ext in spec.file_extensions:
                if path.suffix == ext or any(p.suffix == ext for p in path.glob("*")):
                    logger.info(f"Auto-detected {spec.name} format from {path.suffix}")
                    return spec.loader_class(dataset_path)

        raise ValueError(
            f"Cannot determine dataset format for {dataset_path}. "
            f"Provide format_hint or use supported extensions: {self._all_extensions()}"
        )

    def list_formats(self) -> Dict[str, FormatSpec]:
        """List all registered formats."""
        return {spec.name: spec for spec in self.formats.values()}

    def _all_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        exts = set()
        for spec in self.formats.values():
            exts.update(spec.file_extensions)
        return sorted(list(exts))


# Global registry instance
_registry: Optional[FormatRegistry] = None


def get_registry() -> FormatRegistry:
    """Get or create global format registry."""
    global _registry
    if _registry is None:
        _registry = FormatRegistry()
    return _registry


def load_dataset(dataset_path: str, format_hint: Optional[str] = None) -> DatasetLoader:
    """
    Load dataset with automatic or hinted format detection.

    Args:
        dataset_path: Path to dataset
        format_hint: Optional format (LeRobot, RLDS, HDF5)

    Returns:
        Loader instance ready to load episodes/frames

    Example:
        loader = load_dataset("/path/to/dataset", format_hint="HDF5")
        episode = loader.load_episode(0)
    """
    registry = get_registry()
    return registry.get_loader(dataset_path, format_hint)
