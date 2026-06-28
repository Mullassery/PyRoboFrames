"""HDF5 dataset reader and converter for robot learning data.

Reads HDF5 files from common robot learning formats (ROBOMIMIC, ACT, etc.) and converts
them to LeRobot-compatible Parquet layout or a RoboticsDataFrame for quick analysis.

```python
from pyroboframes.hdf5 import HDF5Dataset, convert_hdf5

# Inspect structure
ds = HDF5Dataset.from_path("robomimic_data.hdf5")
print(ds.inspect())

# Convert to LeRobot format
convert_hdf5("robomimic_data.hdf5", "/out/lerobot_dataset")

# Or load as RoboticsDataFrame for quick analysis
df = ds.to_robotics_dataframe()
```
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ConversionReport:
    """Summary of an HDF5 → LeRobot conversion."""

    episodes_converted: int = 0
    features_extracted: list[str] = field(default_factory=list)
    skipped_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"HDF5 conversion: {self.episodes_converted} episodes",
            f"  Features: {self.features_extracted}",
        ]
        if self.skipped_keys:
            lines.append(f"  Skipped: {self.skipped_keys}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class HDF5Dataset:
    """Reader for HDF5-based robot learning datasets (ROBOMIMIC, ACT, custom).

    Supports layouts where episodes are stored as top-level groups named
    ``demo_0``, ``episode_0``, ``traj_0``, or under a configurable ``episode_key`` prefix.

    Args:
        path: Path to the ``.hdf5`` / ``.h5`` file.
        episode_key: Group name prefix for episodes (default: auto-detect).
    """

    def __init__(self, path: str, episode_key: Optional[str] = None) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "h5py is required to read HDF5 files: pip install h5py"
            ) from exc
        self.path = path
        self._episode_key = episode_key
        self._episode_groups: Optional[list[str]] = None

    @classmethod
    def from_path(cls, path: str, episode_key: Optional[str] = None) -> "HDF5Dataset":
        """Open an HDF5 dataset.

        Args:
            path: Path to the ``.hdf5`` / ``.h5`` file.
            episode_key: Group name prefix for episodes. Auto-detected if None.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path!r}")
        return cls(path, episode_key=episode_key)

    def inspect(self) -> dict[str, Any]:
        """Return a tree of groups, datasets, shapes, and dtypes."""
        import h5py

        def _walk(obj: Any) -> Any:
            if isinstance(obj, h5py.Dataset):
                return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            return {k: _walk(v) for k, v in obj.items()}

        with h5py.File(self.path, "r") as f:
            return {
                "path": self.path,
                "episodes": len(self._get_episode_groups(f)),
                "structure": _walk(f),
            }

    def episode_count(self) -> int:
        """Return the number of episodes in the file."""
        import h5py

        with h5py.File(self.path, "r") as f:
            return len(self._get_episode_groups(f))

    def to_robotics_dataframe(self) -> Any:
        """Load all episodes as a :class:`RoboticsDataFrame` for quick analysis.

        Returns a :class:`~pyroboframes.dataframe.RoboticsDataFrame` with one
        topic per feature key, concatenated across all episodes.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        import tempfile

        from .dataframe import RoboticsDataFrame

        with tempfile.TemporaryDirectory() as tmp:
            report = _write_lerobot_layout(self.path, tmp, self._episode_key)
            return RoboticsDataFrame.from_converted(tmp)

    def to_lerobot(self, out_dir: str) -> ConversionReport:
        """Convert to LeRobot v3.0 layout in ``out_dir``.

        Args:
            out_dir: Output directory (created if it does not exist).

        Returns:
            A :class:`ConversionReport` describing what was converted.
        """
        return _write_lerobot_layout(self.path, out_dir, self._episode_key)

    def _get_episode_groups(self, f: Any) -> list[str]:
        """Return sorted episode group names from an open h5py.File."""
        if self._episode_key:
            groups = sorted(k for k in f.keys() if k.startswith(self._episode_key))
        else:
            # Auto-detect: prefer demo_*, episode_*, traj_* prefixes; fall back to any group.
            for prefix in ("demo", "episode", "traj", "data"):
                groups = sorted(k for k in f.keys()
                                if k.startswith(prefix) and isinstance(f[k], type(f)))
                if groups:
                    break
            else:
                import h5py
                groups = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))
        return groups


def _flatten_group(group: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Recursively flatten an h5py.Group into {key: array} dict."""
    import h5py

    result: dict[str, np.ndarray] = {}
    for key, val in group.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, h5py.Dataset):
            arr = val[()]
            if arr.dtype.kind in ("f", "i", "u", "b"):
                result[full_key] = arr.astype(np.float32) if arr.dtype.kind != "b" else arr
        elif isinstance(val, h5py.Group):
            result.update(_flatten_group(val, full_key))
    return result


def _write_lerobot_layout(
    hdf5_path: str,
    out_dir: str,
    episode_key: Optional[str],
) -> ConversionReport:
    """Convert an HDF5 file to LeRobot v3.0 Parquet layout."""
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    report = ConversionReport()
    ds = HDF5Dataset(hdf5_path, episode_key=episode_key)
    os.makedirs(os.path.join(out_dir, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta", "episodes", "chunk-000"), exist_ok=True)

    all_arrays: dict[str, list[np.ndarray]] = {}
    episode_lengths: list[int] = []

    with h5py.File(hdf5_path, "r") as f:
        groups = ds._get_episode_groups(f)
        for group_name in groups:
            ep_data = _flatten_group(f[group_name])
            if not ep_data:
                report.warnings.append(f"Group {group_name!r} had no numeric datasets; skipped.")
                continue
            # All arrays in this episode must share the same leading dimension (T).
            lengths = {k: arr.shape[0] for k, arr in ep_data.items() if arr.ndim >= 1}
            if not lengths:
                continue
            T = max(lengths.values())
            episode_lengths.append(T)
            for key, arr in ep_data.items():
                if arr.ndim == 0:
                    arr = np.expand_dims(arr, 0)
                # Pad/trim to T if needed (rare mismatches in some datasets).
                if arr.shape[0] != T:
                    report.warnings.append(
                        f"Episode {group_name!r} key {key!r}: length {arr.shape[0]} != {T}; trimming."
                    )
                    arr = arr[:T]
                # Ensure 2D: [T, D]
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                elif arr.ndim > 2:
                    arr = arr.reshape(T, -1)
                all_arrays.setdefault(key, []).append(arr.astype(np.float32))

    if not episode_lengths:
        raise ValueError(f"No valid episode groups found in {hdf5_path!r}")

    # Stack all episodes per feature.
    stacked: dict[str, np.ndarray] = {}
    for key, arrays in all_arrays.items():
        try:
            stacked[key] = np.concatenate(arrays, axis=0)
            report.features_extracted.append(key)
        except ValueError as exc:
            report.skipped_keys.append(key)
            report.warnings.append(f"Could not stack feature {key!r}: {exc}")

    # Write Parquet.
    from .lerobot import write_lerobot_dataset
    write_lerobot_dataset(out_dir, stacked, episode_lengths)
    report.episodes_converted = len(episode_lengths)
    return report


def convert_hdf5(
    path: str,
    out_dir: str,
    *,
    episode_key: Optional[str] = None,
    obs_key: str = "obs",
    action_key: str = "actions",
) -> ConversionReport:
    """Convert an HDF5 robot learning dataset to LeRobot v3.0 Parquet layout.

    Args:
        path: Path to the ``.hdf5`` / ``.h5`` input file.
        out_dir: Output directory (created if needed).
        episode_key: Group name prefix for episodes (auto-detected if None).
        obs_key: Group name for observations within each episode (informational).
        action_key: Dataset name for actions within each episode (informational).

    Returns:
        :class:`ConversionReport` summarising what was converted.
    """
    return _write_lerobot_layout(path, out_dir, episode_key)
