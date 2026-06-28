"""RLDS (Robot Learning Dataset Sharing) reader for Open X-Embodiment datasets.

Reads RLDS/TFRecord datasets from HuggingFace / tensorflow_datasets and converts them
to LeRobot v3.0 Parquet layout compatible with PyRoboFrames.

```python
from pyroboframes.rlds import RLDSDataset, convert_rlds

# From tensorflow_datasets
ds = RLDSDataset.from_tfds("fractal20220817_data", split="train")
convert_rlds("fractal20220817_data", "/out/lerobot_dataset")

# From a local RLDS directory
ds = RLDSDataset.from_directory("/path/to/rlds")
df = ds.to_robotics_dataframe()
```
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np


@dataclass
class ConversionReport:
    """Summary of an RLDS → LeRobot conversion."""

    episodes_converted: int = 0
    features_extracted: list[str] = field(default_factory=list)
    skipped_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"RLDS conversion: {self.episodes_converted} episodes",
            f"  Features: {self.features_extracted}",
        ]
        if self.skipped_keys:
            lines.append(f"  Skipped: {self.skipped_keys}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class RLDSDataset:
    """Reader for RLDS (Robot Learning Dataset Sharing) / Open X-Embodiment datasets.

    RLDS schema mapping:
    - ``steps[i].observation.*`` → feature columns (prefixed with ``observation.``)
    - ``steps[i].action`` → ``action`` column
    - ``steps[i].reward`` → ``reward`` column (if present)
    - Episode-level metadata is stored in stats

    Args:
        _dataset: Internal TFDS dataset object.
        _name: Dataset name for reporting.
    """

    def __init__(self, _dataset: Any, _name: str = "rlds") -> None:
        self._dataset = _dataset
        self._name = _name

    @classmethod
    def from_tfds(
        cls,
        name: str,
        split: str = "train",
        data_dir: Optional[str] = None,
    ) -> "RLDSDataset":
        """Load an RLDS dataset from tensorflow_datasets.

        Args:
            name: Dataset name (e.g. ``"fractal20220817_data"``).
            split: Dataset split (default: ``"train"``).
            data_dir: Local data directory override.

        Raises:
            ImportError: If ``tensorflow_datasets`` is not installed.
        """
        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "tensorflow_datasets is required: pip install tensorflow_datasets"
            ) from exc

        kwargs: dict[str, Any] = {"split": split}
        if data_dir:
            kwargs["data_dir"] = data_dir
        dataset = tfds.load(name, **kwargs)
        return cls(dataset, name)

    @classmethod
    def from_directory(cls, path: str, split: str = "train") -> "RLDSDataset":
        """Load an RLDS dataset from a local directory of TFRecord files.

        Args:
            path: Directory containing RLDS TFRecord files.
            split: Dataset split name.
        """
        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "tensorflow_datasets is required: pip install tensorflow_datasets"
            ) from exc

        builder = tfds.builder_from_directory(path)
        dataset = builder.as_dataset(split=split)
        return cls(dataset, os.path.basename(path))

    @property
    def feature_names(self) -> list[str]:
        """Return flat feature names from the RLDS step spec."""
        try:
            spec = self._dataset.element_spec
            step_spec = spec.get("steps", spec)
            return list(_flatten_spec(step_spec).keys())
        except Exception:
            return []

    def to_robotics_dataframe(self) -> Any:
        """Load as a :class:`~pyroboframes.dataframe.RoboticsDataFrame`."""
        import tempfile

        from .dataframe import RoboticsDataFrame

        with tempfile.TemporaryDirectory() as tmp:
            _write_lerobot_layout(self._dataset, tmp, self._name)
            return RoboticsDataFrame.from_converted(tmp)

    def to_lerobot(self, out_dir: str, video_codec: str = "h264") -> ConversionReport:
        """Convert to LeRobot v3.0 Parquet layout.

        Args:
            out_dir: Output directory.
            video_codec: Codec for image features (``"h264"`` default).
        """
        return _write_lerobot_layout(self._dataset, out_dir, self._name)


def _flatten_spec(spec: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a TFDS FeatureSpec into {key: spec} dict."""
    result: dict[str, Any] = {}
    try:
        for key, val in spec.items():
            full_key = f"{prefix}.{key}" if prefix else key
            try:
                for sub_k, sub_v in val.items():
                    result[f"{full_key}.{sub_k}"] = sub_v
            except (AttributeError, TypeError):
                result[full_key] = val
    except (AttributeError, TypeError):
        pass
    return result


def _flatten_step(step: dict[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a single RLDS step dict into {key: scalar_or_array}."""
    import tensorflow as tf

    result: dict[str, np.ndarray] = {}
    for key, val in step.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            result.update(_flatten_step(val, full_key))
        else:
            try:
                arr = np.array(val)
                if arr.dtype.kind in ("f", "i", "u", "b"):
                    result[full_key] = arr.astype(np.float32).reshape(-1)
                # Skip image arrays (uint8 high-dim) for now — video encoding is a future step.
            except Exception:
                pass
    return result


def _write_lerobot_layout(
    dataset: Any,
    out_dir: str,
    name: str,
) -> ConversionReport:
    """Iterate an RLDS dataset and write LeRobot v3.0 Parquet layout."""
    report = ConversionReport()
    all_arrays: dict[str, list[np.ndarray]] = {}
    episode_lengths: list[int] = []
    feature_set: set[str] = set()

    for episode in dataset:
        steps = episode.get("steps", episode)
        ep_data: dict[str, list[np.ndarray]] = {}
        T = 0

        try:
            steps_list = list(steps)
        except Exception as exc:
            report.warnings.append(f"Could not iterate episode steps: {exc}")
            continue

        for step in steps_list:
            flat = _flatten_step(step)
            for key, arr in flat.items():
                ep_data.setdefault(key, []).append(arr)
            T += 1

        if T == 0:
            continue

        episode_lengths.append(T)
        for key, arrays in ep_data.items():
            try:
                stacked = np.stack(arrays, axis=0)  # [T, D]
                all_arrays.setdefault(key, []).append(stacked)
                feature_set.add(key)
            except ValueError as exc:
                report.skipped_keys.append(key)
                report.warnings.append(f"Feature {key!r}: could not stack steps: {exc}")

    if not episode_lengths:
        raise ValueError(f"No valid episodes found in RLDS dataset {name!r}.")

    # Concatenate across episodes.
    stacked_all: dict[str, np.ndarray] = {}
    for key in feature_set:
        try:
            stacked_all[key] = np.concatenate(all_arrays[key], axis=0)
            report.features_extracted.append(key)
        except ValueError as exc:
            report.skipped_keys.append(key)
            report.warnings.append(f"Could not concatenate feature {key!r}: {exc}")

    if not stacked_all:
        raise ValueError("No features could be extracted from the RLDS dataset.")

    os.makedirs(out_dir, exist_ok=True)
    from .lerobot import write_lerobot_dataset
    write_lerobot_dataset(out_dir, stacked_all, episode_lengths)
    report.episodes_converted = len(episode_lengths)
    return report


def convert_rlds(
    name: str,
    out_dir: str,
    split: str = "train",
    data_dir: Optional[str] = None,
) -> ConversionReport:
    """Load an RLDS dataset from tensorflow_datasets and convert to LeRobot v3.0.

    Args:
        name: TFDS dataset name (e.g. ``"fractal20220817_data"``).
        out_dir: Output directory for the LeRobot dataset.
        split: Dataset split (default: ``"train"``).
        data_dir: Local TFDS data directory override.

    Returns:
        :class:`ConversionReport` summarising the conversion.
    """
    ds = RLDSDataset.from_tfds(name, split=split, data_dir=data_dir)
    return ds.to_lerobot(out_dir)
