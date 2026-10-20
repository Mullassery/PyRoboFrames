"""GR00T-compatible LeRobot v2.1 dataset reader and validator.

GR00T N1.7 consumes LeRobot v2.1 datasets with a ``meta/modality.json``
sidecar. This module validates that layout without importing GR00T, Torch,
or any GPU runtime, and exposes a ``DatasetLoader`` for tabular episodes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._format_registry import DatasetLoader
from .validation import FullValidationReport, ValidationIssue

GR00T_MIN_VERSION = "v2.1"
REQUIRED_META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
REQUIRED_FEATURES = ("observation.state", "action")
RESERVED_ANNOTATION_KEYS = {"task_index", "task", "episode_index", "frame_index"}


@dataclass
class Gr00tDatasetInfo:
    """Parsed GR00T/LeRobot v2.1 metadata."""

    path: Path
    info: dict[str, Any]
    episodes: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    modality: dict[str, Any] = field(default_factory=dict)

    @property
    def codebase_version(self) -> str:
        return str(self.info.get("codebase_version", ""))

    @property
    def chunks_size(self) -> int:
        return int(self.info.get("chunks_size", 1000))

    @property
    def total_episodes(self) -> int:
        return int(self.info.get("total_episodes", len(self.episodes)))

    @property
    def total_frames(self) -> int:
        return int(self.info.get("total_frames", 0))

    @property
    def fps(self) -> float:
        return float(self.info.get("fps", 0.0))


def _add(
    report: FullValidationReport,
    severity: str,
    category: str,
    message: str,
    episode: int | None = None,
) -> None:
    report.issues.append(
        ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            episode=episode,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def _parse_meta(path: Path, report: FullValidationReport) -> Gr00tDatasetInfo | None:
    meta_dir = path / "meta"
    if not path.is_dir():
        _add(report, "error", "metadata", f"dataset path is not a directory: {path}")
        return None
    if not meta_dir.is_dir():
        _add(report, "error", "metadata", f"missing meta directory: {meta_dir}")
        return None

    missing = [name for name in REQUIRED_META_FILES if not (meta_dir / name).is_file()]
    for name in missing:
        _add(report, "error", "metadata", f"missing meta/{name}")

    info: dict[str, Any] = {}
    episodes: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    modality: dict[str, Any] = {}

    if (meta_dir / "info.json").is_file():
        try:
            info = _read_json(meta_dir / "info.json")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _add(report, "error", "metadata", f"invalid meta/info.json: {exc}")
    if (meta_dir / "episodes.jsonl").is_file():
        try:
            episodes = _read_jsonl(meta_dir / "episodes.jsonl")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _add(report, "error", "metadata", f"invalid meta/episodes.jsonl: {exc}")
    if (meta_dir / "tasks.jsonl").is_file():
        try:
            tasks = _read_jsonl(meta_dir / "tasks.jsonl")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _add(report, "error", "metadata", f"invalid meta/tasks.jsonl: {exc}")
    if (meta_dir / "modality.json").is_file():
        try:
            modality = _read_json(meta_dir / "modality.json")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _add(report, "error", "metadata", f"invalid meta/modality.json: {exc}")

    if missing:
        return None
    return Gr00tDatasetInfo(
        path=path, info=info, episodes=episodes, tasks=tasks, modality=modality
    )


def _validate_info(dataset: Gr00tDatasetInfo, report: FullValidationReport) -> None:
    info = dataset.info
    version = dataset.codebase_version
    if not version:
        _add(report, "error", "metadata", "meta/info.json is missing codebase_version")
    elif not version.startswith(GR00T_MIN_VERSION):
        _add(
            report,
            "error",
            "metadata",
            f"GR00T requires LeRobot {GR00T_MIN_VERSION}, got {version!r}",
        )

    for key in (
        "robot_type",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "fps",
        "data_path",
        "features",
    ):
        if key not in info:
            _add(report, "error", "metadata", f"meta/info.json is missing {key!r}")

    features = info.get("features")
    if not isinstance(features, dict):
        _add(report, "error", "metadata", "features must be a JSON object")
        return

    for name in REQUIRED_FEATURES:
        if name not in features:
            _add(report, "error", "modality", f"features is missing {name!r}")

    if "video_path" not in info:
        video_entries = dataset.modality.get("video")
        if isinstance(video_entries, dict) and video_entries:
            _add(
                report,
                "error",
                "metadata",
                "meta/info.json is missing video_path while video modalities exist",
            )

    total_episodes = info.get("total_episodes")
    if total_episodes is not None and total_episodes != len(dataset.episodes):
        _add(
            report,
            "error",
            "episodes",
            f"total_episodes={total_episodes} but episodes.jsonl has "
            f"{len(dataset.episodes)} records",
        )

    total_tasks = info.get("total_tasks")
    if total_tasks is not None and total_tasks != len(dataset.tasks):
        _add(
            report,
            "error",
            "tasks",
            f"total_tasks={total_tasks} but tasks.jsonl has "
            f"{len(dataset.tasks)} records",
        )


def _feature_dim(features: dict[str, Any], name: str) -> int | None:
    feature = features.get(name)
    if not isinstance(feature, dict):
        return None
    shape = feature.get("shape")
    if isinstance(shape, list) and shape:
        try:
            return int(np.prod(shape))
        except (TypeError, ValueError):
            return None
    if "dim" in feature:
        try:
            return int(feature["dim"])
        except (TypeError, ValueError):
            return None
    return None


def _validate_modality(dataset: Gr00tDatasetInfo, report: FullValidationReport) -> None:
    modality = dataset.modality
    features = dataset.info.get("features")
    if not isinstance(features, dict):
        return

    for section in ("state", "action"):
        entries = modality.get(section)
        if not isinstance(entries, dict) or not entries:
            _add(report, "error", "modality", f"modality.json is missing {section!r}")
            continue
        dim = _feature_dim(
            features, f"observation.{section}" if section == "state" else section
        )
        if dim is None:
            continue
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                _add(report, "error", "modality", f"{section}.{name} must be an object")
                continue
            start = entry.get("start")
            end = entry.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                _add(
                    report,
                    "error",
                    "modality",
                    f"{section}.{name} requires integer start and end",
                )
                continue
            if not 0 <= start < end <= dim:
                _add(
                    report,
                    "error",
                    "modality",
                    f"{section}.{name} range [{start}, {end}) is outside dim {dim}",
                )

    videos = modality.get("video")
    if videos is not None:
        if not isinstance(videos, dict):
            _add(report, "error", "modality", "modality.video must be an object")
        else:
            for name, entry in videos.items():
                if not isinstance(entry, dict):
                    _add(report, "error", "modality", f"video.{name} must be an object")
                    continue
                original_key = entry.get("original_key")
                if not isinstance(original_key, str):
                    _add(
                        report,
                        "error",
                        "modality",
                        f"video.{name} requires a string original_key",
                    )
                elif original_key not in features:
                    _add(
                        report,
                        "error",
                        "modality",
                        f"video.{name} references unknown feature {original_key!r}",
                    )

    annotations = modality.get("annotation")
    if annotations is not None:
        if not isinstance(annotations, dict):
            _add(report, "error", "modality", "modality.annotation must be an object")
        else:
            for name, entry in annotations.items():
                if not isinstance(entry, dict):
                    _add(
                        report,
                        "error",
                        "modality",
                        f"annotation.{name} must be an object",
                    )
                    continue
                original_key = entry.get("original_key")
                if not isinstance(original_key, str):
                    _add(
                        report,
                        "error",
                        "modality",
                        f"annotation.{name} requires a string original_key",
                    )
                elif (
                    original_key not in features
                    and original_key not in RESERVED_ANNOTATION_KEYS
                ):
                    _add(
                        report,
                        "error",
                        "modality",
                        f"annotation.{name} references unknown key {original_key!r}",
                    )


def _validate_episodes(dataset: Gr00tDatasetInfo, report: FullValidationReport) -> None:
    indices: list[int] = []
    total_length = 0
    task_names = {
        task.get("task") for task in dataset.tasks if isinstance(task.get("task"), str)
    }
    for record in dataset.episodes:
        index = record.get("episode_index")
        length = record.get("length")
        if not isinstance(index, int):
            _add(
                report,
                "error",
                "episodes",
                "episode record requires integer episode_index",
            )
            continue
        if not isinstance(length, int) or length <= 0:
            _add(
                report, "error", "episodes", f"episode {index} requires positive length"
            )
            continue
        indices.append(index)
        total_length += length
        tasks = record.get("tasks")
        if tasks is not None:
            if not isinstance(tasks, list) or not all(
                isinstance(t, str) for t in tasks
            ):
                _add(
                    report,
                    "error",
                    "episodes",
                    f"episode {index} tasks must be strings",
                )
            else:
                for task in tasks:
                    if task not in task_names:
                        _add(
                            report,
                            "warning",
                            "episodes",
                            f"episode {index} references unknown task {task!r}",
                        )

    if len(indices) != len(set(indices)):
        _add(report, "error", "episodes", "episode_index values must be unique")
    expected = set(range(dataset.total_episodes))
    if set(indices) != expected:
        missing = sorted(expected - set(indices))
        extra = sorted(set(indices) - expected)
        _add(
            report,
            "error",
            "episodes",
            f"episode indices must be 0..{dataset.total_episodes - 1}; "
            f"missing={missing}, extra={extra}",
        )
    if total_length != dataset.total_frames:
        _add(
            report,
            "error",
            "episodes",
            f"sum of episode lengths ({total_length}) != total_frames "
            f"({dataset.total_frames})",
        )

    task_indices = [
        task.get("task_index")
        for task in dataset.tasks
        if isinstance(task.get("task_index"), int)
    ]
    if len(task_indices) != len(set(task_indices)):
        _add(report, "error", "tasks", "task_index values must be unique")
    expected_tasks = set(range(len(dataset.tasks)))
    if set(task_indices) != expected_tasks:
        _add(
            report,
            "error",
            "tasks",
            f"task indices must be 0..{len(dataset.tasks) - 1}",
        )


def _data_path(dataset: Gr00tDatasetInfo, episode_index: int) -> Path:
    template = str(dataset.info["data_path"])
    return dataset.path / template.format(
        episode_chunk=episode_index // dataset.chunks_size,
        episode_index=episode_index,
        chunks_size=dataset.chunks_size,
    )


def _video_path(dataset: Gr00tDatasetInfo, episode_index: int, video_key: str) -> Path:
    template = str(dataset.info["video_path"])
    return dataset.path / template.format(
        episode_chunk=episode_index // dataset.chunks_size,
        episode_index=episode_index,
        video_key=video_key,
        chunks_size=dataset.chunks_size,
    )


def _validate_files(
    dataset: Gr00tDatasetInfo,
    report: FullValidationReport,
    *,
    check_videos: bool = True,
) -> None:
    import pyarrow.parquet as pq

    for record in dataset.episodes:
        index = record.get("episode_index")
        length = record.get("length")
        if not isinstance(index, int) or not isinstance(length, int):
            continue
        path = _data_path(dataset, index)
        if not path.is_file():
            _add(report, "error", "data", f"missing episode parquet: {path}", index)
            continue
        try:
            rows = pq.read_metadata(path).num_rows
        except Exception as exc:
            _add(report, "error", "data", f"cannot read {path}: {exc}", index)
            continue
        if rows != length:
            _add(
                report,
                "error",
                "data",
                f"{path} has {rows} rows but episode length is {length}",
                index,
            )

    if not check_videos:
        return
    videos = dataset.modality.get("video")
    if not isinstance(videos, dict) or not videos:
        return
    if "video_path" not in dataset.info:
        return
    for record in dataset.episodes:
        index = record.get("episode_index")
        if not isinstance(index, int):
            continue
        for name, entry in videos.items():
            if not isinstance(entry, dict) or "original_key" not in entry:
                continue
            video_key = str(entry.get("video_key", entry["original_key"]))
            path = _video_path(dataset, index, video_key)
            if not path.is_file():
                _add(
                    report,
                    "warning",
                    "video",
                    f"missing video for {name}: {path}",
                    index,
                )


def validate_gr00t_dataset(
    path: str | Path, *, check_video_files: bool = True
) -> FullValidationReport:
    """Validate a GR00T-compatible LeRobot v2.1 dataset without loading model code."""
    started = time.perf_counter()
    dataset_path = Path(path)
    report = FullValidationReport()

    dataset = _parse_meta(dataset_path, report)
    if dataset is not None:
        _validate_info(dataset, report)
        _validate_modality(dataset, report)
        _validate_episodes(dataset, report)
        if not any(issue.severity == "error" for issue in report.issues):
            _validate_files(dataset, report, check_videos=check_video_files)
        report.episodes_checked = len(dataset.episodes)
        video_modalities = dataset.modality.get("video")
        if isinstance(video_modalities, dict):
            report.cameras_checked = sorted(video_modalities)

    report.duration_s = time.perf_counter() - started
    return report


def load_gr00t_dataset(path: str | Path) -> Gr00tDatasetInfo:
    """Parse GR00T metadata, raising ``ValueError`` if the dataset is invalid."""
    report = validate_gr00t_dataset(path, check_video_files=False)
    report.raise_if_errors()
    dataset = _parse_meta(Path(path), FullValidationReport())
    if dataset is None:
        raise ValueError(f"not a GR00T dataset: {path}")
    return dataset


class Gr00tLeRobotV2Loader(DatasetLoader):
    """Load tabular episodes from a GR00T-compatible LeRobot v2.1 dataset."""

    def __init__(self, path: str):
        super().__init__(path)
        self.dataset = load_gr00t_dataset(path)
        self.info = self.dataset.info
        self._episodes = {
            int(record["episode_index"]): record for record in self.dataset.episodes
        }
        self._task_index = {
            task["task"]: int(task["task_index"])
            for task in self.dataset.tasks
            if isinstance(task.get("task"), str)
            and isinstance(task.get("task_index"), int)
        }

    def _record(self, episode_id: int) -> dict[str, Any]:
        try:
            return self._episodes[episode_id]
        except KeyError:
            raise KeyError(
                f"episode {episode_id} not found (available: {sorted(self._episodes)})"
            ) from None

    def _data_file(self, episode_id: int) -> Path:
        return _data_path(self.dataset, episode_id)

    def _video_files(self, episode_id: int) -> dict[str, str]:
        videos = self.dataset.modality.get("video")
        if not isinstance(videos, dict) or "video_path" not in self.info:
            return {}
        result: dict[str, str] = {}
        for name, entry in videos.items():
            if not isinstance(entry, dict) or "original_key" not in entry:
                continue
            video_key = str(entry.get("video_key", entry["original_key"]))
            result[name] = str(_video_path(self.dataset, episode_id, video_key))
        return result

    def load_episode(self, episode_id: int) -> dict[str, Any]:
        """Load a full episode as per-frame feature dictionaries."""
        import pyarrow.parquet as pq

        record = self._record(episode_id)
        table = pq.read_table(self._data_file(episode_id))
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        length = int(record["length"])
        tasks = record.get("tasks", [])
        if not isinstance(tasks, list):
            tasks = []
        frames: list[dict[str, Any]] = []
        for frame_index in range(length):
            frame = {name: values[frame_index] for name, values in columns.items()}
            frame["episode_index"] = episode_id
            frame["frame_index"] = frame_index
            frames.append(frame)
        return {
            "episode_id": episode_id,
            "length": length,
            "tasks": tasks,
            "task_index": [
                self._task_index[task] for task in tasks if task in self._task_index
            ],
            "frames": frames,
            "videos": self._video_files(episode_id),
        }

    def load_frame(self, episode_id: int, frame_idx: int) -> dict[str, Any]:
        """Load one frame from an episode."""
        import pyarrow.parquet as pq

        record = self._record(episode_id)
        length = int(record["length"])
        if frame_idx < 0 or frame_idx >= length:
            raise IndexError(
                f"frame {frame_idx} out of range for episode {episode_id} "
                f"(length {length})"
            )
        table = pq.read_table(self._data_file(episode_id)).slice(frame_idx, 1)
        frame = table.to_pylist()[0]
        frame["episode_index"] = episode_id
        frame["frame_index"] = frame_idx
        return frame

    def get_episode_count(self) -> int:
        return len(self._episodes)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "format": "gr00t",
            "path": str(self.path),
            "description": "GR00T-compatible LeRobot v2.1 dataset",
            "codebase_version": self.dataset.codebase_version,
            "robot_type": self.info.get("robot_type"),
            "fps": self.dataset.fps,
            "n_episodes": self.get_episode_count(),
            "total_frames": self.dataset.total_frames,
            "n_tasks": len(self.dataset.tasks),
            "modalities": sorted(self.dataset.modality),
        }
