"""Tests for the GR00T LeRobot v2.1 reader and validator."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pyroboframes import (
    DatasetFormat,
    Gr00tLeRobotV2Loader,
    load_dataset,
    load_gr00t_dataset,
    validate_gr00t_dataset,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _make_dataset(root, *, total_frames=4, modality=None, video=True):
    meta = root / "meta"
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": 2,
        "total_frames": total_frames,
        "total_tasks": 2,
        "total_videos": 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": "0:2"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [4]},
            "action": {"dtype": "float32", "shape": [4]},
            "observation.images.front": {"dtype": "video", "shape": [2, 2, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    _write_json(meta / "info.json", info)
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick"], "length": 2},
            {"episode_index": 1, "tasks": ["place"], "length": 2},
        ],
    )
    _write_jsonl(
        meta / "tasks.jsonl",
        [
            {"task_index": 0, "task": "pick"},
            {"task_index": 1, "task": "place"},
        ],
    )
    _write_json(
        meta / "modality.json",
        modality
        if modality is not None
        else {
            "state": {"arm": {"start": 0, "end": 4}},
            "action": {"arm": {"start": 0, "end": 4}},
            "video": {"front": {"original_key": "observation.images.front"}},
            "annotation": {"task": {"original_key": "task_index"}},
        },
    )

    for episode_index in (0, 1):
        data_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "observation.state": [[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]],
                "action": [[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]],
                "timestamp": [0.0, 0.1],
                "frame_index": [0, 1],
                "episode_index": [episode_index, episode_index],
                "index": [episode_index * 2, episode_index * 2 + 1],
                "task_index": [episode_index, episode_index],
            }
        )
        pq.write_table(table, data_path)

        if video:
            video_path = (
                root
                / "videos"
                / "chunk-000"
                / "observation.images.front"
                / f"episode_{episode_index:06d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"")

    return root


def test_valid_dataset(tmp_path):
    root = _make_dataset(tmp_path)
    report = validate_gr00t_dataset(root)

    assert report.ok
    assert report.episodes_checked == 2
    assert report.cameras_checked == ["front"]


def test_loader_reads_episode_and_frame(tmp_path):
    root = _make_dataset(tmp_path)
    loader = Gr00tLeRobotV2Loader(str(root))

    assert isinstance(loader, Gr00tLeRobotV2Loader)
    assert loader.get_episode_count() == 2
    assert loader.get_metadata()["format"] == "gr00t"

    episode = loader.load_episode(0)
    assert episode["length"] == 2
    assert episode["tasks"] == ["pick"]
    assert episode["task_index"] == [0]
    assert len(episode["frames"]) == 2
    assert episode["frames"][1]["observation.state"] == [1.0, 2.0, 3.0, 4.0]

    frame = loader.load_frame(1, 1)
    assert frame["episode_index"] == 1
    assert frame["frame_index"] == 1
    assert frame["task_index"] == 1


def test_loader_rejects_bad_frame_index(tmp_path):
    root = _make_dataset(tmp_path)
    loader = Gr00tLeRobotV2Loader(str(root))

    with pytest.raises(IndexError):
        loader.load_frame(0, 2)


def test_registry_auto_detects_gr00t(tmp_path):
    root = _make_dataset(tmp_path)
    loader = load_dataset(str(root))

    assert isinstance(loader, Gr00tLeRobotV2Loader)


def test_registry_hint_selects_gr00t(tmp_path):
    root = _make_dataset(tmp_path)
    loader = load_dataset(str(root), format_hint="GR00T")

    assert isinstance(loader, Gr00tLeRobotV2Loader)
    assert DatasetFormat.GROOT.value == "gr00t"


def test_missing_modality_is_error(tmp_path):
    root = _make_dataset(tmp_path)
    (root / "meta" / "modality.json").unlink()

    report = validate_gr00t_dataset(root)

    assert not report.ok
    assert any("modality.json" in issue.message for issue in report.errors)


def test_total_frames_mismatch_is_error(tmp_path):
    root = _make_dataset(tmp_path, total_frames=5)

    report = validate_gr00t_dataset(root)

    assert not report.ok
    assert any("total_frames" in issue.message for issue in report.errors)


def test_out_of_range_modality_is_error(tmp_path):
    modality = {
        "state": {"arm": {"start": 0, "end": 5}},
        "action": {"arm": {"start": 0, "end": 4}},
        "video": {"front": {"original_key": "observation.images.front"}},
    }
    root = _make_dataset(tmp_path, modality=modality)

    report = validate_gr00t_dataset(root)

    assert not report.ok
    assert any("outside dim 4" in issue.message for issue in report.errors)


def test_missing_data_file_is_error(tmp_path):
    root = _make_dataset(tmp_path)
    (root / "data" / "chunk-000" / "episode_000001.parquet").unlink()

    report = validate_gr00t_dataset(root)

    assert not report.ok
    assert any("missing episode parquet" in issue.message for issue in report.errors)


def test_missing_video_is_warning(tmp_path):
    root = _make_dataset(tmp_path, video=False)

    report = validate_gr00t_dataset(root)

    assert report.ok
    assert len(report.warnings) == 2


def test_load_gr00t_dataset_returns_metadata(tmp_path):
    root = _make_dataset(tmp_path)
    dataset = load_gr00t_dataset(root)

    assert dataset.codebase_version == "v2.1"
    assert dataset.total_episodes == 2
    assert dataset.total_frames == 4
    assert dataset.fps == 10.0


def test_loader_raises_on_invalid_dataset(tmp_path):
    root = _make_dataset(tmp_path)
    (root / "meta" / "modality.json").unlink()

    with pytest.raises(ValueError):
        Gr00tLeRobotV2Loader(str(root))
