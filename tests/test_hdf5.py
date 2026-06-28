"""Tests for HDF5 dataset reader and converter."""

import os
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pyroboframes.hdf5 import HDF5Dataset, ConversionReport, convert_hdf5


def _write_hdf5(path: str, n_episodes: int = 2, length: int = 10) -> str:
    """Write a minimal HDF5 file with demo_0, demo_1, ... episode groups."""
    with h5py.File(path, "w") as f:
        for i in range(n_episodes):
            grp = f.create_group(f"demo_{i}")
            obs = grp.create_group("obs")
            obs.create_dataset("state", data=np.random.rand(length, 4).astype(np.float32))
            obs.create_dataset("gripper", data=np.random.rand(length, 1).astype(np.float32))
            grp.create_dataset("actions", data=np.random.rand(length, 4).astype(np.float32))
    return path


def test_hdf5_dataset_from_path(tmp_path):
    path = str(tmp_path / "data.h5")
    _write_hdf5(path)
    ds = HDF5Dataset.from_path(path)
    assert ds.path == path


def test_hdf5_dataset_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        HDF5Dataset.from_path("/nonexistent/data.h5")


def test_hdf5_inspect_has_episodes(tmp_path):
    path = str(tmp_path / "data.h5")
    _write_hdf5(path, n_episodes=3)
    ds = HDF5Dataset.from_path(path)
    info = ds.inspect()
    assert info["episodes"] == 3
    assert "structure" in info


def test_hdf5_inspect_has_structure_keys(tmp_path):
    path = str(tmp_path / "data.h5")
    _write_hdf5(path, n_episodes=2)
    ds = HDF5Dataset.from_path(path)
    info = ds.inspect()
    assert "demo_0" in info["structure"]


def test_hdf5_episode_count(tmp_path):
    path = str(tmp_path / "data.h5")
    _write_hdf5(path, n_episodes=4)
    ds = HDF5Dataset.from_path(path)
    assert ds.episode_count() == 4


def test_convert_hdf5_creates_parquet(tmp_path):
    hdf5_path = str(tmp_path / "data.h5")
    out_dir = str(tmp_path / "lerobot")
    _write_hdf5(hdf5_path, n_episodes=2, length=8)

    report = convert_hdf5(hdf5_path, out_dir)

    assert isinstance(report, ConversionReport)
    assert report.episodes_converted == 2
    assert len(report.features_extracted) > 0
    # Check LeRobot layout files exist.
    assert os.path.exists(os.path.join(out_dir, "meta", "info.json"))
    assert os.path.exists(os.path.join(out_dir, "data", "chunk-000", "file-000.parquet"))


def test_convert_hdf5_features_include_actions(tmp_path):
    hdf5_path = str(tmp_path / "data.h5")
    out_dir = str(tmp_path / "lerobot")
    _write_hdf5(hdf5_path, n_episodes=2, length=5)

    report = convert_hdf5(hdf5_path, out_dir)
    # "actions" should appear in extracted features.
    assert any("actions" in f for f in report.features_extracted)


def test_convert_hdf5_lerobot_loadable(tmp_path):
    import pyroboframes as prf

    hdf5_path = str(tmp_path / "data.h5")
    out_dir = str(tmp_path / "lerobot")
    _write_hdf5(hdf5_path, n_episodes=2, length=10)
    convert_hdf5(hdf5_path, out_dir)

    ds = prf.RoboFrameDataset.from_path(out_dir)
    assert ds.num_episodes() == 2
    assert ds.total_frames() == 20


def test_hdf5_to_lerobot_report(tmp_path):
    hdf5_path = str(tmp_path / "data.h5")
    out_dir = str(tmp_path / "lerobot")
    _write_hdf5(hdf5_path, n_episodes=3, length=6)

    ds = HDF5Dataset.from_path(hdf5_path)
    report = ds.to_lerobot(out_dir)
    assert report.episodes_converted == 3
