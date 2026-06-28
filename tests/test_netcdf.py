"""Tests for NetCDF dataset reader and converter."""

import os
import numpy as np
import pytest

xarray = pytest.importorskip("xarray")

from pyroboframes.netcdf import NetCDFDataset, ConversionReport, convert_netcdf


def _write_netcdf(path: str, n_timesteps: int = 30, n_episodes: int = 3) -> str:
    """Write a minimal NetCDF file with state, action, and done variables."""
    import xarray as xr

    T = n_timesteps
    ep_len = T // n_episodes

    state = np.random.rand(T, 4).astype(np.float32)
    action = np.random.rand(T, 4).astype(np.float32)
    done = np.zeros(T, dtype=np.float32)
    # Mark episode boundaries.
    for i in range(1, n_episodes):
        done[i * ep_len - 1] = 1.0

    ds = xr.Dataset(
        {
            "state": (["time", "state_dim"], state),
            "action": (["time", "action_dim"], action),
            "done": (["time"], done),
        },
        coords={"time": np.arange(T, dtype=np.float64)},
    )
    ds.to_netcdf(path)
    return path


def test_netcdf_from_path(tmp_path):
    path = str(tmp_path / "data.nc")
    _write_netcdf(path)
    ds = NetCDFDataset.from_path(path)
    assert ds.path == path


def test_netcdf_not_found():
    with pytest.raises(FileNotFoundError):
        NetCDFDataset.from_path("/no/such/file.nc")


def test_netcdf_inspect_has_variables(tmp_path):
    path = str(tmp_path / "data.nc")
    _write_netcdf(path, n_timesteps=20)
    ds = NetCDFDataset.from_path(path)
    info = ds.inspect()
    assert "variables" in info
    assert "state" in info["variables"]
    assert "action" in info["variables"]


def test_netcdf_inspect_has_dimensions(tmp_path):
    path = str(tmp_path / "data.nc")
    _write_netcdf(path, n_timesteps=20)
    ds = NetCDFDataset.from_path(path)
    info = ds.inspect()
    assert "dimensions" in info
    assert "time" in info["dimensions"]


def test_convert_netcdf_auto_detects_episodes(tmp_path):
    path = str(tmp_path / "data.nc")
    out_dir = str(tmp_path / "lerobot")
    _write_netcdf(path, n_timesteps=30, n_episodes=3)

    report = convert_netcdf(path, out_dir)
    assert isinstance(report, ConversionReport)
    assert report.episodes_converted == 3


def test_convert_netcdf_explicit_breaks(tmp_path):
    path = str(tmp_path / "data.nc")
    out_dir = str(tmp_path / "lerobot")

    import xarray as xr
    T = 40
    ds_xr = xr.Dataset(
        {
            "state": (["time", "d"], np.random.rand(T, 3).astype(np.float32)),
            "action": (["time", "d"], np.random.rand(T, 3).astype(np.float32)),
        },
        coords={"time": np.arange(T, dtype=np.float64)},
    )
    ds_xr.to_netcdf(path)

    report = convert_netcdf(path, out_dir, episode_breaks=np.array([0, 20]))
    assert report.episodes_converted == 2


def test_convert_netcdf_creates_parquet(tmp_path):
    path = str(tmp_path / "data.nc")
    out_dir = str(tmp_path / "lerobot")
    _write_netcdf(path, n_timesteps=30, n_episodes=3)

    convert_netcdf(path, out_dir)
    assert os.path.exists(os.path.join(out_dir, "meta", "info.json"))
    assert os.path.exists(os.path.join(out_dir, "data", "chunk-000", "file-000.parquet"))


def test_convert_netcdf_lerobot_loadable(tmp_path):
    import pyroboframes as prf

    path = str(tmp_path / "data.nc")
    out_dir = str(tmp_path / "lerobot")
    _write_netcdf(path, n_timesteps=30, n_episodes=3)
    convert_netcdf(path, out_dir)

    ds = prf.RoboFrameDataset.from_path(out_dir)
    assert ds.num_episodes() == 3
    assert ds.total_frames() == 30
