"""Tests for pyroboframes.security path validation and its wiring into loader entry points."""

import os

import pytest

from pyroboframes.security import validate_dataset_path
from test_loader import make_dataset

import pyroboframes as prf


def test_validate_dataset_path_rejects_traversal_component():
    with pytest.raises(ValueError, match="Directory traversal"):
        validate_dataset_path("some/../escape")


def test_validate_dataset_path_rejects_leading_traversal():
    with pytest.raises(ValueError, match="Directory traversal"):
        validate_dataset_path("../escape")


def test_validate_dataset_path_allows_dotdot_free_relative_path(tmp_path):
    # No ".." component and no base_dir: just resolves to an absolute path.
    resolved = validate_dataset_path(tmp_path / "sub" / "file.h5")
    assert resolved.is_absolute()


def test_validate_dataset_path_allows_contained_path(tmp_path):
    inside = tmp_path / "data" / "file.h5"
    inside.parent.mkdir()
    resolved = validate_dataset_path(inside, base_dir=tmp_path)
    assert resolved == inside.resolve()


def test_validate_dataset_path_rejects_escape_from_base_dir(tmp_path):
    base = tmp_path / "sandbox"
    base.mkdir()
    outside = tmp_path / "outside" / "file.h5"
    outside.parent.mkdir()
    with pytest.raises(ValueError, match="must be within"):
        validate_dataset_path(outside, base_dir=base)


def test_robo_frame_dataset_from_path_base_dir_allows_contained(tmp_path):
    root = tmp_path / "datasets" / "ds"
    os.makedirs(root)
    make_dataset(str(root), episodes=1, length=4)

    ds = prf.RoboFrameDataset.from_path(str(root), base_dir=str(tmp_path))
    assert ds.num_episodes == 1


def test_robo_frame_dataset_from_path_base_dir_rejects_escape(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside" / "ds"
    os.makedirs(outside)
    make_dataset(str(outside), episodes=1, length=4)

    with pytest.raises(ValueError, match="escapes base_dir"):
        prf.RoboFrameDataset.from_path(str(outside), base_dir=str(sandbox))


def test_robo_frame_dataset_from_path_without_base_dir_is_unrestricted(tmp_path):
    # No base_dir: preserves pre-existing unrestricted behavior.
    root = tmp_path / "ds"
    os.makedirs(root)
    make_dataset(str(root), episodes=1, length=4)

    ds = prf.RoboFrameDataset.from_path(str(root))
    assert ds.num_episodes == 1


def test_convert_mcap_base_dir_rejects_escape(tmp_path):
    mcap_writer = pytest.importorskip("mcap.writer")

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    mcap_path = tmp_path / "run.mcap"
    with open(mcap_path, "wb") as fh:
        w = mcap_writer.Writer(fh)
        w.start()
        w.finish()

    with pytest.raises(ValueError, match="escapes base_dir"):
        prf.convert_mcap(str(mcap_path), str(tmp_path / "out"), base_dir=str(sandbox))


def test_validate_dataset_path_secure_resolves_relative_against_root(tmp_path):
    resolved = validate_dataset_path(
        "data/file.h5", secure=True, project_root=tmp_path
    )
    assert resolved == (tmp_path / "data" / "file.h5").resolve()


def test_validate_dataset_path_secure_rejects_absolute_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside" / "file.h5"
    with pytest.raises(ValueError, match="project root"):
        validate_dataset_path(outside, secure=True, project_root=root)


def test_validate_dataset_path_secure_allow_outside_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside" / "file.h5"
    assert (
        validate_dataset_path(
            outside, secure=True, allow_outside_root=True, project_root=root
        )
        == outside.resolve()
    )


def test_validate_dataset_path_explicit_base_dir_stays_authoritative(tmp_path):
    base = tmp_path / "sandbox"
    base.mkdir()
    outside = tmp_path / "outside.h5"
    with pytest.raises(ValueError, match="must be within base_dir"):
        validate_dataset_path(
            outside,
            base_dir=base,
            secure=True,
            allow_outside_root=True,
        )


def test_hdf5_and_netcdf_secure_use_project_root_env(tmp_path, monkeypatch):
    import numpy as np

    h5py = pytest.importorskip("h5py")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")

    monkeypatch.setenv("PYROBOFRAMES_PROJECT_ROOT", str(tmp_path))
    h5_path = tmp_path / "data.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_group("demo_0").create_dataset(
            "actions", data=np.zeros((2, 2), dtype="f4")
        )
    nc_path = tmp_path / "data.nc"
    xr.Dataset(
        {"state": (("time", "dim"), np.zeros((2, 2), dtype="f4"))},
        coords={"time": [0.0, 1.0]},
    ).to_netcdf(nc_path)

    h5_ds = prf.HDF5Dataset.from_path("data.h5", secure=True)
    nc_ds = prf.NetCDFDataset.from_path("data.nc", secure=True)
    assert h5_ds.path == str(h5_path.resolve())
    assert nc_ds.path == str(nc_path.resolve())

    h5_report = prf.convert_hdf5(
        "data.h5", str(tmp_path / "out-h5"), secure=True
    )
    nc_report = prf.convert_netcdf(
        "data.nc", str(tmp_path / "out-nc"), secure=True
    )
    assert h5_report.episodes_converted == 1
    assert nc_report.episodes_converted == 1

    with pytest.raises(ValueError, match="project root"):
        prf.HDF5Dataset.from_path(str(tmp_path.parent / "escape.h5"), secure=True)
    with pytest.raises(ValueError, match="project root"):
        prf.NetCDFDataset.from_path(str(tmp_path.parent / "escape.nc"), secure=True)
    with pytest.raises(ValueError, match="project root"):
        prf.convert_hdf5(
            str(tmp_path.parent / "escape.h5"),
            str(tmp_path / "out-escape"),
            secure=True,
        )
    with pytest.raises(ValueError, match="project root"):
        prf.convert_netcdf(
            str(tmp_path.parent / "escape.nc"),
            str(tmp_path / "out-escape-nc"),
            secure=True,
        )


def test_robo_frame_dataset_secure_relative_and_escape(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    os.makedirs(root / "ds")
    make_dataset(str(root / "ds"), episodes=1, length=4)
    monkeypatch.setenv("PYROBOFRAMES_PROJECT_ROOT", str(root))

    ds = prf.RoboFrameDataset.from_path("ds", secure=True)
    assert ds.num_episodes == 1

    outside = tmp_path / "outside" / "ds"
    os.makedirs(outside)
    make_dataset(str(outside), episodes=1, length=4)
    with pytest.raises(ValueError, match="escapes project root"):
        prf.RoboFrameDataset.from_path(str(outside), secure=True)
    allowed = prf.RoboFrameDataset.from_path(
        str(outside), secure=True, allow_outside_root=True
    )
    assert allowed.num_episodes == 1


def test_convert_mcap_secure_project_root_and_escape_hatch(tmp_path, monkeypatch):
    mcap_writer = pytest.importorskip("mcap.writer")

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("PYROBOFRAMES_PROJECT_ROOT", str(project))
    mcap_path = tmp_path / "run.mcap"
    with open(mcap_path, "wb") as fh:
        writer = mcap_writer.Writer(fh)
        writer.start()
        writer.finish()

    with pytest.raises(ValueError, match="escapes project root"):
        prf.convert_mcap(
            str(mcap_path), str(tmp_path / "out"), secure=True
        )

    report = prf.convert_mcap(
        str(mcap_path),
        str(tmp_path / "out"),
        secure=True,
        allow_outside_root=True,
    )
    assert isinstance(report, dict)
