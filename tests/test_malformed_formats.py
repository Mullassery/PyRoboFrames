"""Malformed HDF5/NetCDF inputs must fail with a clear error, not surface
backend internals or hang. These use the real libraries and real corrupt files.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("netCDF4")

from pyroboframes.hdf5 import HDF5Dataset  # noqa: E402
from pyroboframes.netcdf import NetCDFDataset  # noqa: E402


def test_hdf5_rejects_corrupt_file(tmp_path) -> None:
    path = tmp_path / "corrupt.h5"
    path.write_bytes(b"this is not an HDF5 file")
    with pytest.raises(ValueError, match="Invalid or unreadable HDF5"):
        HDF5Dataset.from_path(str(path))


def test_hdf5_accepts_valid_empty_file(tmp_path) -> None:
    import h5py

    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    assert HDF5Dataset.from_path(str(path)).path == str(path)


def test_netcdf_rejects_corrupt_file(tmp_path) -> None:
    path = tmp_path / "corrupt.nc"
    path.write_bytes(b"this is not a NetCDF file")
    with pytest.raises(ValueError, match="Invalid or unreadable NetCDF"):
        NetCDFDataset.from_path(str(path))


def test_netcdf_accepts_valid_file(tmp_path) -> None:
    import xarray as xr

    path = tmp_path / "valid.nc"
    xr.Dataset({"x": ("time", np.arange(3.0))}).to_netcdf(path)
    assert NetCDFDataset.from_path(str(path)).path == str(path)
