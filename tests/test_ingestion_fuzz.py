"""Deterministic property tests for corrupt HDF5/NetCDF ingestion.

These are *bounded* smoke fuzz tests, not a long-running campaign: Hypothesis
runs a small, fixed number of cases (``max_examples=20``) and derandomizes the
search so failures are reproducible. They are part of the default suite; real
overnight fuzzing should live outside pytest.

The fixtures are real, minimal files written by h5py / xarray. Mutations cover
truncation, bit flips, and random headers. Every loader call has a wall-clock
limit so a C-library hang fails the test instead of wedging CI.
"""

from __future__ import annotations

import signal
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")
xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from hypothesis import HealthCheck, given, settings, strategies as st

from pyroboframes.hdf5 import HDF5Dataset
from pyroboframes.netcdf import NetCDFDataset


class IngestHang(AssertionError):
    """Raised when a loader exceeds the per-case wall-clock budget."""


@contextmanager
def _no_hang(seconds: float = 5.0):
    """Fail instead of hanging forever (POSIX main-thread alarm)."""
    if not hasattr(signal, "setitimer"):
        yield
        return

    def _timeout(_signum, _frame):
        raise IngestHang(f"loader exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@lru_cache(maxsize=1)
def _valid_hdf5_bytes() -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.h5"
        with h5py.File(path, "w") as handle:
            for episode in range(2):
                group = handle.create_group(f"demo_{episode}")
                group.create_dataset(
                    "actions", data=[[0.0, 1.0], [2.0, 3.0]], dtype="f4"
                )
                group.create_dataset(
                    "observation.state", data=[[4.0], [5.0]], dtype="f4"
                )
        return path.read_bytes()


@lru_cache(maxsize=1)
def _valid_netcdf_bytes() -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.nc"
        dataset = xr.Dataset(
            {
                "state": (("time", "dim"), [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
                "action": (("time", "dim"), [[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]),
                "done": (("time",), [0.0, 0.0, 1.0]),
            },
            coords={"time": [0.0, 1.0, 2.0]},
        )
        dataset.to_netcdf(path)
        return path.read_bytes()


@st.composite
def _guaranteed_corrupt_header(draw, valid: bytes):
    """Produce a blob whose first 8 signature bytes cannot match ``valid``."""
    mode = draw(st.sampled_from(("truncate", "bitflip", "random_header")))
    if mode == "truncate":
        return valid[: draw(st.integers(min_value=0, max_value=7))]
    if mode == "bitflip":
        mutated = bytearray(valid)
        mutated[draw(st.integers(0, 7))] ^= 1 << draw(st.integers(0, 7))
        return bytes(mutated)

    length = draw(st.integers(min_value=1, max_value=8))
    replacement = draw(st.binary(min_size=length, max_size=length))
    mutated = replacement + valid[length:]
    if mutated[:8] == valid[:8]:
        mutated = bytes([mutated[0] ^ 0xFF]) + mutated[1:]
    return mutated


@st.composite
def _arbitrary_mutation(draw, valid: bytes):
    """Truncate/bit-flip anywhere; a mutation may still leave a readable file."""
    mode = draw(st.sampled_from(("truncate", "bitflip", "bitflip_many")))
    if mode == "truncate":
        return valid[: draw(st.integers(min_value=0, max_value=len(valid) - 1))]
    if mode == "bitflip":
        mutated = bytearray(valid)
        mutated[draw(st.integers(0, len(valid) - 1))] ^= 1 << draw(
            st.integers(0, 7)
        )
        return bytes(mutated)
    mutated = bytearray(valid)
    for _ in range(draw(st.integers(min_value=2, max_value=8))):
        mutated[draw(st.integers(0, len(valid) - 1))] ^= 1 << draw(
            st.integers(0, 7)
        )
    return bytes(mutated)


_FUZZ_SETTINGS = settings(
    derandomize=True,
    database=None,
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)


def _write(tmp: Path, name: str, data: bytes) -> str:
    path = tmp / name
    path.write_bytes(data)
    return str(path)


@_FUZZ_SETTINGS
@given(mutated=_guaranteed_corrupt_header(_valid_hdf5_bytes()))
def test_hdf5_corrupt_header_raises_cleanly(mutated: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), "mutated.h5", mutated)
        with _no_hang():
            with pytest.raises((ValueError, OSError)):
                HDF5Dataset.from_path(path)


@_FUZZ_SETTINGS
@given(mutated=_guaranteed_corrupt_header(_valid_netcdf_bytes()))
def test_netcdf_corrupt_header_raises_cleanly(mutated: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), "mutated.nc", mutated)
        with _no_hang():
            with pytest.raises((ValueError, OSError)):
                NetCDFDataset.from_path(path)


@_FUZZ_SETTINGS
@given(mutated=_arbitrary_mutation(_valid_hdf5_bytes()))
def test_hdf5_arbitrary_mutation_never_crashes_or_hangs(mutated: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), "mutated.h5", mutated)
        try:
            with _no_hang():
                HDF5Dataset.from_path(path)
        except (ValueError, OSError):
            pass


@_FUZZ_SETTINGS
@given(mutated=_arbitrary_mutation(_valid_netcdf_bytes()))
def test_netcdf_arbitrary_mutation_never_crashes_or_hangs(mutated: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), "mutated.nc", mutated)
        try:
            with _no_hang():
                NetCDFDataset.from_path(path)
        except (ValueError, OSError):
            pass
