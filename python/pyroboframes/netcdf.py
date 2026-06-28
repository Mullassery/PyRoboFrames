"""NetCDF dataset reader and converter for robot learning data.

Reads NetCDF files (common in scientific robotics, simulation, and ocean/climate-style datasets)
and converts them to LeRobot-compatible Parquet layout.

```python
from pyroboframes.netcdf import NetCDFDataset, convert_netcdf

ds = NetCDFDataset.from_path("simulation_data.nc")
print(ds.inspect())
df = ds.to_robotics_dataframe()

# With explicit episode boundaries
convert_netcdf("data.nc", "/out/lerobot", episode_breaks=[0, 500, 1200])
```
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ConversionReport:
    """Summary of a NetCDF → LeRobot conversion."""

    episodes_converted: int = 0
    features_extracted: list[str] = field(default_factory=list)
    skipped_vars: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"NetCDF conversion: {self.episodes_converted} episodes",
            f"  Features: {self.features_extracted}",
        ]
        if self.skipped_vars:
            lines.append(f"  Skipped: {self.skipped_vars}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class NetCDFDataset:
    """Reader for NetCDF robot learning datasets.

    Episode boundaries are inferred from a ``done``/``terminal`` variable if present,
    from an explicit ``episode_breaks`` array, or the entire file is treated as one episode.

    Args:
        path: Path to the ``.nc`` / ``.netcdf`` file.
        time_dim: Name of the time dimension (default: ``"time"``).
        episode_breaks: Array of frame indices marking episode starts (0 is always included).
    """

    def __init__(
        self,
        path: str,
        time_dim: str = "time",
        episode_breaks: Optional[np.ndarray] = None,
    ) -> None:
        try:
            import xarray  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "xarray is required to read NetCDF files: pip install xarray netCDF4"
            ) from exc
        self.path = path
        self.time_dim = time_dim
        self.episode_breaks = episode_breaks

    @classmethod
    def from_path(
        cls,
        path: str,
        time_dim: str = "time",
        episode_breaks: Optional[np.ndarray] = None,
    ) -> "NetCDFDataset":
        """Open a NetCDF dataset.

        Args:
            path: Path to the file.
            time_dim: Name of the time dimension.
            episode_breaks: Frame indices where new episodes start.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"NetCDF file not found: {path!r}")
        return cls(path, time_dim=time_dim, episode_breaks=episode_breaks)

    def inspect(self) -> dict[str, Any]:
        """Return variable names, shapes, dtypes, and dimension info."""
        import xarray as xr

        ds = xr.open_dataset(self.path)
        result: dict[str, Any] = {
            "path": self.path,
            "dimensions": dict(ds.dims),
            "variables": {},
        }
        for name, var in ds.data_vars.items():
            result["variables"][name] = {
                "dims": list(var.dims),
                "shape": list(var.shape),
                "dtype": str(var.dtype),
            }
        ds.close()
        return result

    def to_robotics_dataframe(self) -> Any:
        """Load as a :class:`~pyroboframes.dataframe.RoboticsDataFrame`."""
        import tempfile

        from .dataframe import RoboticsDataFrame

        with tempfile.TemporaryDirectory() as tmp:
            _write_lerobot_layout(self.path, tmp, self.time_dim, self.episode_breaks)
            return RoboticsDataFrame.from_converted(tmp)

    def to_lerobot(self, out_dir: str) -> ConversionReport:
        """Convert to LeRobot v3.0 layout.

        Args:
            out_dir: Output directory.
        """
        return _write_lerobot_layout(self.path, out_dir, self.time_dim, self.episode_breaks)


def _infer_episode_breaks(
    ds: Any, time_dim: str, explicit_breaks: Optional[np.ndarray]
) -> list[int]:
    """Return sorted episode start indices (always includes 0)."""
    if explicit_breaks is not None:
        breaks = sorted(set([0] + list(map(int, explicit_breaks))))
        return breaks

    T = ds.dims.get(time_dim, 0)
    if T == 0:
        return [0]

    # Look for a 'done' or 'terminal' variable along the time dimension.
    for candidate in ("done", "terminal", "is_terminal", "episode_done"):
        if candidate in ds.data_vars and time_dim in ds[candidate].dims:
            done = ds[candidate].values.astype(bool)
            # Episode break at frame after each True (transition to new episode).
            breaks = [0] + [int(i + 1) for i in range(len(done) - 1) if done[i]]
            return sorted(set(breaks))

    # Single episode.
    return [0]


def _write_lerobot_layout(
    nc_path: str,
    out_dir: str,
    time_dim: str,
    episode_breaks: Optional[np.ndarray],
) -> ConversionReport:
    """Convert a NetCDF file to LeRobot v3.0 Parquet layout."""
    import xarray as xr

    report = ConversionReport()
    ds = xr.open_dataset(nc_path)
    T = ds.dims.get(time_dim, 0)
    if T == 0:
        ds.close()
        raise ValueError(f"No time dimension {time_dim!r} found or it is length 0.")

    breaks = _infer_episode_breaks(ds, time_dim, episode_breaks)
    # Add sentinel for last episode end.
    episode_slices = list(zip(breaks, breaks[1:] + [T]))

    # Extract numeric variables along the time dimension.
    arrays: dict[str, np.ndarray] = {}
    for name, var in ds.data_vars.items():
        if time_dim not in var.dims:
            report.skipped_vars.append(name)
            continue
        if var.dtype.kind not in ("f", "i", "u", "b"):
            report.skipped_vars.append(name)
            continue
        arr = var.values.astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim > 2:
            arr = arr.reshape(T, -1)
        arrays[name] = arr
        report.features_extracted.append(name)
    ds.close()

    if not arrays:
        raise ValueError("No numeric variables along the time dimension found.")

    episode_lengths = [end - start for start, end in episode_slices]
    os.makedirs(out_dir, exist_ok=True)

    from .lerobot import write_lerobot_dataset
    write_lerobot_dataset(out_dir, arrays, episode_lengths)
    report.episodes_converted = len(episode_lengths)
    return report


def convert_netcdf(
    path: str,
    out_dir: str,
    *,
    time_dim: str = "time",
    episode_breaks: Optional[np.ndarray] = None,
) -> ConversionReport:
    """Convert a NetCDF robot dataset to LeRobot v3.0 Parquet layout.

    Args:
        path: Path to the ``.nc`` input file.
        out_dir: Output directory (created if needed).
        time_dim: Name of the time dimension (default: ``"time"``).
        episode_breaks: Frame indices where new episodes begin. If None, auto-detected
            from a ``done``/``terminal`` variable or treated as a single episode.

    Returns:
        :class:`ConversionReport` summarising the conversion.
    """
    return _write_lerobot_layout(path, out_dir, time_dim, episode_breaks)
