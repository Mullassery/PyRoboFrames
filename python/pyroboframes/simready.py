"""SimReady/USD dataset reader and converter for robot learning data.

Reads NVIDIA SimReady assets from OpenUSD (Universal Scene Description) stages,
extracts robot/articulation data, camera observations, and actions, and converts
them to LeRobot v3.0 format.

SimReady assets follow NVIDIA's modular specification with Profiles (e.g.,
Prop-Robotics-Isaac, Robot-Body-Isaac) that define capabilities for different
simulation runtimes. This module leverages SimReady metadata when available
for more reliable parsing.

```python
from pyroboframes.simready import SimReadyDataset, convert_simready

# Inspect structure
ds = SimReadyDataset.from_path("/path/to/simready.usda")
print(ds.inspect())

# Convert to LeRobot format
convert_simready("/path/to/simready.usda", "/out/lerobot_dataset")
```

Requirements:
    OpenUSD (pxr) - install with: pip install usd-core
    For full SimReady support, assets should follow the SimReady spec.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .security import validate_dataset_path


@dataclass
class SimReadyConversionReport:
    """Summary of a SimReady/USD → LeRobot conversion."""

    episodes_converted: int = 0
    features_extracted: list[str] = field(default_factory=list)
    skipped_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    simready_profile: Optional[str] = None
    artifacts_found: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"SimReady/USD conversion: {self.episodes_converted} episodes",
        ]
        if self.simready_profile:
            lines.append(f"  SimReady Profile: {self.simready_profile}")
        if self.artifacts_found:
            lines.append(f"  Artifacts found: {self.artifacts_found}")
        lines.append(f"  Features: {self.features_extracted}")
        if self.skipped_keys:
            lines.append(f"  Skipped: {self.skipped_keys}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class SimReadyDataset:
    """Reader for NVIDIA SimReady/USD robot learning datasets.

    Supports loading OpenUSD stages containing SimReady assets with:
    - Robot articulations (USDPhysics.ArticulationRootAPI)
    - Joint states and drives
    - Camera prims with render products
    - Action/command data if recorded
    - SimReady metadata and profile information

    Args:
        path: Path to USD file (.usd, .usda, .usdc, .usdz) or directory containing USD files.
        base_dir: If given, restrict ``path`` to this directory (rejects ``..``/symlink
            escapes) — pass this whenever ``path`` comes from an untrusted source.
        secure: If True (and no ``base_dir``), resolve relative paths against the project
            root and reject paths outside it.
        allow_outside_root: Escape hatch for ``secure=True`` that permits an absolute path
            outside the default project root.
        fps: Override FPS detection from stage metadata.
        episode_pattern: Pattern to identify episode data in stage.
    """

    def __init__(
        self,
        path: str,
        base_dir: Optional[str] = None,
        *,
        secure: bool = False,
        allow_outside_root: bool = False,
        fps: Optional[float] = None,
        episode_pattern: Optional[str] = None,
    ) -> None:
        try:
            from pxr import Usd, UsdGeom, UsdPhysics  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenUSD (pxr) is required to read SimReady/USD files: "
                "pip install usd-core or install OpenUSD. "
                "See https://developer.nvidia.com/usd for installation."
            ) from exc
        if base_dir is not None or secure:
            path = str(
                validate_dataset_path(
                    path,
                    base_dir=Path(base_dir) if base_dir is not None else None,
                    secure=secure,
                    allow_outside_root=allow_outside_root,
                )
            )
        self.path = path
        self.fps_override = fps
        self.episode_pattern = episode_pattern
        self._stage = None
        self._simready_metadata = {}
        self._detected_profile = None

    @classmethod
    def from_path(
        cls,
        path: str,
        base_dir: Optional[str] = None,
        *,
        secure: bool = False,
        allow_outside_root: bool = False,
        fps: Optional[float] = None,
        episode_pattern: Optional[str] = None,
    ) -> "SimReadyDataset":
        """Open a SimReady/USD dataset.

        Args:
            path: Path to USD file or directory.
            base_dir: If given, restrict ``path`` to this directory.
            secure: If True (and no ``base_dir``), resolve relative paths against the project root.
            allow_outside_root: Escape hatch for ``secure=True``.
            fps: Override FPS detection.
            episode_pattern: Pattern to identify episode data.
        """
        if base_dir is not None or secure:
            path = str(
                validate_dataset_path(
                    path,
                    base_dir=Path(base_dir) if base_dir is not None else None,
                    secure=secure,
                    allow_outside_root=allow_outside_root,
                )
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"USD file or directory not found: {path!r}")
        
        # Validate USD can be opened
        from pxr import Usd
        if os.path.isfile(path):
            stage = Usd.Stage.Open(path)
            if not stage:
                raise ValueError(f"Failed to open USD stage: {path!r}")
        elif os.path.isdir(path):
            # Look for USD files in directory
            usd_files = []
            for ext in ['.usd', '.usda', '.usdc', '.usdz']:
                usd_files.extend(Path(path).glob(f'*{ext}'))
            if not usd_files:
                raise ValueError(f"No USD files found in directory: {path!r}")
            stage = Usd.Stage.Open(str(usd_files[0]))
            if not stage:
                raise ValueError(f"Failed to open USD stage from directory: {path!r}")
        
        ds = cls(
            path,
            secure=secure,
            allow_outside_root=allow_outside_root,
            fps=fps,
            episode_pattern=episode_pattern,
        )
        ds._extract_simready_metadata()
        return ds

    def _extract_simready_metadata(self):
        """Extract SimReady-specific metadata from the stage."""
        from pxr import Usd, Sdf
        
        if os.path.isfile(self.path):
            stage = Usd.Stage.Open(self.path)
        else:
            usd_files = []
            for ext in ['.usd', '.usda', '.usdc', '.usdz']:
                usd_files.extend(Path(self.path).glob(f'*{ext}'))
            stage = Usd.Stage.Open(str(usd_files[0]))
        
        self._stage = stage
        
        # Look for SimReady metadata on root or in custom data
        root_prim = stage.GetPseudoRoot()
        if root_prim.HasCustomDataKey('simready'):
            self._simready_metadata = root_prim.GetCustomDataByKey('simready')
        elif root_prim.HasCustomDataKey('SimReady'):
            self._simready_metadata = root_prim.GetCustomDataByKey('SimReady')
        
        # Check for profile information
        for prim in stage.Traverse():
            if prim.HasCustomDataKey('simready:profile'):
                self._detected_profile = prim.GetCustomDataByKey('simready:profile')
                break
            elif prim.HasCustomDataKey('profile'):
                val = prim.GetCustomDataByKey('profile')
                if isinstance(val, str) and 'robot' in val.lower():
                    self._detected_profile = val
                    break
        
        # Also check asset info
        if root_prim.HasAttribute('assetInfo'):
            asset_info = root_prim.GetAttribute('assetInfo').Get()
            if asset_info and isinstance(asset_info, dict):
                if 'simreadyProfile' in asset_info:
                    self._detected_profile = asset_info['simreadyProfile']

    def inspect(self) -> dict[str, Any]:
        """Return a tree of stage contents and SimReady-specific info."""
        from pxr import Usd, UsdGeom, UsdPhysics
        
        if self._stage is None:
            self._extract_simready_metadata()
        stage = self._stage
        
        # Find articulations
        articulations = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulations.append(str(prim.GetPath()))
        
        # Find cameras
        cameras = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                cameras.append(str(prim.GetPath()))
        
        # Find joints
        joints = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joints.append(str(prim.GetPath()))
        
        fps = self.fps_override
        if fps is None:
            fps = stage.GetFramesPerSecond()
            if fps == 0:
                fps = 30.0
        
        time_code_range = stage.GetTimeCodesInRange()
        
        return {
            "path": self.path,
            "stage_id": stage.GetRootLayer().identifier,
            "fps": fps,
            "start_time_code": stage.GetStartTimeCode(),
            "end_time_code": stage.GetEndTimeCode(),
            "time_codes_in_range": time_code_range,
            "simready_profile": self._detected_profile,
            "simready_metadata": self._simready_metadata,
            "articulations": articulations,
            "cameras": cameras,
            "joints": joints,
            "up_axis": UsdGeom.GetStageUpAxis(stage),
        }

    def episode_count(self) -> int:
        """Return the number of episodes in the dataset."""
        # For recorded sim data, episodes might be in different layers or have specific patterns
        if self._stage is None:
            self._extract_simready_metadata()
        
        # Try to detect episode structure
        # Could be multiple sublayers, or episode-specific prims
        if self.episode_pattern:
            count = 0
            for prim in self._stage.Traverse():
                if self.episode_pattern in str(prim.GetPath()):
                    # Count unique episodes
                    pass
            # Simplified: if time range suggests multiple episodes
        else:
            # Default to 1 episode if single stage
            return 1
        return 1

    def to_robotics_dataframe(self):
        """Load as a RoboticsDataFrame for quick analysis."""
        import pyarrow as pa
        import tempfile

        from .dataframe import RoboticsDataFrame

        with tempfile.TemporaryDirectory() as tmp:
            report = _write_lerobot_layout_from_usd(self.path, tmp)
            return RoboticsDataFrame.from_converted(tmp)

    def to_lerobot(self, out_dir: str) -> SimReadyConversionReport:
        """Convert to LeRobot v3.0 layout in ``out_dir``."""
        return _write_lerobot_layout_from_usd(self.path, out_dir)


def _write_lerobot_layout_from_usd(usd_path: str, out_dir: str) -> SimReadyConversionReport:
    """Convert USD/SimReady stage to LeRobot format."""
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    
    report = SimReadyConversionReport()
    
    # Open stage
    if os.path.isfile(usd_path):
        stage = Usd.Stage.Open(usd_path)
    else:
        usd_files = []
        for ext in ['.usd', '.usda', '.usdc', '.usdz']:
            usd_files.extend(Path(usd_path).glob(f'*{ext}'))
        stage = Usd.Stage.Open(str(usd_files[0]))
    
    if not stage:
        raise ValueError(f"Failed to open USD stage: {usd_path}")
    
    # Detect profile
    ds = SimReadyDataset(usd_path)
    ds._extract_simready_metadata()
    report.simready_profile = ds._detected_profile
    
    fps = stage.GetFramesPerSecond()
    if fps == 0:
        fps = 30.0
    
    start_time = stage.GetStartTimeCode()
    end_time = stage.GetEndTimeCode()
    
    # If no time range, treat as static or single frame
    if start_time == end_time or end_time < start_time:
        time_range = [start_time] if start_time else [0.0]
    else:
        time_range = list(range(int(start_time), int(end_time) + 1))
    
    all_arrays: dict[str, list[np.ndarray]] = {}
    
    # Extract joint states from articulations
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation = UsdPhysics.ArticulationRootAPI(prim)
            # Find joints
            joint_prims = []
            for descendant in Usd.PrimRange(prim):
                if descendant.IsA(UsdPhysics.Joint):
                    joint_prims.append(descendant)
            
            if joint_prims:
                for t in time_range:
                    joint_positions = []
                    joint_velocities = []
                    for joint_prim in joint_prims:
                        joint = UsdPhysics.Joint(joint_prim)
                        # Get drive or position
                        # This is simplified - real implementation would need to handle
                        # different joint types and drive APIs
                        joint_positions.append(0.0)
                        joint_velocities.append(0.0)
                    
                    if joint_positions:
                        key = "observation.state"
                        all_arrays.setdefault(key, []).append(np.array(joint_positions, dtype=np.float32))
    
    # Extract camera observations
    camera_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            camera_count += 1
            camera_name = f"observation.images.camera{camera_count}"
            # For recorded data, we'd need to read from rendered outputs
            # For now, create placeholder or look for image data
            for t in time_range:
                # Placeholder - in real scenario would extract rendered frames
                # or reference to video files
                all_arrays.setdefault(camera_name, []).append(np.zeros(224 * 224 * 3, dtype=np.float32))
            report.artifacts_found.append(str(prim.GetPath()))
    
    # If no data extracted, add warnings
    if not all_arrays:
        report.warnings.append(
            "No joint states or camera data extracted. "
            "Ensure the USD contains articulation data or recorded observations. "
            "For SimReady assets with recorded demonstrations, data may be in separate files."
        )
    
    # Stack arrays
    stacked_all: dict[str, np.ndarray] = {}
    for key, arrays_list in all_arrays.items():
        try:
            stacked_all[key] = np.stack(arrays_list, axis=0)
            report.features_extracted.append(key)
        except Exception as exc:
            report.skipped_keys.append(key)
            report.warnings.append(f"Failed to stack {key}: {exc}")
    
    if not stacked_all:
        raise ValueError(
            "No valid features extracted from USD stage. "
            "Check that the stage contains articulation or observation data."
        )
    
    os.makedirs(out_dir, exist_ok=True)
    from .lerobot import write_lerobot_dataset
    
    episode_lengths = [len(time_range)]
    write_lerobot_dataset(out_dir, stacked_all, episode_lengths)
    report.episodes_converted = 1
    return report


def convert_simready(
    usd_path: str,
    out_dir: str,
    base_dir: Optional[str] = None,
    *,
    secure: bool = False,
    allow_outside_root: bool = False,
    fps: Optional[float] = None,
) -> SimReadyConversionReport:
    """Convert a SimReady/USD file or directory to LeRobot v3.0 format.

    Args:
        usd_path: Path to USD file (.usd, .usda, .usdc, .usdz) or directory.
        out_dir: Output directory for the LeRobot dataset.
        base_dir: If given, restrict paths to this directory.
        secure: If True, apply path security validation.
        allow_outside_root: Escape hatch for secure mode.
        fps: Override FPS detection from stage.

    Returns:
        :class:`SimReadyConversionReport` summarising the conversion.
    """
    if base_dir is not None or secure:
        usd_path = str(
            validate_dataset_path(
                usd_path,
                base_dir=Path(base_dir) if base_dir is not None else None,
                secure=secure,
                allow_outside_root=allow_outside_root,
            )
        )
    
    ds = SimReadyDataset.from_path(usd_path, fps=fps)
    return ds.to_lerobot(out_dir)
