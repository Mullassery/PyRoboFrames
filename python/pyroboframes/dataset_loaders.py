"""Real-world autonomous driving dataset loaders.

Supports Waymo Open Dataset, nuScenes, and KITTI.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
import warnings


@dataclass
class CameraCalibration:
    """Camera intrinsic and extrinsic parameters."""

    fx: float  # Focal length X
    fy: float  # Focal length Y
    cx: float  # Principal point X
    cy: float  # Principal point Y
    width: int
    height: int
    distortion: Optional[np.ndarray] = None  # Distortion coefficients
    pose: Optional[np.ndarray] = None  # 4x4 extrinsic matrix


@dataclass
class FrameMetadata:
    """Metadata for a single frame."""

    timestamp: float
    camera_name: str
    frame_index: int
    image_path: str
    calibration: CameraCalibration


class WaymoDatasetLoader:
    """Loader for Waymo Open Dataset."""

    def __init__(self, dataset_path: str):
        """Initialize Waymo loader.

        Args:
            dataset_path: Path to extracted Waymo dataset
        """
        self.dataset_path = Path(dataset_path)
        self.scenes = self._index_scenes()

    def _index_scenes(self) -> List[Dict[str, Any]]:
        """Index all scenes in the dataset."""
        scenes = []

        # Look for segment folders
        for segment_dir in sorted(self.dataset_path.glob("*")):
            if segment_dir.is_dir() and segment_dir.name.endswith(".tfrecord"):
                scene_info = self._parse_waymo_scene(segment_dir)
                if scene_info:
                    scenes.append(scene_info)

        return scenes

    def _parse_waymo_scene(self, segment_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a Waymo scene directory."""
        try:
            # Try to load metadata
            metadata_path = segment_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                return {
                    "segment_id": segment_path.name,
                    "path": segment_path,
                    "metadata": metadata,
                }
        except Exception as e:
            warnings.warn(f"Failed to parse Waymo scene {segment_path}: {e}")

        return None

    def get_scene(self, scene_idx: int) -> Optional[Dict[str, Any]]:
        """Get a specific scene by index."""
        if 0 <= scene_idx < len(self.scenes):
            return self.scenes[scene_idx]
        return None

    def get_frame(
        self, scene_idx: int, frame_idx: int, camera: str = "FRONT"
    ) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """Get a specific frame from a scene.

        Args:
            scene_idx: Scene index
            frame_idx: Frame index within scene
            camera: Camera name (FRONT, FRONT_LEFT, etc.)

        Returns:
            Tuple of (image, metadata)
        """
        scene = self.get_scene(scene_idx)
        if not scene:
            return None, None

        # Construct frame path
        frame_path = scene["path"] / f"{camera}" / f"frame_{frame_idx:06d}.png"

        if frame_path.exists():
            try:
                import cv2

                image = cv2.imread(str(frame_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Load calibration
                    calib_path = scene["path"] / "calibration.json"
                    calibration = self._load_calibration(calib_path, camera)

                    metadata = FrameMetadata(
                        timestamp=float(frame_idx) * 0.1,  # Waymo: 10 Hz
                        camera_name=camera,
                        frame_index=frame_idx,
                        image_path=str(frame_path),
                        calibration=calibration,
                    )

                    return image, metadata
            except Exception as e:
                warnings.warn(f"Failed to load frame {frame_path}: {e}")

        return None, None

    def _load_calibration(self, calib_path: Path, camera: str) -> CameraCalibration:
        """Load camera calibration."""
        try:
            if calib_path.exists():
                with open(calib_path) as f:
                    calib_data = json.load(f)

                camera_calib = calib_data.get(camera, {})

                return CameraCalibration(
                    fx=camera_calib.get("fx", 2015.0),
                    fy=camera_calib.get("fy", 2015.0),
                    cx=camera_calib.get("cx", 960.0),
                    cy=camera_calib.get("cy", 600.0),
                    width=camera_calib.get("width", 1920),
                    height=camera_calib.get("height", 1200),
                    distortion=np.array(camera_calib.get("distortion", [])) if "distortion" in camera_calib else None,
                )
        except Exception as e:
            warnings.warn(f"Failed to load calibration: {e}")

        # Return default calibration
        return CameraCalibration(fx=2015.0, fy=2015.0, cx=960.0, cy=600.0, width=1920, height=1200)


class nuScenesDatasetLoader:
    """Loader for nuScenes dataset."""

    def __init__(self, dataset_path: str, version: str = "v1.0-trainval"):
        """Initialize nuScenes loader.

        Args:
            dataset_path: Path to nuScenes dataset root
            version: Dataset version (v1.0-trainval, v1.0-test, etc.)
        """
        self.dataset_path = Path(dataset_path)
        self.version = version
        self.scenes = self._load_scene_index()

    def _load_scene_index(self) -> List[Dict[str, Any]]:
        """Load scene index from metadata."""
        scenes = []

        # Try to load metadata
        metadata_path = self.dataset_path / "v1.0-trainval" / "scenes.json"
        if metadata_path.exists():
            try:
                import json

                with open(metadata_path) as f:
                    metadata = json.load(f)
                    for scene in metadata:
                        scenes.append(
                            {
                                "name": scene.get("name"),
                                "first_sample_token": scene.get("first_sample_token"),
                                "nbr_samples": scene.get("nbr_samples"),
                            }
                        )
            except Exception as e:
                warnings.warn(f"Failed to load nuScenes metadata: {e}")

        return scenes

    def get_scene(self, scene_idx: int) -> Optional[Dict[str, Any]]:
        """Get a specific scene."""
        if 0 <= scene_idx < len(self.scenes):
            return self.scenes[scene_idx]
        return None

    def get_frame(self, scene_idx: int, frame_idx: int, camera: str = "CAM_FRONT") -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """Get a frame from nuScenes."""
        scene = self.get_scene(scene_idx)
        if not scene:
            return None, None

        # Construct frame path
        frame_path = self.dataset_path / "samples" / camera / f"frame_{frame_idx:06d}.jpg"

        if frame_path.exists():
            try:
                import cv2

                image = cv2.imread(str(frame_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # nuScenes standard calibration
                    calibration = CameraCalibration(
                        fx=1266.0,
                        fy=1266.0,
                        cx=816.0,
                        cy=512.0,
                        width=1600,
                        height=900,
                    )

                    metadata = FrameMetadata(
                        timestamp=float(frame_idx) * 0.05,  # nuScenes: 20 Hz
                        camera_name=camera,
                        frame_index=frame_idx,
                        image_path=str(frame_path),
                        calibration=calibration,
                    )

                    return image, metadata
            except Exception as e:
                warnings.warn(f"Failed to load nuScenes frame: {e}")

        return None, None


class KITTIDatasetLoader:
    """Loader for KITTI dataset."""

    def __init__(self, dataset_path: str, split: str = "training"):
        """Initialize KITTI loader.

        Args:
            dataset_path: Path to KITTI dataset root
            split: "training" or "testing"
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.sequences = self._index_sequences()

    def _index_sequences(self) -> List[str]:
        """Index available sequences."""
        sequences = []

        seq_dir = self.dataset_path / self.split / "sequences"
        if seq_dir.exists():
            for seq_path in sorted(seq_dir.glob("*")):
                if seq_path.is_dir():
                    sequences.append(seq_path.name)

        return sequences

    def get_sequence(self, seq_idx: int) -> Optional[str]:
        """Get a specific sequence name."""
        if 0 <= seq_idx < len(self.sequences):
            return self.sequences[seq_idx]
        return None

    def get_frame(self, seq_idx: int, frame_idx: int, camera: int = 0) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """Get a frame from KITTI.

        Args:
            seq_idx: Sequence index
            frame_idx: Frame index within sequence
            camera: Camera ID (0=left, 1=right)

        Returns:
            Tuple of (image, metadata)
        """
        seq = self.get_sequence(seq_idx)
        if not seq:
            return None, None

        # Construct frame path
        frame_path = (
            self.dataset_path
            / self.split
            / "sequences"
            / seq
            / "image_"
            / str(camera)
            / f"{frame_idx:06d}.png"
        )

        if frame_path.exists():
            try:
                import cv2

                image = cv2.imread(str(frame_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Load KITTI calibration
                    calib_path = self.dataset_path / self.split / "calib" / f"{seq}.txt"
                    calibration = self._load_kitti_calib(calib_path, camera)

                    metadata = FrameMetadata(
                        timestamp=float(frame_idx) * 0.1,  # KITTI: 10 Hz
                        camera_name=f"camera_{camera}",
                        frame_index=frame_idx,
                        image_path=str(frame_path),
                        calibration=calibration,
                    )

                    return image, metadata
            except Exception as e:
                warnings.warn(f"Failed to load KITTI frame: {e}")

        return None, None

    def _load_kitti_calib(self, calib_path: Path, camera: int) -> CameraCalibration:
        """Load KITTI calibration for a camera."""
        try:
            if calib_path.exists():
                with open(calib_path) as f:
                    for line in f:
                        if line.startswith(f"P{camera}:"):
                            parts = line.split(":")[1].strip().split()
                            P = np.array([float(x) for x in parts]).reshape(3, 4)

                            return CameraCalibration(
                                fx=P[0, 0],
                                fy=P[1, 1],
                                cx=P[0, 2],
                                cy=P[1, 2],
                                width=1242,
                                height=375,
                            )
        except Exception as e:
            warnings.warn(f"Failed to load KITTI calibration: {e}")

        # Return default KITTI calibration
        return CameraCalibration(
            fx=718.856,
            fy=718.856,
            cx=607.1928,
            cy=185.2157,
            width=1242,
            height=375,
        )
