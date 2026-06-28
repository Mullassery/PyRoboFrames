"""nuScenes dataset JSON parsing utilities.

Handles efficient parsing of nuScenes' JSON metadata format.
Supports camera images, lidar, radar, and 3D annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


class NuScenesMetadataManager:
    """Manage nuScenes JSON metadata efficiently.

    Provides indexed access to scenes, samples, sensor data, and calibrations.
    """

    def __init__(self, root_dir: Path):
        """Initialize metadata manager.

        Args:
            root_dir: Path to nuScenes root (contains v1.0-trainval/)
        """
        self.root = Path(root_dir)
        self.scenes = []
        self.samples = []
        self.sample_data = {}
        self.sensor_calibrations = {}
        self.annotations = []

        self._load_all_metadata()

    def _load_all_metadata(self):
        """Load all JSON metadata files."""
        # Load scenes
        scenes_file = self.root / "scenes.json"
        if scenes_file.exists():
            with open(scenes_file) as f:
                self.scenes = json.load(f)

        # Load samples
        samples_file = self.root / "samples.json"
        if samples_file.exists():
            with open(samples_file) as f:
                self.samples = json.load(f)

        # Load sample_data (create indexed lookup)
        sample_data_file = self.root / "sample_data.json"
        if sample_data_file.exists():
            with open(sample_data_file) as f:
                sample_data_list = json.load(f)
                for sd in sample_data_list:
                    self.sample_data[sd["token"]] = sd

        # Load calibrations (create indexed lookup)
        calibrations_file = self.root / "calibrated_sensor_record.json"
        if calibrations_file.exists():
            with open(calibrations_file) as f:
                calibrations_list = json.load(f)
                for calib in calibrations_list:
                    self.sensor_calibrations[calib["token"]] = calib

        # Load annotations
        annotations_file = self.root / "sample_annotation.json"
        if annotations_file.exists():
            with open(annotations_file) as f:
                self.annotations = json.load(f)

    def get_sample_data_files(
        self,
        sample_token: str,
        sensor_modality: str = "camera",
    ) -> Dict[str, str]:
        """Get data file paths for a sample.

        Args:
            sample_token: Sample token from samples.json
            sensor_modality: "camera", "lidar", or "radar"

        Returns:
            {channel_name -> file_path}
        """
        files = {}

        # Find sample
        sample = None
        for s in self.samples:
            if s["token"] == sample_token:
                sample = s
                break

        if not sample:
            return files

        # Find sample data for each sensor
        for sd_token, sd in self.sample_data.items():
            if sd["sample_token"] != sample_token:
                continue

            # Get sensor calibration
            sensor_calib = self.sensor_calibrations.get(sd["calibrated_sensor_token"], {})
            sensor_name = sensor_calib.get("sensor_channel", "")

            # Filter by modality
            if sensor_modality == "camera" and "CAM" in sensor_name:
                file_path = self.root / sd["filename"]
                files[sensor_name] = str(file_path)
            elif sensor_modality == "lidar" and "LIDAR" in sensor_name:
                file_path = self.root / sd["filename"]
                files[sensor_name] = str(file_path)
            elif sensor_modality == "radar" and "RADAR" in sensor_name:
                file_path = self.root / sd["filename"]
                files[sensor_name] = str(file_path)

        return files

    def get_calibration(self, sensor_token: str) -> Dict[str, Any]:
        """Get sensor calibration parameters.

        Args:
            sensor_token: Calibrated sensor token

        Returns:
            Calibration dict with intrinsics, extrinsics, etc.
        """
        calib = self.sensor_calibrations.get(sensor_token, {})

        return {
            "sensor_channel": calib.get("sensor_channel"),
            "camera_intrinsic": calib.get("camera_intrinsic"),
            "translation": calib.get("translation"),
            "rotation": calib.get("rotation"),
        }

    def get_annotations_for_sample(self, sample_token: str) -> List[Dict[str, Any]]:
        """Get 3D object annotations for a sample.

        Args:
            sample_token: Sample token

        Returns:
            List of annotation dicts (3D bboxes)
        """
        return [a for a in self.annotations if a["sample_token"] == sample_token]


def load_camera_image(file_path: str) -> np.ndarray:
    """Load camera image from nuScenes.

    Args:
        file_path: Path to image file (JPEG)

    Returns:
        [H, W, 3] uint8 RGB image
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Image loading requires: pip install pillow")

    img = Image.open(file_path).convert("RGB")

    return np.array(img, dtype=np.uint8)


def load_lidar_sweep(file_path: str) -> np.ndarray:
    """Load lidar point cloud from nuScenes.

    Args:
        file_path: Path to .pcd file (Point Cloud Data format)

    Returns:
        [N, 5] point cloud (x, y, z, intensity, ring)
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Lidar loading requires: pip install open3d")

    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)

    # If colors available, use as intensity
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)
        intensity = np.mean(colors, axis=1, keepdims=True)
        points = np.hstack([points, intensity])
    else:
        # Default intensity
        intensity = np.ones((len(points), 1), dtype=np.float32)
        points = np.hstack([points, intensity])

    # Add ring index (placeholder - would need to track ring from original data)
    ring = np.zeros((len(points), 1), dtype=np.float32)
    points = np.hstack([points, ring])

    return points


def load_radar_detections(file_path: str) -> np.ndarray:
    """Load radar detections from nuScenes.

    Args:
        file_path: Path to radar CSV file

    Returns:
        [N, 4] detections (x, y, vx, vy) in m and m/s
    """
    try:
        import csv
    except ImportError:
        raise ImportError("CSV parsing requires: pip install pytz")

    detections = []

    try:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row.get("x", 0))
                y = float(row.get("y", 0))
                vx = float(row.get("vx", 0))
                vy = float(row.get("vy", 0))

                detections.append([x, y, vx, vy])

    except Exception as e:
        print(f"Warning: Error reading radar file {file_path}: {e}")

    if len(detections) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    return np.array(detections, dtype=np.float32)


def parse_3d_bbox(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Parse 3D bounding box from nuScenes annotation.

    Args:
        annotation: Annotation dict from sample_annotation.json

    Returns:
        Parsed bbox dict with:
        - category: object class
        - center: [x, y, z] in ego frame
        - size: [width, length, height]
        - rotation: [w, x, y, z] quaternion
        - velocity: [vx, vy] if available
    """
    return {
        "token": annotation.get("token"),
        "category": annotation.get("category_name"),
        "center": annotation.get("translation"),
        "size": annotation.get("size"),
        "rotation": annotation.get("rotation"),
        "velocity": annotation.get("velocity"),
        "instance_token": annotation.get("instance_token"),
    }


def get_sensor_camera_intrinsics(
    calibration: Dict[str, Any],
) -> np.ndarray:
    """Extract camera intrinsics from calibration.

    Args:
        calibration: Calibration dict

    Returns:
        [3, 3] camera intrinsic matrix
    """
    intrinsics = calibration.get("camera_intrinsic")

    if intrinsics:
        return np.array(intrinsics, dtype=np.float32)

    # Default nuScenes intrinsics
    return np.array(
        [
            [1266.417, 0, 816.2670],
            [0, 1266.417, 491.2070],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def get_sensor_pose(
    calibration: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Get sensor pose (translation + rotation) from calibration.

    Args:
        calibration: Calibration dict with translation and rotation

    Returns:
        (translation [3], rotation matrix [3, 3])
    """
    translation = np.array(
        calibration.get("translation", [0, 0, 0]),
        dtype=np.float32,
    )

    # Rotation as list [x, y, z, w] (quaternion)
    rotation_quat = np.array(
        calibration.get("rotation", [0, 0, 0, 1]),
        dtype=np.float32,
    )

    # Convert quaternion to rotation matrix
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_quat(rotation_quat).as_matrix().astype(np.float32)

    return translation, rotation
