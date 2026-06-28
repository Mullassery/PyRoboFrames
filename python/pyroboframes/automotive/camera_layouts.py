"""Camera layout presets for autonomous driving datasets.

Defines standard camera configurations for major AV datasets:
- Waymo Open Dataset: 5 cameras (front, front-left, front-right, side-left, side-right)
- nuScenes: 6 cameras (front, front-left, front-right, back-left, back-right, back)
- KITTI: 4 cameras (stereo + grayscale pair)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraLayout:
    """Configuration for a multi-camera vehicle setup."""

    name: str
    cameras: dict[str, dict]  # camera_id -> {yaw_deg, pitch_deg, roll_deg, fx, fy, cx, cy}

    def __post_init__(self):
        """Validate camera layout."""
        if not self.cameras:
            raise ValueError("Camera layout must have at least one camera")

    def __repr__(self) -> str:
        return f"CameraLayout(name='{self.name}', cameras={len(self.cameras)})"


CAMERA_LAYOUTS = {
    "waymo": CameraLayout(
        name="waymo",
        cameras={
            "FRONT": {
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 2015.0,
                "fy": 2015.0,
                "cx": 640.0,
                "cy": 360.0,
                "width": 1280,
                "height": 720,
            },
            "FRONT_LEFT": {
                "yaw_deg": 45.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 2015.0,
                "fy": 2015.0,
                "cx": 640.0,
                "cy": 360.0,
                "width": 1280,
                "height": 720,
            },
            "FRONT_RIGHT": {
                "yaw_deg": -45.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 2015.0,
                "fy": 2015.0,
                "cx": 640.0,
                "cy": 360.0,
                "width": 1280,
                "height": 720,
            },
            "SIDE_LEFT": {
                "yaw_deg": 90.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 2015.0,
                "fy": 2015.0,
                "cx": 640.0,
                "cy": 360.0,
                "width": 1280,
                "height": 720,
            },
            "SIDE_RIGHT": {
                "yaw_deg": -90.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 2015.0,
                "fy": 2015.0,
                "cx": 640.0,
                "cy": 360.0,
                "width": 1280,
                "height": 720,
            },
        },
    ),
    "nuscenes": CameraLayout(
        name="nuscenes",
        cameras={
            "CAM_FRONT": {
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 1266.0,
                "fy": 1266.0,
                "cx": 816.0,
                "cy": 612.0,
                "width": 1600,
                "height": 900,
            },
            "CAM_FRONT_LEFT": {
                "yaw_deg": 60.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 1266.0,
                "fy": 1266.0,
                "cx": 816.0,
                "cy": 612.0,
                "width": 1600,
                "height": 900,
            },
            "CAM_FRONT_RIGHT": {
                "yaw_deg": -60.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 1266.0,
                "fy": 1266.0,
                "cx": 816.0,
                "cy": 612.0,
                "width": 1600,
                "height": 900,
            },
            "CAM_BACK_LEFT": {
                "yaw_deg": 120.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 1266.0,
                "fy": 1266.0,
                "cx": 816.0,
                "cy": 612.0,
                "width": 1600,
                "height": 900,
            },
            "CAM_BACK_RIGHT": {
                "yaw_deg": -120.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 1266.0,
                "fy": 1266.0,
                "cx": 816.0,
                "cy": 612.0,
                "width": 1600,
                "height": 900,
            },
            "CAM_BACK": {
                "yaw_deg": 180.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 1266.0,
                "fy": 1266.0,
                "cx": 816.0,
                "cy": 612.0,
                "width": 1600,
                "height": 900,
            },
        },
    ),
    "kitti": CameraLayout(
        name="kitti",
        cameras={
            "CAM_LEFT": {
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 718.856,
                "fy": 718.856,
                "cx": 607.1928,
                "cy": 185.2157,
                "width": 1242,
                "height": 375,
            },
            "CAM_RIGHT": {
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fx": 718.856,
                "fy": 718.856,
                "cx": 607.1928,
                "cy": 185.2157,
                "width": 1242,
                "height": 375,
                "baseline_m": 0.54,  # Stereo baseline
            },
        },
    ),
}


def get_waymo_layout() -> CameraLayout:
    """Get Waymo Open Dataset camera layout (5 cameras)."""
    return CAMERA_LAYOUTS["waymo"]


def get_nuscenes_layout() -> CameraLayout:
    """Get nuScenes camera layout (6 cameras)."""
    return CAMERA_LAYOUTS["nuscenes"]


def get_kitti_layout() -> CameraLayout:
    """Get KITTI camera layout (stereo pair)."""
    return CAMERA_LAYOUTS["kitti"]
