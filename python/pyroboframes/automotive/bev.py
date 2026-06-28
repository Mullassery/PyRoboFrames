"""Bird's-eye-view projection for 3D object detection.

Phase 3 technique: Transform multi-camera images into top-down BEV
representation for 3D perception (FCOS3D, BEVFormer, etc.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class BEVProjector:
    """Project multi-camera images to bird's-eye-view.

    Transforms perspective camera views into top-down BEV coordinates
    for 3D object detection and autonomous driving perception.

    Usage:
        ```python
        projector = BEVProjector(
            camera_calibrations=cals,
            bev_size=(400, 400),
            bev_range=(-50, 50, -25, 25),  # x_min, x_max, y_min, y_max (m)
        )

        bev = projector.project_frames(frames)
        # bev shape: [batch, 400, 400, C]
        ```
    """

    def __init__(
        self,
        camera_calibrations: dict,
        bev_size: tuple[int, int] = (400, 400),
        bev_range: tuple[float, float, float, float] = (-50, 50, -25, 25),
        pixel_size: Optional[float] = None,
    ):
        """Initialize BEV projector.

        Args:
            camera_calibrations: {camera_name -> {fx, fy, cx, cy, width, height}}
            bev_size: (height, width) of output BEV grid
            bev_range: (x_min, x_max, y_min, y_max) in meters
            pixel_size: Size of each BEV pixel in meters (auto-computed if None)
        """
        self.calibrations = camera_calibrations
        self.bev_size = bev_size
        self.bev_range = bev_range

        x_min, x_max, y_min, y_max = bev_range
        bev_h, bev_w = bev_size

        if pixel_size is None:
            # Auto-compute from BEV range
            self.pixel_size_x = (x_max - x_min) / bev_w
            self.pixel_size_y = (y_max - y_min) / bev_h
        else:
            self.pixel_size_x = pixel_size
            self.pixel_size_y = pixel_size

    def image_to_bev(
        self,
        image: np.ndarray,
        camera_name: str,
        camera_height: float = 1.5,
    ) -> np.ndarray:
        """Project single camera image to BEV.

        Args:
            image: [H, W, 3] camera image
            camera_name: Name of camera
            camera_height: Height of camera above ground (m)

        Returns:
            [bev_h, bev_w, 3] BEV image
        """
        if camera_name not in self.calibrations:
            raise KeyError(f"Camera '{camera_name}' not in calibrations")

        cal = self.calibrations[camera_name]
        bev_h, bev_w = self.bev_size
        x_min, x_max, y_min, y_max = self.bev_range

        bev = np.zeros((bev_h, bev_w, 3), dtype=image.dtype)

        # Extract intrinsics
        fx = cal.get("fx", 500.0)
        fy = cal.get("fy", 500.0)
        cx = cal.get("cx", image.shape[1] / 2)
        cy = cal.get("cy", image.shape[0] / 2)

        # For each BEV pixel, back-project to image
        for bev_y in range(bev_h):
            for bev_x in range(bev_w):
                # BEV coordinates to world
                world_x = x_min + bev_x * self.pixel_size_x
                world_y = y_min + bev_y * self.pixel_size_y
                world_z = -camera_height  # Ground plane

                # Project to image plane using pinhole model
                # (simplified: assumes camera looking down, intrinsics only)
                u = (fx * world_x / world_z + cx)
                v = (fy * world_y / world_z + cy)

                u = int(u)
                v = int(v)

                # Bilinear interpolation if valid
                if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                    bev[bev_y, bev_x] = image[v, u]

        return bev

    def frames_to_bev(
        self,
        frames: dict[str, np.ndarray],
        camera_height: float = 1.5,
        fusion_method: str = "max",
    ) -> np.ndarray:
        """Project all camera frames to BEV and fuse.

        Args:
            frames: {camera_name -> [H, W, 3] or [batch, H, W, 3]}
            camera_height: Height above ground (m)
            fusion_method: "max" (max pooling), "mean" (average), or "stack"

        Returns:
            BEV image [bev_h, bev_w, 3] or [bev_h, bev_w, 3*num_cameras] if stack
        """
        if not frames:
            raise ValueError("No frames provided")

        bev_h, bev_w = self.bev_size

        # Project all cameras
        bevs = []
        for cam_name, frame in frames.items():
            if frame.ndim == 4:
                # Batch dimension - process each frame
                frame = frame[0]  # Take first frame for now

            bev_cam = self.image_to_bev(frame, cam_name, camera_height)
            bevs.append(bev_cam)

        if not bevs:
            return np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

        # Fusion
        if fusion_method == "max":
            # Max pooling: brightest pixel wins
            result = np.max(bevs, axis=0)
        elif fusion_method == "mean":
            # Average: smooth blend
            result = np.mean(bevs, axis=0).astype(np.uint8)
        elif fusion_method == "stack":
            # Concatenate all channels
            result = np.concatenate(bevs, axis=-1)
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")

        return result.astype(np.uint8)

    def get_bev_size(self) -> tuple[int, int]:
        """Get BEV grid dimensions."""
        return self.bev_size

    def get_bev_range(self) -> tuple[float, float, float, float]:
        """Get BEV coverage in meters."""
        return self.bev_range

    def __repr__(self) -> str:
        h, w = self.bev_size
        x_min, x_max, y_min, y_max = self.bev_range
        return (
            f"BEVProjector("
            f"size={self.bev_size}, "
            f"range=[{x_min}, {x_max}]×[{y_min}, {y_max}]m, "
            f"cameras={len(self.calibrations)})"
        )


def create_bev_grid(
    bev_size: tuple[int, int],
    bev_range: tuple[float, float, float, float],
) -> np.ndarray:
    """Create coordinate grid for BEV space.

    Args:
        bev_size: (height, width) of grid
        bev_range: (x_min, x_max, y_min, y_max) in meters

    Returns:
        [height, width, 2] grid with (x, y) world coordinates
    """
    h, w = bev_size
    x_min, x_max, y_min, y_max = bev_range

    x = np.linspace(x_min, x_max, w)
    y = np.linspace(y_min, y_max, h)
    X, Y = np.meshgrid(x, y)

    grid = np.stack([X, Y], axis=-1)
    return grid


def warp_bev_to_panorama(
    bev: np.ndarray,
    panorama_height: int = 480,
) -> np.ndarray:
    """Warp BEV back to panoramic strip (inverse of panorama-to-BEV).

    Args:
        bev: [height, width, 3] BEV image
        panorama_height: Target panorama height

    Returns:
        [panorama_height, panorama_width, 3] panoramic image
    """
    bev_h, bev_w = bev.shape[:2]

    # Panorama width is ~3.6x height for 360° coverage
    panorama_w = int(panorama_height * 3.6)
    panorama = np.zeros((panorama_height, panorama_w, 3), dtype=bev.dtype)

    # Simple cylindrical warp: BEV center maps to panorama center
    for pan_y in range(panorama_height):
        for pan_x in range(panorama_w):
            # Panorama coordinates to angle
            angle = (pan_x / panorama_w) * 2 * np.pi - np.pi

            # Map angle to BEV x coordinate
            bev_x = int((angle + np.pi) / (2 * np.pi) * bev_w)
            bev_x = max(0, min(bev_w - 1, bev_x))

            # Map height to BEV y coordinate (front of vehicle)
            bev_y = int((panorama_height - pan_y) / panorama_height * bev_h)
            bev_y = max(0, min(bev_h - 1, bev_y))

            panorama[pan_y, pan_x] = bev[bev_y, bev_x]

    return panorama


def compute_bev_occupancy(
    frames: dict[str, np.ndarray],
    bev_size: tuple[int, int] = (400, 400),
    threshold: int = 128,
) -> np.ndarray:
    """Compute occupancy map from BEV projection.

    Args:
        frames: {camera_name -> [H, W, 3]}
        bev_size: Output size
        threshold: Brightness threshold for occupancy

    Returns:
        [height, width] occupancy map (0=empty, 1=occupied)
    """
    bev_h, bev_w = bev_size
    occupancy = np.zeros((bev_h, bev_w), dtype=np.uint8)

    # Simple occupancy: pixels brighter than threshold
    for frame in frames.values():
        if frame.ndim == 4:
            frame = frame[0]

        # Downsample to BEV size
        from scipy.ndimage import zoom

        frame_bev = zoom(frame, (bev_h / frame.shape[0], bev_w / frame.shape[1], 1))
        brightness = frame_bev.mean(axis=-1)

        occupancy = np.maximum(occupancy, (brightness > threshold).astype(np.uint8))

    return occupancy
