"""Cylindrical projection for panoramic video stitching.

Mathematical models for projecting images onto a cylinder
and merging them into panoramic panorama strips.
"""

from __future__ import annotations

import numpy as np


def spherical_to_cylindrical(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert spherical coordinates to cylindrical panorama coordinates.

    Args:
        xyz: [batch, H, W, 3] 3D point coordinates

    Returns:
        (u_pan, v_pan): Panoramic coordinates [batch, H, W]
        - u_pan: Horizontal coordinate (wraps around cylinder) [0, 2π]
        - v_pan: Vertical coordinate (on cylinder) [0, 1]
    """
    batch, height, width, _ = xyz.shape

    # Extract coordinates
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    # Compute panoramic angle and height
    u_pan = np.arctan2(x, z)  # Azimuth angle [-π, π]
    u_pan = (u_pan + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

    # Vertical coordinate (height along cylinder)
    v_pan = y / (np.max(np.abs(xyz)) + 1e-6)  # Normalize to approx [-1, 1]
    v_pan = (v_pan + 1.0) / 2.0  # Map to [0, 1]

    return u_pan, v_pan


def project_image_to_cylinder(
    image: np.ndarray,
    intrinsics: dict,
    extrinsics: dict,
    focal_length: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project an image onto cylindrical surface.

    Args:
        image: [H, W, 3] or [H, W] image
        intrinsics: Camera intrinsics {fx, fy, cx, cy}
        extrinsics: Camera pose {yaw, pitch, roll} in degrees
        focal_length: Cylinder focal length (default 1.0)

    Returns:
        (projected, mask): Projected image and valid pixel mask [H, W]
    """
    height, width = image.shape[:2]

    # Create pixel grid
    u_pixels = np.arange(width)
    v_pixels = np.arange(height)
    u_grid, v_grid = np.meshgrid(u_pixels, v_pixels)

    # Normalize to camera space (pinhole model)
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # Back-project to 3D rays
    x_cam = (u_grid - cx) / fx
    y_cam = (v_grid - cy) / fy
    z_cam = np.ones_like(x_cam)  # Normalized depth

    # Apply camera rotation (extrinsics)
    yaw = np.radians(extrinsics.get("yaw_deg", 0.0))
    pitch = np.radians(extrinsics.get("pitch_deg", 0.0))
    roll = np.radians(extrinsics.get("roll_deg", 0.0))

    # Rotation matrix (ZYX convention)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)

    # Yaw rotation
    x_yaw = x_cam * cos_yaw - z_cam * sin_yaw
    y_yaw = y_cam
    z_yaw = x_cam * sin_yaw + z_cam * cos_yaw

    # Pitch rotation
    x_pitch = x_yaw
    y_pitch = y_yaw * cos_pitch - z_yaw * sin_pitch
    z_pitch = y_yaw * sin_pitch + z_yaw * cos_pitch

    # Roll rotation
    x_roll = x_pitch * cos_roll - y_pitch * sin_roll
    y_roll = x_pitch * sin_roll + y_pitch * cos_roll
    z_roll = z_pitch

    # Project onto cylinder (u_pan, v_pan)
    u_pan = np.arctan2(x_roll, z_roll)  # Azimuth
    u_pan = (u_pan + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

    # Vertical (height on cylinder)
    v_pan = y_roll / (np.sqrt(x_roll**2 + z_roll**2) + 1e-6)
    v_pan = np.clip((v_pan + 1.0) / 2.0, 0, 1)  # Map to [0, 1]

    # Create mask for valid pixels (within cylinder bounds)
    mask = (u_pan >= 0) & (u_pan <= 1) & (v_pan >= 0) & (v_pan <= 1)

    return (u_pan, v_pan), mask


def blend_seam_linear(
    left_image: np.ndarray,
    right_image: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    overlap_width: int = 64,
) -> np.ndarray:
    """Blend two images at seam with linear interpolation.

    Args:
        left_image: [H, W, 3] left image
        right_image: [H, W, 3] right image
        left_mask: [H, W] valid pixel mask for left
        right_mask: [H, W] valid pixel mask for right
        overlap_width: Width of overlap region

    Returns:
        Blended image [H, W, 3]
    """
    height, width = left_image.shape[:2]
    blended = np.zeros_like(left_image, dtype=np.float32)

    # Define seam region
    seam_left = width // 2 - overlap_width // 2
    seam_right = width // 2 + overlap_width // 2

    # Non-overlapping left region
    blended[:, :seam_left] = left_image[:, :seam_left]

    # Overlapping region with linear blend
    for x in range(seam_left, seam_right):
        alpha = (x - seam_left) / (seam_right - seam_left)  # 0 -> 1
        blend_weight = 0.5 + 0.5 * np.sin((alpha - 0.5) * np.pi)  # Smooth S-curve

        left_contrib = left_image[:, x] * (1.0 - blend_weight)
        right_contrib = right_image[:, x] * blend_weight

        # Apply masks
        left_valid = left_mask[:, x:x+1]
        right_valid = right_mask[:, x:x+1]

        blended[:, x] = (left_contrib * left_valid + right_contrib * right_valid).astype(
            left_image.dtype
        )

    # Non-overlapping right region
    blended[:, seam_right:] = right_image[:, seam_right:]

    return blended.astype(left_image.dtype)


def create_panorama_grid(
    height: int,
    panorama_width: int,
    num_cameras: int,
    camera_fov: float = 90.0,
) -> np.ndarray:
    """Create grid for mapping cameras to panorama strip.

    Args:
        height: Output panorama height
        panorama_width: Output panorama width
        num_cameras: Number of cameras (determines coverage)
        camera_fov: Field of view of each camera in degrees

    Returns:
        Grid [height, panorama_width, 2] mapping panorama coords to camera coords
    """
    # Panorama spans full 360 degrees
    pan_angles = np.linspace(-np.pi, np.pi, panorama_width)
    cam_heights = np.linspace(-1, 1, height)

    grid = np.zeros((height, panorama_width, 2), dtype=np.float32)

    for y, v in enumerate(cam_heights):
        for x, theta in enumerate(pan_angles):
            grid[y, x, 0] = theta  # Azimuth angle
            grid[y, x, 1] = v  # Height coordinate

    return grid


def compute_panorama_bounds(
    calibrations: dict,
    panorama_height: int = 480,
) -> dict:
    """Compute bounding box for panoramic stitching output.

    Args:
        calibrations: Dictionary of camera calibrations
        panorama_height: Height of output panorama

    Returns:
        Dictionary with panorama dimensions and coverage analysis
    """
    num_cameras = len(calibrations)
    min_width = 480  # Minimum panorama width (1:1 aspect)
    base_width = int(panorama_height * 3.6)  # 360 degrees coverage

    return {
        "num_cameras": num_cameras,
        "height": panorama_height,
        "width": max(min_width, base_width),
        "coverage_degrees": 360.0,
        "estimated_pixel_per_degree": max(min_width, base_width) / 360.0,
    }
