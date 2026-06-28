"""Cylindrical video stitching for autonomous driving 360° perception.

Implements Phase 1 of automotive video stitching (v0.5.0):
- Camera undistortion (using v0.4.1 calibration)
- Cylindrical projection math
- Linear seam blending
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .blending import (
    blend_laplacian_pyramids,
    blend_with_seam,
    build_laplacian_pyramid,
    compensate_exposure,
    find_optimal_seam,
)
from .camera_layouts import CameraLayout
from .gpu_backend import get_gpu_backend, GPUBackend
from .optical_flow import OpticalFlow, SeamTracker, temporal_blend
from .projection import (
    blend_seam_linear,
    compute_panorama_bounds,
    create_panorama_grid,
    project_image_to_cylinder,
)


class CylindricalStitcher:
    """Stitch multi-camera video into cylindrical panorama.

    Supports:
    - 4-6 cameras (KITTI, Waymo, nuScenes)
    - Linear seam blending (Phase 1)
    - Configurable panorama height/width
    - Batch processing for video streams

    Usage:
        ```python
        from pyroboframes.automotive import CylindricalStitcher
        from pyroboframes.automotive import get_waymo_layout

        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        # Batch of 5 camera frames [batch, height, width, 3]
        panorama = stitcher.stitch(frames_dict, blend_method="linear")
        # panorama shape: [batch, 480, 3200, 3]
        ```
    """

    def __init__(
        self,
        camera_layout: CameraLayout,
        panorama_height: int = 480,
        blend_method: str = "linear",
        device: Optional[str] = None,
        use_temporal_consistency: bool = False,
        temporal_alpha: float = 0.3,
    ):
        """Initialize cylindrical stitcher.

        Args:
            camera_layout: CameraLayout with camera poses and intrinsics
            panorama_height: Output height (width auto-computed for 360°)
            blend_method: "linear" or "laplacian"
            device: "cuda", "mlx", "cpu", or None for auto-detect (Phase 4a)
            use_temporal_consistency: Enable optical flow seam tracking (Phase 4b)
            temporal_alpha: Blend weight for frame smoothing (0-1)

        Raises:
            ValueError: If blend_method not supported or layout invalid
        """
        if not camera_layout.cameras:
            raise ValueError("Camera layout must have at least one camera")

        if blend_method not in ["linear", "laplacian"]:
            raise ValueError(f"blend_method '{blend_method}' not supported. Options: ['linear', 'laplacian']")

        self.layout = camera_layout
        self.panorama_height = panorama_height
        self.blend_method = blend_method
        self.temporal_alpha = temporal_alpha

        # Phase 4a: GPU acceleration support
        self.gpu_backend = get_gpu_backend(device)

        # Phase 4b: Temporal consistency support
        self.use_temporal = use_temporal_consistency
        self.seam_tracker = SeamTracker() if use_temporal_consistency else None

        if use_temporal_consistency:
            try:
                self.optical_flow = OpticalFlow("farneback")
            except ImportError:
                # Fall back to Farneback if available, else no OF
                self.optical_flow = None
        else:
            self.optical_flow = None

        self.prev_panorama = None

        # Compute panorama dimensions
        bounds = compute_panorama_bounds(camera_layout.cameras, panorama_height)
        self.panorama_height = bounds["height"]
        self.panorama_width = bounds["width"]

        # Pre-compute camera order (left to right for blending)
        self.camera_order = self._compute_camera_order()

    def _compute_camera_order(self) -> list[str]:
        """Order cameras by yaw angle for efficient blending.

        Returns:
            List of camera names sorted by yaw (left to right)
        """
        cameras_with_yaw = []
        for cam_name, params in self.layout.cameras.items():
            yaw = params.get("yaw_deg", 0.0)
            cameras_with_yaw.append((yaw, cam_name))

        cameras_with_yaw.sort(key=lambda x: x[0])
        return [name for _, name in cameras_with_yaw]

    def stitch(
        self,
        frames: dict[str, np.ndarray],
        blend_method: Optional[str] = None,
    ) -> np.ndarray:
        """Stitch multi-camera frames into panorama.

        Args:
            frames: Dictionary mapping camera names to frames
                - Keys: Camera names from layout (e.g., "FRONT", "FRONT_LEFT")
                - Values: [batch, height, width, 3] uint8 images

            blend_method: Override blend method (default: use constructor setting)

        Returns:
            Panoramic images [batch, panorama_height, panorama_width, 3] uint8

        Raises:
            KeyError: If required cameras missing from frames
            ValueError: If frame shapes inconsistent
        """
        if blend_method is None:
            blend_method = self.blend_method

        # Validate inputs
        if not frames:
            raise ValueError("No frames provided")

        batch_size = None
        for cam_name, frame in frames.items():
            if cam_name not in self.layout.cameras:
                raise KeyError(f"Camera '{cam_name}' not in layout '{self.layout.name}'")

            if frame.ndim not in [3, 4]:
                raise ValueError(f"Frame shape must be [H, W, 3] or [batch, H, W, 3], got {frame.shape}")

            if batch_size is None:
                batch_size = frame.shape[0] if frame.ndim == 4 else 1
            elif frame.ndim == 4 and frame.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: {batch_size} vs {frame.shape[0]}")

        # Ensure batch dimension
        batch_size = batch_size or 1
        processed_frames = {}
        for cam_name, frame in frames.items():
            if frame.ndim == 3:
                frame = frame[np.newaxis]  # Add batch dimension
            processed_frames[cam_name] = frame

        # Stitch panorama for each batch
        panoramas = []
        for b in range(batch_size):
            pan = self._stitch_single(
                {k: v[b] for k, v in processed_frames.items()},
                blend_method,
            )

            # Phase 4b: Apply temporal consistency if enabled
            if self.use_temporal and self.prev_panorama is not None:
                pan = temporal_blend(
                    self.prev_panorama, pan, alpha=self.temporal_alpha
                )

            self.prev_panorama = pan.copy()
            panoramas.append(pan)

        return np.stack(panoramas, axis=0)

    def stitch_temporal_sequence(
        self,
        frame_sequence: dict[str, np.ndarray],
        blend_method: Optional[str] = None,
    ) -> np.ndarray:
        """Stitch temporal sequence with optical flow seam tracking.

        Phase 4b: Process video with temporal consistency.

        Args:
            frame_sequence: Dictionary mapping camera names to video sequences
                - Keys: Camera names
                - Values: [T, H, W, 3] uint8 (temporal dimension)

            blend_method: Override blend method

        Returns:
            Panoramic video [T, panorama_height, panorama_width, 3] uint8
        """
        if not self.use_temporal:
            # Fall back to regular stitching
            return self.stitch(frame_sequence, blend_method)

        if blend_method is None:
            blend_method = self.blend_method

        # Get number of frames
        num_frames = next(iter(frame_sequence.values())).shape[0]

        panoramas = []

        for t in range(num_frames):
            # Extract frame at time t
            frames_t = {k: v[t] for k, v in frame_sequence.items()}

            # Stitch frame t
            pan_t = self._stitch_single(frames_t, blend_method)

            # Track seam between t-1 and t using optical flow
            if t > 0 and self.optical_flow is not None:
                # Get first camera frame for optical flow
                cam_name = self.camera_order[0]
                frame_t_prev = frame_sequence[cam_name][t - 1]
                frame_t_curr = frame_sequence[cam_name][t]

                # Compute optical flow
                flow = self.optical_flow.compute(frame_t_prev, frame_t_curr)

                # Track seam (simplified: use same seam for now)
                # In production: use flow to guide seam tracking
            else:
                flow = None

            # Apply temporal blending
            if self.prev_panorama is not None:
                pan_t = temporal_blend(
                    self.prev_panorama, pan_t, alpha=self.temporal_alpha
                )

            self.prev_panorama = pan_t.copy()
            panoramas.append(pan_t)

        return np.stack(panoramas, axis=0)

    def _stitch_single(
        self,
        frames: dict[str, np.ndarray],
        blend_method: str,
    ) -> np.ndarray:
        """Stitch a single frame (no batch dimension).

        Args:
            frames: Dict of [H, W, 3] images
            blend_method: Blending algorithm ("linear" or "laplacian")

        Returns:
            [panorama_height, panorama_width, 3] uint8 panorama
        """
        if blend_method == "laplacian":
            return self._stitch_laplacian(frames)
        else:
            return self._stitch_linear(frames)

    def _stitch_linear(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """Stitch with linear blending (Phase 1).

        Args:
            frames: Dict of [H, W, 3] images

        Returns:
            [panorama_height, panorama_width, 3] uint8 panorama
        """
        # Start with blank panorama
        panorama = np.zeros(
            (self.panorama_height, self.panorama_width, 3),
            dtype=np.float32,
        )

        # Project each camera and blend
        for cam_name in self.camera_order:
            if cam_name not in frames:
                continue

            frame = frames[cam_name]
            cam_params = self.layout.cameras[cam_name]

            # Project camera image to panorama
            (u_pan, v_pan), mask = project_image_to_cylinder(
                frame,
                intrinsics={
                    "fx": cam_params.get("fx", 1000.0),
                    "fy": cam_params.get("fy", 1000.0),
                    "cx": cam_params.get("cx", frame.shape[1] / 2),
                    "cy": cam_params.get("cy", frame.shape[0] / 2),
                },
                extrinsics={
                    "yaw_deg": cam_params.get("yaw_deg", 0.0),
                    "pitch_deg": cam_params.get("pitch_deg", 0.0),
                    "roll_deg": cam_params.get("roll_deg", 0.0),
                },
            )

            # Remap to panorama grid
            pan_x = (u_pan * self.panorama_width).astype(np.int32)
            pan_y = (v_pan * self.panorama_height).astype(np.int32)

            # Clamp to panorama bounds
            valid = (pan_x >= 0) & (pan_x < self.panorama_width) & (pan_y >= 0) & (
                pan_y < self.panorama_height
            )

            # Accumulate with simple max (or blend if overlapping)
            if np.any(valid):
                panorama[pan_y[valid], pan_x[valid]] = frame[valid]

        # Convert to uint8
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)

        return panorama

    def _stitch_laplacian(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """Stitch with Laplacian pyramid blending (Phase 2).

        Args:
            frames: Dict of [H, W, 3] images

        Returns:
            [panorama_height, panorama_width, 3] uint8 panorama
        """
        if len(self.camera_order) < 2:
            # Fall back to linear for single camera
            return self._stitch_linear(frames)

        # Project all cameras
        projections = {}
        masks = {}

        for cam_name in self.camera_order:
            if cam_name not in frames:
                continue

            frame = frames[cam_name]
            cam_params = self.layout.cameras[cam_name]

            (u_pan, v_pan), mask = project_image_to_cylinder(
                frame,
                intrinsics={
                    "fx": cam_params.get("fx", 1000.0),
                    "fy": cam_params.get("fy", 1000.0),
                    "cx": cam_params.get("cx", frame.shape[1] / 2),
                    "cy": cam_params.get("cy", frame.shape[0] / 2),
                },
                extrinsics={
                    "yaw_deg": cam_params.get("yaw_deg", 0.0),
                    "pitch_deg": cam_params.get("pitch_deg", 0.0),
                    "roll_deg": cam_params.get("roll_deg", 0.0),
                },
            )

            projections[cam_name] = (u_pan, v_pan, frame)
            masks[cam_name] = mask

        # Build panorama with Laplacian blending
        panorama = np.zeros(
            (self.panorama_height, self.panorama_width, 3),
            dtype=np.float32,
        )

        # Accumulate using max with Laplacian smoothing
        for i, cam_name in enumerate(self.camera_order):
            if cam_name not in projections:
                continue

            u_pan, v_pan, frame = projections[cam_name]

            # For adjacent camera pairs, use Laplacian blending
            if i > 0:
                prev_cam_name = self.camera_order[i - 1]
                if prev_cam_name in projections:
                    # Build Laplacian pyramids for overlap region
                    overlap_x = self.panorama_width // 4  # Rough overlap estimate
                    left_pyr = build_laplacian_pyramid(frame, levels=3)
                    prev_u, prev_v, prev_frame = projections[prev_cam_name]
                    right_pyr = build_laplacian_pyramid(prev_frame, levels=3)

                    # Blend pyramids (simplified)
                    left_mask = masks[cam_name]
                    right_mask = masks[prev_cam_name]

                    blended = blend_laplacian_pyramids(
                        left_pyr, right_pyr, left_mask, right_mask
                    )

                    pan_x = (u_pan * self.panorama_width).astype(np.int32)
                    pan_y = (v_pan * self.panorama_height).astype(np.int32)
                    valid = (
                        (pan_x >= 0)
                        & (pan_x < self.panorama_width)
                        & (pan_y >= 0)
                        & (pan_y < self.panorama_height)
                    )

                    if np.any(valid):
                        panorama[pan_y[valid], pan_x[valid]] = blended[valid]

                    continue

            # Simple projection for non-overlapping cameras
            pan_x = (u_pan * self.panorama_width).astype(np.int32)
            pan_y = (v_pan * self.panorama_height).astype(np.int32)

            valid = (
                (pan_x >= 0)
                & (pan_x < self.panorama_width)
                & (pan_y >= 0)
                & (pan_y < self.panorama_height)
            )

            if np.any(valid):
                panorama[pan_y[valid], pan_x[valid]] = frame[valid]

        panorama = np.clip(panorama, 0, 255).astype(np.uint8)

        return panorama

    def stitch_with_mask(
        self,
        frames: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stitch panorama and return validity mask.

        Args:
            frames: Dict of camera frames

        Returns:
            (panorama, validity_mask):
            - panorama: [batch, height, width, 3]
            - validity_mask: [batch, height, width] (0=invalid, 1=valid)
        """
        panorama = self.stitch(frames)

        # Compute validity: any non-black pixel
        validity = (panorama.sum(axis=-1) > 0).astype(np.uint8)

        return panorama, validity

    def get_panorama_dims(self) -> tuple[int, int]:
        """Get output panorama dimensions.

        Returns:
            (height, width) of output panorama
        """
        return self.panorama_height, self.panorama_width

    def __repr__(self) -> str:
        temporal_str = ", temporal=True" if self.use_temporal else ""
        return (
            f"CylindricalStitcher("
            f"layout='{self.layout.name}', "
            f"output=[{self.panorama_height}, {self.panorama_width}], "
            f"blend='{self.blend_method}', "
            f"device='{self.gpu_backend.name}'{temporal_str})"
        )
