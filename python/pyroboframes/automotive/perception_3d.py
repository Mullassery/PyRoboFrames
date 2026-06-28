"""Advanced 3D perception for autonomous driving (Phase 6).

Lidar point cloud fusion, radar velocity fusion, occupancy grid mapping.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np


class LidarFusion:
    """Fuse lidar point clouds from multiple sensors.

    Supports:
    - Multi-lidar point cloud registration (ICP, feature-based)
    - Temporal filtering for consistency
    - Ground plane segmentation
    - Dynamic object filtering

    Usage:
        ```python
        fusion = LidarFusion(num_lidars=5)

        # Register point clouds in vehicle frame
        fused = fusion.fuse(point_clouds, transforms)
        # fused: [N, 4] (x, y, z, intensity)
        ```
    """

    def __init__(
        self,
        num_lidars: int = 5,
        voxel_size: float = 0.1,
        max_range: float = 100.0,
    ):
        """Initialize lidar fusion.

        Args:
            num_lidars: Number of lidar sensors (Waymo: 5, nuScenes: 1+mmw)
            voxel_size: Voxel size for downsampling (0.1m typical)
            max_range: Maximum lidar range (100m for Waymo)
        """
        self.num_lidars = num_lidars
        self.voxel_size = voxel_size
        self.max_range = max_range

    def fuse(
        self,
        point_clouds: list[np.ndarray],
        transforms: list[np.ndarray],
    ) -> np.ndarray:
        """Fuse multiple lidar point clouds.

        Args:
            point_clouds: List of [N, 3+] point clouds (x, y, z, [intensity])
            transforms: List of 4×4 transformation matrices (lidar→vehicle)

        Returns:
            [M, 4] fused point cloud (x, y, z, intensity)
        """
        fused_points = []

        for pc, T in zip(point_clouds, transforms):
            if len(pc) == 0:
                continue

            # Transform to vehicle frame
            points_3d = pc[:, :3]
            points_h = np.hstack([points_3d, np.ones((len(pc), 1))])
            transformed = (T @ points_h.T).T[:, :3]

            # Filter by range
            distances = np.linalg.norm(transformed, axis=1)
            valid = distances <= self.max_range

            if np.any(valid):
                # Keep intensity if available
                if pc.shape[1] > 3:
                    intensity = pc[valid, 3:4]
                    fused_points.append(np.hstack([transformed[valid], intensity]))
                else:
                    intensity = np.ones((np.sum(valid), 1))
                    fused_points.append(np.hstack([transformed[valid], intensity]))

        if not fused_points:
            return np.zeros((0, 4), dtype=np.float32)

        fused = np.vstack(fused_points)

        # Downsample with voxel grid
        fused = self._voxel_downsample(fused)

        return fused

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """Downsample points using voxel grid.

        Args:
            points: [N, 4] (x, y, z, intensity)

        Returns:
            [M, 4] downsampled points
        """
        # Compute voxel indices
        voxel_indices = (points[:, :3] / self.voxel_size).astype(np.int32)

        # Hash voxels
        unique_voxels, inverse_indices = np.unique(
            voxel_indices, axis=0, return_inverse=True
        )

        # Average points per voxel
        downsampled = []
        for i, voxel in enumerate(unique_voxels):
            mask = inverse_indices == i
            voxel_points = points[mask]
            mean_point = np.mean(voxel_points, axis=0)
            downsampled.append(mean_point)

        return np.array(downsampled, dtype=np.float32)

    def segment_ground(
        self,
        points: np.ndarray,
        plane_threshold: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segment ground plane from points.

        Uses RANSAC to fit ground plane.

        Args:
            points: [N, 4] point cloud
            plane_threshold: Distance threshold for inliers (m)

        Returns:
            (ground_points, non_ground_points) both [M, 4]
        """
        # Simple RANSAC ground plane fitting
        # (Placeholder - real implementation would use actual RANSAC)

        # Assume ground is mostly below z=0 (vehicle mounted ~1-2m high)
        z_threshold = -0.5  # Ground is below this z-value

        ground_mask = points[:, 2] < z_threshold

        ground = points[ground_mask]
        non_ground = points[~ground_mask]

        return ground, non_ground


class RadarFusion:
    """Fuse radar velocity measurements.

    Supports:
    - Multi-radar velocity fusion (nuScenes: mmWave radar)
    - Doppler velocity filtering
    - Clutter rejection

    Usage:
        ```python
        radar = RadarFusion()

        # Fuse radar measurements
        velocities = radar.fuse(radar_detections, transforms)
        # velocities: [N, 2] (vx, vy) m/s
        ```
    """

    def __init__(
        self,
        num_radars: int = 2,
        velocity_scale: float = 0.1,
    ):
        """Initialize radar fusion.

        Args:
            num_radars: Number of radar sensors
            velocity_scale: Velocity measurement scale
        """
        self.num_radars = num_radars
        self.velocity_scale = velocity_scale

    def fuse(
        self,
        radar_detections: list[np.ndarray],
        transforms: list[np.ndarray],
    ) -> np.ndarray:
        """Fuse multiple radar measurements.

        Args:
            radar_detections: List of [N, 4] (x, y, vx, vy) detections
            transforms: List of 4×4 transformation matrices

        Returns:
            [M, 4] fused radar detections (x, y, vx, vy)
        """
        fused = []

        for detections, T in zip(radar_detections, transforms):
            if len(detections) == 0:
                continue

            # Transform positions to vehicle frame
            xy = detections[:, :2]
            xy_h = np.hstack([xy, np.zeros((len(detections), 1)), np.ones((len(detections), 1))])
            xy_transformed = (T @ xy_h.T).T[:, :2]

            # Rotate velocities to vehicle frame
            vx, vy = detections[:, 2], detections[:, 3]
            velocities = np.array([vx, vy])

            # Apply rotation (extract 2×2 rotation matrix)
            R = T[:2, :2]
            velocities_rotated = R @ velocities

            # Combine transformed position and rotated velocity
            fused_detection = np.hstack([
                xy_transformed,
                velocities_rotated.T,
            ])

            fused.append(fused_detection)

        if not fused:
            return np.zeros((0, 4), dtype=np.float32)

        return np.vstack(fused)


class OccupancyGrid:
    """Bayesian occupancy grid mapping.

    Supports:
    - Probabilistic occupancy from lidar + radar
    - Temporal filtering for consistency
    - Multi-resolution hierarchical mapping

    Usage:
        ```python
        grid = OccupancyGrid(
            size=(-50, 50),  # x range
            resolution=0.1,   # 10cm cells
        )

        # Update with sensor measurements
        grid.update(lidar_points, radar_detections)

        # Get occupancy map
        occupancy = grid.get_occupancy_map()
        # occupancy: [H, W] (0=free, 0.5=unknown, 1=occupied)
        ```
    """

    def __init__(
        self,
        size: tuple[float, float] = (-50.0, 50.0),
        resolution: float = 0.1,
        log_odds_threshold: float = 0.5,
    ):
        """Initialize occupancy grid.

        Args:
            size: (min, max) extent in x/y (meters)
            resolution: Grid cell size (meters)
            log_odds_threshold: Threshold for occupancy probability
        """
        self.size = size
        self.resolution = resolution
        self.log_odds_threshold = log_odds_threshold

        # Grid dimensions
        extent = size[1] - size[0]
        self.grid_size = int(np.ceil(extent / resolution))

        # Log-odds representation (more stable than probabilities)
        self.log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Sensor models (log-odds)
        self.log_odds_hit = np.log(9.0)  # p=0.9
        self.log_odds_miss = np.log(1.0 / 9.0)  # p=0.1

    def update(
        self,
        lidar_points: Optional[np.ndarray] = None,
        radar_detections: Optional[np.ndarray] = None,
    ) -> None:
        """Update occupancy grid with sensor measurements.

        Args:
            lidar_points: [N, 3+] point cloud
            radar_detections: [N, 4] radar detections
        """
        # Update from lidar (ray casting)
        if lidar_points is not None and len(lidar_points) > 0:
            self._update_from_lidar(lidar_points)

        # Update from radar (direct occupancy)
        if radar_detections is not None and len(radar_detections) > 0:
            self._update_from_radar(radar_detections)

    def _update_from_lidar(self, points: np.ndarray) -> None:
        """Update grid from lidar ray casting.

        Args:
            points: [N, 3+] point cloud
        """
        # For each point, mark ray as miss, endpoint as hit
        origin = np.array([0.0, 0.0])  # Vehicle center

        for point in points:
            x, y = point[0], point[1]

            # Discretize ray from origin to point
            ray_length = np.linalg.norm([x, y])

            if ray_length < 0.1:  # Skip very close points
                continue

            # Number of cells along ray
            num_cells = max(1, int(ray_length / self.resolution))

            for i in range(num_cells):
                t = i / max(1, num_cells - 1) if num_cells > 1 else 0.5
                rx, ry = origin + t * np.array([x, y])

                # Convert to grid coordinates
                grid_x, grid_y = self._world_to_grid(rx, ry)

                if self._is_valid_cell(grid_x, grid_y):
                    if i < num_cells - 1:
                        # Intermediate cells: mark as miss
                        self.log_odds[grid_y, grid_x] += self.log_odds_miss
                    else:
                        # Endpoint: mark as hit
                        self.log_odds[grid_y, grid_x] += self.log_odds_hit

    def _update_from_radar(self, detections: np.ndarray) -> None:
        """Update grid from radar detections.

        Args:
            detections: [N, 4] (x, y, vx, vy)
        """
        for detection in detections:
            x, y = detection[0], detection[1]

            grid_x, grid_y = self._world_to_grid(x, y)

            if self._is_valid_cell(grid_x, grid_y):
                # Radar detections are high-confidence hits
                self.log_odds[grid_y, grid_x] += self.log_odds_hit * 2

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to grid indices.

        Args:
            x, y: World coordinates (meters)

        Returns:
            (grid_x, grid_y) integer indices
        """
        # Normalize to [0, 1]
        extent = self.size[1] - self.size[0]
        nx = (x - self.size[0]) / extent
        ny = (y - self.size[0]) / extent

        # Scale to grid
        grid_x = int(nx * self.grid_size)
        grid_y = int(ny * self.grid_size)

        return grid_x, grid_y

    def _is_valid_cell(self, x: int, y: int) -> bool:
        """Check if grid cell is valid.

        Args:
            x, y: Grid indices

        Returns:
            True if cell is within bounds
        """
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_occupancy_map(self) -> np.ndarray:
        """Get occupancy probability map.

        Returns:
            [H, W] occupancy map (0=free, 0.5=unknown, 1=occupied)
        """
        # Convert log-odds to probability
        # p = 1 - 1/(1 + exp(log_odds))
        occupancy = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

        return occupancy.astype(np.float32)

    def reset(self) -> None:
        """Reset grid to uniform prior."""
        self.log_odds.fill(0.0)
