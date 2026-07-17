"""Occupancy grid mapping and 3D perception for autonomous driving."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class OccupancyGridConfig:
    """Configuration for occupancy grid."""

    size_x: float = 100.0  # Grid size in X (meters)
    size_y: float = 100.0  # Grid size in Y (meters)
    resolution: float = 0.1  # Resolution (meters per cell)
    height_min: float = -1.0  # Minimum height (meters)
    height_max: float = 3.0  # Maximum height (meters)


class OccupancyGrid:
    """Occupancy grid for 3D perception."""

    def __init__(self, config: OccupancyGridConfig):
        """Initialize occupancy grid.

        Args:
            config: Grid configuration
        """
        self.config = config
        self.grid_size_x = int(config.size_x / config.resolution)
        self.grid_size_y = int(config.size_y / config.resolution)

        # Occupancy grid: 0=free, 1=occupied, 0.5=unknown
        self.occupancy = np.full((self.grid_size_y, self.grid_size_x), 0.5, dtype=np.float32)
        self.height_map = np.zeros((self.grid_size_y, self.grid_size_x), dtype=np.float32)
        self.variance = np.ones((self.grid_size_y, self.grid_size_x), dtype=np.float32)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        grid_x = int((x + self.config.size_x / 2) / self.config.resolution)
        grid_y = int((y + self.config.size_y / 2) / self.config.resolution)

        grid_x = np.clip(grid_x, 0, self.grid_size_x - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size_y - 1)

        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates."""
        x = grid_x * self.config.resolution - self.config.size_x / 2
        y = grid_y * self.config.resolution - self.config.size_y / 2
        return x, y

    def add_point_cloud(self, points: np.ndarray, heights: Optional[np.ndarray] = None):
        """Add a point cloud to the occupancy grid.

        Args:
            points: [N, 2] XY coordinates (world frame)
            heights: [N] heights (optional), or None to mark as occupied
        """
        if points.shape[0] == 0:
            return

        for i, (x, y) in enumerate(points):
            gx, gy = self.world_to_grid(x, y)

            # Bresenham's line algorithm to trace ray from origin
            self._trace_ray(0, 0, gx, gy, occupied=True)

            # Mark endpoint as occupied
            if 0 <= gx < self.grid_size_x and 0 <= gy < self.grid_size_y:
                h = heights[i] if heights is not None else 0.0
                self._update_cell(gx, gy, occupied=True, height=h)

    def add_bounding_box(self, bbox_3d: Dict[str, float]):
        """Add a 3D bounding box to occupancy grid.

        Args:
            bbox_3d: Dict with keys "x", "y", "width", "length", "height"
        """
        x = bbox_3d.get("x", 0.0)
        y = bbox_3d.get("y", 0.0)
        width = bbox_3d.get("width", 1.0)
        length = bbox_3d.get("length", 1.0)
        height = bbox_3d.get("height", 1.0)

        # Generate box corners
        half_w = width / 2
        half_l = length / 2

        corners = [
            (x - half_l, y - half_w),
            (x - half_l, y + half_w),
            (x + half_l, y + half_w),
            (x + half_l, y - half_w),
        ]

        # Fill box cells
        for i in range(len(corners)):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % len(corners)]
            self._bresenham_line(x1, y1, x2, y2, occupied=True, height=height)

    def dilate(self, kernel_size: int = 3):
        """Apply morphological dilation."""
        from scipy import ndimage

        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        self.occupancy = ndimage.binary_dilation(self.occupancy > 0.5, structure=kernel).astype(np.float32)

    def erode(self, kernel_size: int = 3):
        """Apply morphological erosion."""
        from scipy import ndimage

        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        self.occupancy = ndimage.binary_erosion(self.occupancy > 0.5, structure=kernel).astype(np.float32)

    def get_free_space_mask(self) -> np.ndarray:
        """Get binary mask of free space."""
        return (self.occupancy < 0.5).astype(np.uint8)

    def get_occupied_cells(self) -> List[Tuple[int, int]]:
        """Get list of occupied grid cells."""
        y_indices, x_indices = np.where(self.occupancy > 0.5)
        return list(zip(x_indices, y_indices))

    def _update_cell(self, gx: int, gy: int, occupied: bool, height: float = 0.0):
        """Update a single grid cell."""
        if not (0 <= gx < self.grid_size_x and 0 <= gy < self.grid_size_y):
            return

        if occupied:
            self.occupancy[gy, gx] = 0.9
            self.height_map[gy, gx] = height
            self.variance[gy, gx] *= 0.9  # Reduce uncertainty
        else:
            self.occupancy[gy, gx] = 0.1
            self.variance[gy, gx] *= 0.95

    def _trace_ray(self, x1: int, y1: int, x2: int, y2: int, occupied: bool):
        """Trace a ray using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            # Mark as free space along the ray (except endpoint)
            if (x1, y1) != (x2, y2):
                self._update_cell(x1, y1, occupied=False)

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

    def _bresenham_line(self, x1: float, y1: float, x2: float, y2: float, occupied: bool = True, height: float = 0.0):
        """Draw a line using Bresenham's algorithm in world coordinates."""
        gx1, gy1 = self.world_to_grid(x1, y1)
        gx2, gy2 = self.world_to_grid(x2, y2)

        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1
        sy = 1 if gy1 < gy2 else -1
        err = dx - dy

        x, y = gx1, gy1
        while True:
            self._update_cell(x, y, occupied=occupied, height=height)

            if x == gx2 and y == gy2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy


class LiDARProcessor:
    """Process LiDAR point clouds for occupancy grid and 3D detection."""

    @staticmethod
    def filter_by_distance(points: np.ndarray, max_distance: float = 100.0) -> np.ndarray:
        """Filter points by distance from origin.

        Args:
            points: [N, 3] or [N, 4] point cloud (X, Y, Z[, intensity])
            max_distance: Maximum distance threshold

        Returns:
            Filtered points
        """
        distances = np.linalg.norm(points[:, :3], axis=1)
        mask = distances <= max_distance
        return points[mask]

    @staticmethod
    def filter_by_height(points: np.ndarray, min_height: float = -2.0, max_height: float = 3.0) -> np.ndarray:
        """Filter points by height.

        Args:
            points: [N, 3+] point cloud
            min_height: Minimum height
            max_height: Maximum height

        Returns:
            Filtered points
        """
        z = points[:, 2]
        mask = (z >= min_height) & (z <= max_height)
        return points[mask]

    @staticmethod
    def ground_segmentation(points: np.ndarray, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Segment ground from non-ground points.

        Args:
            points: [N, 3+] point cloud
            threshold: Height threshold for ground

        Returns:
            Tuple of (ground_points, non_ground_points)
        """
        z = points[:, 2]
        ground_mask = z < threshold
        return points[ground_mask], points[~ground_mask]

    @staticmethod
    def cluster_points(points: np.ndarray, distance_threshold: float = 0.2, min_points: int = 5) -> List[np.ndarray]:
        """Cluster points using DBSCAN-like algorithm.

        Args:
            points: [N, 3] point cloud
            distance_threshold: Distance threshold for clustering
            min_points: Minimum points per cluster

        Returns:
            List of point cloud clusters
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            warnings.warn("scikit-learn not available, using simple distance clustering")
            return LiDARProcessor._simple_cluster(points, distance_threshold, min_points)

        clustering = DBSCAN(eps=distance_threshold, min_samples=min_points).fit(points[:, :3])
        labels = clustering.labels_

        clusters = []
        for label in set(labels):
            if label >= 0:  # -1 is noise
                clusters.append(points[labels == label])

        return clusters

    @staticmethod
    def _simple_cluster(points: np.ndarray, distance_threshold: float, min_points: int) -> List[np.ndarray]:
        """Simple distance-based clustering."""
        if len(points) == 0:
            return []

        clusters = []
        used = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if used[i]:
                continue

            # Start new cluster
            cluster = [i]
            used[i] = True
            queue = [i]

            while queue:
                current = queue.pop(0)
                # Find neighbors
                distances = np.linalg.norm(points - points[current], axis=1)
                neighbors = np.where((distances < distance_threshold) & (~used))[0]

                for neighbor in neighbors:
                    cluster.append(neighbor)
                    used[neighbor] = True
                    queue.append(neighbor)

            if len(cluster) >= min_points:
                clusters.append(points[cluster])

        return clusters

    @staticmethod
    def compute_normals(points: np.ndarray, k: int = 10) -> np.ndarray:
        """Compute surface normals using k-nearest neighbors.

        Args:
            points: [N, 3] point cloud
            k: Number of neighbors

        Returns:
            [N, 3] normal vectors
        """
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            warnings.warn("scikit-learn not available, returning zero normals")
            return np.zeros_like(points)

        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, indices = nbrs.kneighbors(points)

        normals = np.zeros_like(points)

        for i in range(len(points)):
            neighbors = points[indices[i]]
            # Compute covariance
            mean = neighbors.mean(axis=0)
            centered = neighbors - mean
            cov = centered.T @ centered

            # Compute SVD
            try:
                U, S, Vt = np.linalg.svd(cov)
                # Normal is the last column of V (smallest singular value)
                normals[i] = U[:, -1]
            except:
                normals[i] = np.array([0, 0, 1])

        return normals


class RadarFusionProcessor:
    """Process radar data for velocity estimation and fusion."""

    @staticmethod
    def estimate_velocity_from_doppler(radar_detections: List[Dict]) -> np.ndarray:
        """Estimate velocity from radar Doppler shift.

        Args:
            radar_detections: List of radar detection dicts with keys:
                - "x", "y", "z": position
                - "vx", "vy", "vz": velocity from Doppler

        Returns:
            [N, 3] velocity vectors
        """
        if not radar_detections:
            return np.array([])

        velocities = np.array([
            [det.get("vx", 0.0), det.get("vy", 0.0), det.get("vz", 0.0)]
            for det in radar_detections
        ])

        return velocities

    @staticmethod
    def fuse_radar_lidar(
        lidar_points: np.ndarray,
        radar_detections: List[Dict],
        distance_threshold: float = 0.5,
    ) -> np.ndarray:
        """Fuse radar and LiDAR for combined detection and velocity.

        Args:
            lidar_points: [N, 3+] LiDAR point cloud
            radar_detections: List of radar detections
            distance_threshold: Max distance to associate

        Returns:
            [M, 6] array of (x, y, z, vx, vy, vz)
        """
        if len(radar_detections) == 0:
            # No radar data, return LiDAR only
            return lidar_points[:, :3]

        fused = []

        for radar_det in radar_detections:
            radar_pos = np.array([radar_det.get("x", 0), radar_det.get("y", 0), radar_det.get("z", 0)])
            radar_vel = np.array([radar_det.get("vx", 0), radar_det.get("vy", 0), radar_det.get("vz", 0)])

            # Find nearby LiDAR points
            if len(lidar_points) > 0:
                distances = np.linalg.norm(lidar_points[:, :3] - radar_pos, axis=1)
                nearby = distances < distance_threshold

                if nearby.any():
                    # Average nearby points
                    avg_pos = lidar_points[nearby, :3].mean(axis=0)
                    fused.append(np.concatenate([avg_pos, radar_vel]))
                else:
                    # Use radar position only
                    fused.append(np.concatenate([radar_pos, radar_vel]))
            else:
                # No LiDAR, use radar
                fused.append(np.concatenate([radar_pos, radar_vel]))

        return np.array(fused)
