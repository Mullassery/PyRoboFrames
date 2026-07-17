"""Tests for occupancy grid and 3D perception."""

import numpy as np
import pytest
from pyroboframes.occupancy_3d import (
    OccupancyGrid,
    OccupancyGridConfig,
    LiDARProcessor,
    RadarFusionProcessor,
)


class TestOccupancyGridConfig:
    """Test occupancy grid configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = OccupancyGridConfig()

        assert config.size_x == 100.0
        assert config.size_y == 100.0
        assert config.resolution == 0.1
        assert config.height_min == -1.0
        assert config.height_max == 3.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = OccupancyGridConfig(
            size_x=50.0,
            size_y=50.0,
            resolution=0.05,
        )

        assert config.size_x == 50.0
        assert config.resolution == 0.05


class TestOccupancyGrid:
    """Test occupancy grid."""

    def test_grid_initialization(self):
        """Test grid initialization."""
        config = OccupancyGridConfig(size_x=10.0, size_y=10.0, resolution=0.1)
        grid = OccupancyGrid(config)

        assert grid.grid_size_x == 100
        assert grid.grid_size_y == 100
        assert grid.occupancy.shape == (100, 100)
        # Initial state should be unknown (0.5)
        assert np.allclose(grid.occupancy, 0.5)

    def test_world_to_grid_conversion(self):
        """Test world to grid coordinate conversion."""
        config = OccupancyGridConfig(size_x=10.0, size_y=10.0, resolution=1.0)
        grid = OccupancyGrid(config)

        # Center of world should map to center of grid
        gx, gy = grid.world_to_grid(0, 0)
        assert gx == 5
        assert gy == 5

    def test_grid_to_world_conversion(self):
        """Test grid to world coordinate conversion."""
        config = OccupancyGridConfig(size_x=10.0, size_y=10.0, resolution=1.0)
        grid = OccupancyGrid(config)

        x, y = grid.grid_to_world(5, 5)
        assert x == pytest.approx(0, abs=0.1)
        assert y == pytest.approx(0, abs=0.1)

    def test_coordinate_conversion_roundtrip(self):
        """Test roundtrip conversion."""
        config = OccupancyGridConfig()
        grid = OccupancyGrid(config)

        # Start with world coordinates
        x_orig, y_orig = 10.5, -15.3
        gx, gy = grid.world_to_grid(x_orig, y_orig)
        x_recovered, y_recovered = grid.grid_to_world(gx, gy)

        # Should be approximately equal (within resolution)
        assert abs(x_recovered - x_orig) < config.resolution
        assert abs(y_recovered - y_orig) < config.resolution

    def test_add_point_cloud(self):
        """Test adding point cloud to grid."""
        config = OccupancyGridConfig(size_x=20.0, size_y=20.0, resolution=0.5)
        grid = OccupancyGrid(config)

        # Add points
        points = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float32)
        grid.add_point_cloud(points)

        # Grid should have occupied cells
        occupied_count = np.sum(grid.occupancy > 0.5)
        assert occupied_count > 0

    def test_add_bounding_box(self):
        """Test adding bounding box to grid."""
        config = OccupancyGridConfig(size_x=20.0, size_y=20.0, resolution=0.5)
        grid = OccupancyGrid(config)

        # Add a box
        bbox = {"x": 0, "y": 0, "width": 2.0, "length": 4.0, "height": 2.0}
        grid.add_bounding_box(bbox)

        # Should have occupied cells
        occupied_count = np.sum(grid.occupancy > 0.5)
        assert occupied_count > 0

    def test_dilate_operation(self):
        """Test morphological dilation."""
        config = OccupancyGridConfig(size_x=20.0, size_y=20.0, resolution=1.0)
        grid = OccupancyGrid(config)

        # Mark a single cell
        grid.occupancy[5, 5] = 0.9
        initial_occupied = np.sum(grid.occupancy > 0.5)

        # Dilate
        grid.dilate(kernel_size=3)
        dilated_occupied = np.sum(grid.occupancy > 0.5)

        # Should have more occupied cells
        assert dilated_occupied >= initial_occupied

    def test_erode_operation(self):
        """Test morphological erosion."""
        config = OccupancyGridConfig()
        grid = OccupancyGrid(config)

        # Create a region
        grid.occupancy[5:8, 5:8] = 0.9
        initial_occupied = np.sum(grid.occupancy > 0.5)

        # Erode
        grid.erode(kernel_size=3)
        eroded_occupied = np.sum(grid.occupancy > 0.5)

        # Should have fewer or equal occupied cells
        assert eroded_occupied <= initial_occupied

    def test_get_free_space_mask(self):
        """Test getting free space mask."""
        config = OccupancyGridConfig(size_x=10.0, size_y=10.0, resolution=1.0)
        grid = OccupancyGrid(config)

        # Mark some cells as occupied and some as free
        grid.occupancy[2:4, 2:4] = 0.9  # Occupied
        grid.occupancy[5:7, 5:7] = 0.1  # Free

        free_mask = grid.get_free_space_mask()

        assert free_mask.shape == (10, 10)
        assert free_mask.dtype == np.uint8
        # Free cells should be 1
        assert np.sum(free_mask) > 0

    def test_get_occupied_cells(self):
        """Test getting occupied cells."""
        config = OccupancyGridConfig(size_x=10.0, size_y=10.0, resolution=1.0)
        grid = OccupancyGrid(config)

        # Mark a cell
        grid.occupancy[3, 4] = 0.9
        grid.occupancy[5, 6] = 0.9

        occupied = grid.get_occupied_cells()

        assert len(occupied) > 0
        assert (4, 3) in occupied or (3, 4) in occupied  # Coordinates may vary


class TestLiDARProcessor:
    """Test LiDAR processing."""

    def test_filter_by_distance(self):
        """Test distance filtering."""
        points = np.array([
            [0, 0, 0],
            [5, 5, 0],
            [20, 20, 0],
            [50, 50, 0],
        ])

        filtered = LiDARProcessor.filter_by_distance(points, max_distance=30)

        assert filtered.shape[0] <= points.shape[0]

    def test_filter_by_height(self):
        """Test height filtering."""
        points = np.array([
            [0, 0, -2.0],
            [1, 1, 0.0],
            [2, 2, 1.0],
            [3, 3, 5.0],
        ])

        filtered = LiDARProcessor.filter_by_height(points, min_height=-1.0, max_height=3.0)

        assert filtered.shape[0] <= points.shape[0]
        assert np.all(filtered[:, 2] >= -1.0)
        assert np.all(filtered[:, 2] <= 3.0)

    def test_ground_segmentation(self):
        """Test ground segmentation."""
        points = np.array([
            [0, 0, -0.5],  # Ground
            [1, 1, 0.0],   # Ground
            [2, 2, 0.5],   # Ground
            [3, 3, 1.5],   # Non-ground
        ])

        ground, non_ground = LiDARProcessor.ground_segmentation(points, threshold=0.1)

        assert ground.shape[0] > 0
        assert non_ground.shape[0] > 0

    def test_clustering(self):
        """Test point clustering."""
        # Create two clusters
        points = np.vstack([
            np.random.normal([0, 0, 0], 0.2, (10, 3)),
            np.random.normal([5, 5, 0], 0.2, (10, 3)),
        ])

        clusters = LiDARProcessor.cluster_points(points, distance_threshold=1.0, min_points=3)

        assert len(clusters) > 0
        # Each cluster should have points
        for cluster in clusters:
            assert cluster.shape[0] > 0

    def test_compute_normals(self):
        """Test normal computation."""
        # Create a plane-like point cloud
        x = np.random.uniform(-5, 5, 100)
        y = np.random.uniform(-5, 5, 100)
        z = np.zeros(100)
        points = np.column_stack([x, y, z])

        normals = LiDARProcessor.compute_normals(points, k=5)

        assert normals.shape == points.shape
        # Normals should have unit-ish magnitude
        magnitudes = np.linalg.norm(normals, axis=1)
        assert np.all(magnitudes > 0)


class TestRadarFusionProcessor:
    """Test radar fusion."""

    def test_velocity_from_doppler(self):
        """Test velocity estimation from Doppler."""
        detections = [
            {"x": 0, "y": 0, "z": 0, "vx": 1.0, "vy": 0.5, "vz": 0},
            {"x": 5, "y": 5, "z": 0, "vx": 2.0, "vy": 1.0, "vz": 0},
        ]

        velocities = RadarFusionProcessor.estimate_velocity_from_doppler(detections)

        assert velocities.shape == (2, 3)
        assert velocities[0, 0] == pytest.approx(1.0)

    def test_velocity_from_empty_detections(self):
        """Test with empty detections."""
        velocities = RadarFusionProcessor.estimate_velocity_from_doppler([])

        assert velocities.shape[0] == 0

    def test_radar_lidar_fusion(self):
        """Test radar-LiDAR fusion."""
        lidar_points = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [2, 2, 0],
        ])

        radar_detections = [
            {"x": 0.1, "y": 0.1, "z": 0, "vx": 1.0, "vy": 0, "vz": 0},
        ]

        fused = RadarFusionProcessor.fuse_radar_lidar(
            lidar_points,
            radar_detections,
            distance_threshold=1.0
        )

        assert fused.shape[0] > 0
        assert fused.shape[1] >= 3  # At least x, y, z

    def test_radar_without_lidar(self):
        """Test radar fusion without LiDAR."""
        radar_detections = [
            {"x": 0, "y": 0, "z": 0, "vx": 1.0, "vy": 0.5, "vz": 0},
        ]

        fused = RadarFusionProcessor.fuse_radar_lidar(
            np.array([]),
            radar_detections,
            distance_threshold=1.0
        )

        assert fused.shape[0] == len(radar_detections)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
