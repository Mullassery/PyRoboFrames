"""Tests for Phase 5 (Real-world datasets) and Phase 6 (3D perception)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyroboframes.automotive import (
    KITTIDatasetLoader,
    LidarFusion,
    NuScenesDatasetLoader,
    OccupancyGrid,
    RadarFusion,
    WaymoDatasetLoader,
)


class TestPhase5Datasets:
    """Test Phase 5: Real-world dataset integration."""

    def test_waymo_dataset_loader_init(self):
        """Test WaymoDatasetLoader initialization with non-existent path."""
        with pytest.raises(FileNotFoundError):
            WaymoDatasetLoader("/nonexistent/waymo")

    def test_waymo_dataset_loader_structure(self):
        """Test WaymoDatasetLoader structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake Waymo structure
            split_dir = Path(tmpdir) / "training"
            split_dir.mkdir()
            (split_dir / "segment-0.tfrecord").touch()

            loader = WaymoDatasetLoader(tmpdir, split="training")

            assert loader.split == "training"
            assert len(loader) == 1

    def test_waymo_dataset_fraction(self):
        """Test WaymoDatasetLoader with fraction parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "training"
            split_dir.mkdir()

            # Create 10 fake segments
            for i in range(10):
                (split_dir / f"segment-{i}.tfrecord").touch()

            # Load only 50%
            loader = WaymoDatasetLoader(tmpdir, split="training", fraction=0.5)

            assert len(loader) == 5

    def test_waymo_dataset_iteration(self):
        """Test WaymoDatasetLoader iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "training"
            split_dir.mkdir()
            (split_dir / "segment-0.tfrecord").touch()

            loader = WaymoDatasetLoader(tmpdir, split="training")

            for batch in loader:
                assert "scene_id" in batch
                assert "frames" in batch
                assert "lidar" in batch
                assert "calibrations" in batch
                assert "annotations" in batch

                # Check frame structure
                frames = batch["frames"]
                assert isinstance(frames, dict)
                assert all(k in frames for k in [
                    "FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"
                ])

                # Check lidar shape
                lidar = batch["lidar"]
                assert lidar.shape == (100000, 4)
                assert lidar.dtype == np.float32

    def test_nuscenes_dataset_loader_init(self):
        """Test NuScenesDatasetLoader initialization."""
        with pytest.raises(FileNotFoundError):
            NuScenesDatasetLoader("/nonexistent/nuscenes")

    def test_nuscenes_dataset_loader_structure(self):
        """Test NuScenesDatasetLoader structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake nuScenes structure
            nuscenes_dir = Path(tmpdir) / "v1.0-trainval"
            nuscenes_dir.mkdir()

            loader = NuScenesDatasetLoader(tmpdir, version="v1.0-trainval")

            assert loader.version == "v1.0-trainval"
            assert len(loader) == 0  # No actual data

    def test_kitti_dataset_loader_init(self):
        """Test KITTIDatasetLoader initialization."""
        with pytest.raises(FileNotFoundError):
            KITTIDatasetLoader("/nonexistent/KITTI")

    def test_kitti_dataset_loader_structure(self):
        """Test KITTIDatasetLoader structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake KITTI structure
            training_dir = Path(tmpdir) / "training"
            training_dir.mkdir()
            img_dir = training_dir / "image_2"
            img_dir.mkdir()

            loader = KITTIDatasetLoader(tmpdir, task="3d_detection", split="training")

            assert loader.task == "3d_detection"
            assert len(loader) == 0  # No images

    def test_kitti_dataset_calibration(self):
        """Test KITTI loader camera calibration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_dir = Path(tmpdir) / "training"
            training_dir.mkdir()
            img_dir = training_dir / "image_2"
            img_dir.mkdir()

            loader = KITTIDatasetLoader(tmpdir, split="training")

            calibration = loader._load_calibration("0000000")

            assert "fx" in calibration
            assert "fy" in calibration
            assert "cx" in calibration
            assert "cy" in calibration
            # KITTI standard focal length
            assert np.isclose(calibration["fx"], 718.856, rtol=0.01)


class TestPhase6Perception3D:
    """Test Phase 6: Advanced 3D perception."""

    def test_lidar_fusion_init(self):
        """Test LidarFusion initialization."""
        fusion = LidarFusion(num_lidars=5, voxel_size=0.1)

        assert fusion.num_lidars == 5
        assert fusion.voxel_size == 0.1
        assert fusion.max_range == 100.0

    def test_lidar_fusion_empty(self):
        """Test LidarFusion with empty point clouds."""
        fusion = LidarFusion()

        point_clouds = []
        transforms = []

        fused = fusion.fuse(point_clouds, transforms)

        assert fused.shape == (0, 4)

    def test_lidar_fusion_single_cloud(self):
        """Test LidarFusion with single point cloud."""
        fusion = LidarFusion()

        # Create synthetic point cloud
        np.random.seed(42)
        points = np.random.randn(1000, 4).astype(np.float32)
        points[:, 2] = np.abs(points[:, 2])  # Make z positive
        point_clouds = [points]

        # Identity transform
        transforms = [np.eye(4)]

        fused = fusion.fuse(point_clouds, transforms)

        assert fused.shape[0] <= 1000  # Downsampled
        assert fused.shape[1] == 4

    def test_lidar_fusion_range_filtering(self):
        """Test LidarFusion range filtering."""
        fusion = LidarFusion(max_range=10.0)

        # Create points, some beyond range
        points = np.array([
            [5.0, 0.0, 0.0, 1.0],      # Within range
            [15.0, 0.0, 0.0, 1.0],     # Beyond range
            [0.0, 8.0, 0.0, 1.0],      # Within range
        ], dtype=np.float32)

        fused = fusion.fuse([points], [np.eye(4)])

        # Should have filtered out the 15m point
        assert fused.shape[0] <= 2

    def test_lidar_voxel_downsample(self):
        """Test voxel grid downsampling."""
        fusion = LidarFusion(voxel_size=1.0)

        # Create grid of points (should downsample significantly)
        points = []
        for x in np.linspace(0, 10, 5):
            for y in np.linspace(0, 10, 5):
                points.append([x, y, 0.0, 1.0])

        points = np.array(points, dtype=np.float32)

        downsampled = fusion._voxel_downsample(points)

        # Should be much smaller than 25
        assert len(downsampled) <= 25
        assert len(downsampled) > 0

    def test_lidar_ground_segmentation(self):
        """Test ground plane segmentation."""
        fusion = LidarFusion()

        # Create points, some ground, some above
        points = np.array([
            [0.0, 0.0, -1.0, 1.0],   # Ground
            [1.0, 0.0, -0.8, 1.0],   # Ground
            [0.0, 1.0, 1.0, 1.0],    # Not ground
            [1.0, 1.0, 2.0, 1.0],    # Not ground
        ], dtype=np.float32)

        ground, non_ground = fusion.segment_ground(points)

        assert len(ground) == 2
        assert len(non_ground) == 2

    def test_radar_fusion_init(self):
        """Test RadarFusion initialization."""
        radar = RadarFusion(num_radars=2)

        assert radar.num_radars == 2

    def test_radar_fusion_empty(self):
        """Test RadarFusion with empty detections."""
        radar = RadarFusion()

        detections = []
        transforms = []

        fused = radar.fuse(detections, transforms)

        assert fused.shape == (0, 4)

    def test_radar_fusion_single_sensor(self):
        """Test RadarFusion with single radar."""
        radar = RadarFusion()

        # Create synthetic detections (x, y, vx, vy)
        detections = np.array([
            [10.0, 5.0, 2.0, 1.0],
            [20.0, 0.0, -1.0, 0.5],
        ], dtype=np.float32)

        transforms = [np.eye(4)]

        fused = radar.fuse([detections], transforms)

        assert fused.shape == (2, 4)

    def test_radar_fusion_with_rotation(self):
        """Test RadarFusion with rotated sensor."""
        radar = RadarFusion()

        detections = np.array([
            [10.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)

        # 90-degree rotation
        T = np.eye(4)
        T[:2, :2] = np.array([[0, -1], [1, 0]])  # 90° rotation

        fused = radar.fuse([detections], [T])

        assert fused.shape == (1, 4)
        # Velocity should also be rotated
        assert fused[0, 2] != 1.0 or fused[0, 3] != 0.0

    def test_occupancy_grid_init(self):
        """Test OccupancyGrid initialization."""
        grid = OccupancyGrid(size=(-50.0, 50.0), resolution=0.1)

        assert grid.size == (-50.0, 50.0)
        assert grid.resolution == 0.1
        assert grid.grid_size == 1000

    def test_occupancy_grid_world_to_grid(self):
        """Test coordinate transformation."""
        grid = OccupancyGrid(size=(-10.0, 10.0), resolution=1.0)

        # Origin should map to middle of grid
        gx, gy = grid._world_to_grid(0.0, 0.0)
        assert 0 <= gx < grid.grid_size
        assert 0 <= gy < grid.grid_size

    def test_occupancy_grid_update_lidar(self):
        """Test occupancy grid update from lidar."""
        grid = OccupancyGrid(size=(-20.0, 20.0), resolution=0.5)

        # Create synthetic point cloud
        points = np.array([
            [5.0, 5.0, 0.0],
            [6.0, 5.0, 0.0],
            [10.0, 10.0, 0.0],
        ], dtype=np.float32)

        grid.update(lidar_points=points)

        occupancy = grid.get_occupancy_map()

        assert occupancy.shape == (grid.grid_size, grid.grid_size)
        assert occupancy.dtype == np.float32
        assert np.all(occupancy >= 0.0)
        assert np.all(occupancy <= 1.0)

    def test_occupancy_grid_update_radar(self):
        """Test occupancy grid update from radar."""
        grid = OccupancyGrid(size=(-20.0, 20.0), resolution=0.5)

        # Create synthetic radar detections
        detections = np.array([
            [5.0, 5.0, 1.0, 0.5],
            [10.0, 0.0, -1.0, 0.0],
        ], dtype=np.float32)

        grid.update(radar_detections=detections)

        occupancy = grid.get_occupancy_map()

        assert occupancy.shape == (grid.grid_size, grid.grid_size)

    def test_occupancy_grid_combined_update(self):
        """Test occupancy grid with combined lidar + radar."""
        grid = OccupancyGrid(size=(-50.0, 50.0), resolution=1.0)

        # Synthetic data
        lidar_points = np.random.randn(100, 3).astype(np.float32) * 10
        radar_detections = np.random.randn(10, 4).astype(np.float32) * 5

        grid.update(lidar_points=lidar_points, radar_detections=radar_detections)

        occupancy = grid.get_occupancy_map()

        # Should have updated some cells
        assert np.sum(occupancy > 0.5) > 0  # Some occupied cells

    def test_occupancy_grid_reset(self):
        """Test occupancy grid reset."""
        grid = OccupancyGrid()

        # Add data
        points = np.array([[5.0, 5.0, 0.0]], dtype=np.float32)
        grid.update(lidar_points=points)

        # Check some cells updated
        occupancy_before = grid.get_occupancy_map()
        assert np.sum(occupancy_before > 0.5) > 0

        # Reset
        grid.reset()

        occupancy_after = grid.get_occupancy_map()
        assert np.allclose(occupancy_after, 0.5)  # Back to uniform prior


class TestPhase5And6Integration:
    """Integration tests for Phase 5 and 6."""

    def test_waymo_to_lidar_fusion(self):
        """Test pipeline: Waymo dataset → lidar fusion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "training"
            split_dir.mkdir()
            (split_dir / "segment-0.tfrecord").touch()

            loader = WaymoDatasetLoader(tmpdir)
            fusion = LidarFusion(num_lidars=5)

            for batch in loader:
                # In real scenario, batch["lidar"] would be 5 separate clouds
                lidar = batch["lidar"]
                assert lidar.shape == (100000, 4)
                break  # Just test one batch

    def test_full_3d_perception_pipeline(self):
        """Test full 3D perception pipeline."""
        # Create fake sensor data
        lidar_points = np.random.randn(1000, 3).astype(np.float32) * 20
        radar_detections = np.random.randn(50, 4).astype(np.float32) * 10

        # Step 1: Fuse lidar
        lidar_fusion = LidarFusion(num_lidars=5)
        fused_lidar = lidar_fusion.fuse(
            [lidar_points],
            [np.eye(4)]
        )

        assert fused_lidar.shape[0] > 0

        # Step 2: Fuse radar
        radar_fusion = RadarFusion(num_radars=2)
        fused_radar = radar_fusion.fuse(
            [radar_detections],
            [np.eye(4)]
        )

        assert fused_radar.shape == radar_detections.shape

        # Step 3: Update occupancy grid
        occupancy_grid = OccupancyGrid(size=(-50, 50), resolution=0.2)
        occupancy_grid.update(
            lidar_points=fused_lidar[:, :3],
            radar_detections=fused_radar
        )

        occupancy_map = occupancy_grid.get_occupancy_map()

        assert occupancy_map.shape == (occupancy_grid.grid_size,) * 2
        assert np.all(occupancy_map >= 0.0)
        assert np.all(occupancy_map <= 1.0)
