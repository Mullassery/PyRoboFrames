"""
Phase 5 & 6: Real-world datasets + 3D perception example.

Demonstrates:
- Phase 5: Loading datasets (Waymo, nuScenes, KITTI)
- Phase 6: 3D perception (lidar fusion, occupancy grids, radar)
"""

import numpy as np

from pyroboframes.automotive import (
    CylindricalStitcher,
    KITTIDatasetLoader,
    LidarFusion,
    NuScenesDatasetLoader,
    OccupancyGrid,
    RadarFusion,
    WaymoDatasetLoader,
    get_waymo_layout,
)


def phase5_waymo_example():
    """Phase 5: Load and process Waymo Open Dataset."""
    print("=" * 70)
    print("Phase 5: Waymo Open Dataset Loader")
    print("=" * 70)

    # In production, replace with actual Waymo dataset path
    # loader = WaymoDatasetLoader("/path/to/waymo_open_dataset", split="training")

    # Demo: Waymo dataset structure
    print(f"Waymo dataset features:")
    print(f"  - 5 cameras: FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT")
    print(f"  - 5 lidar units with different scan patterns")
    print(f"  - 3D bounding boxes with class annotations")
    print(f"  - Auto-calibration detection from metadata")
    print()


def phase5_nuscenes_example():
    """Phase 5: Load and process nuScenes dataset."""
    print("=" * 70)
    print("Phase 5: nuScenes Dataset Loader")
    print("=" * 70)

    # Demo: nuScenes dataset structure
    print(f"nuScenes dataset features:")
    print(f"  - 6 cameras: FRONT, FRONT_LEFT, FRONT_RIGHT, BACK_LEFT, BACK_RIGHT, BACK")
    print(f"  - Lidar + mmWave radar")
    print(f"  - JSON-based metadata (easier to parse than Waymo TFRecords)")
    print(f"  - 1000 driving scenarios, 1.4M frames")
    print()


def phase5_kitti_example():
    """Phase 5: Load and process KITTI dataset."""
    print("=" * 70)
    print("Phase 5: KITTI Dataset Loader")
    print("=" * 70)

    # Demo: KITTI dataset
    print(f"KITTI dataset features:")
    print(f"  - Stereo pairs (left + right 1242×375)")
    print(f"  - 3D object detection annotations")
    print(f"  - Known good camera calibration")
    print(f"  - 7,000+ training images")
    print()


def phase6_lidar_fusion_example():
    """Phase 6: Multi-lidar point cloud fusion."""
    print("=" * 70)
    print("Phase 6: Lidar Fusion")
    print("=" * 70)

    # Create synthetic lidar data (simulating 5 Waymo lidar units)
    np.random.seed(42)

    # Generate 5 point clouds from different sensors
    point_clouds = []
    for sensor_id in range(5):
        # Each sensor captures ~20k points
        points = np.random.randn(20000, 3).astype(np.float32) * 30
        # Add intensity channel
        intensity = np.random.rand(20000, 1).astype(np.float32)
        pc = np.hstack([points, intensity])
        point_clouds.append(pc)

    # Simulate sensor transforms (lidar → vehicle frame)
    transforms = []
    for sensor_id in range(5):
        # Simple rotation + translation
        angle = sensor_id * 72  # 72° apart
        rad = np.radians(angle)

        T = np.eye(4)
        T[:2, :2] = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        T[0, 3] = 0.5 * np.cos(rad)  # Slight offset
        T[1, 3] = 0.5 * np.sin(rad)

        transforms.append(T)

    # Fuse point clouds
    fusion = LidarFusion(num_lidars=5, voxel_size=0.1, max_range=100.0)
    fused_pc = fusion.fuse(point_clouds, transforms)

    print(f"Multi-lidar fusion:")
    print(f"  - 5 input point clouds: {sum(len(pc) for pc in point_clouds):,} points total")
    print(f"  - Fused point cloud: {len(fused_pc):,} points (after voxel downsampling)")
    print(f"  - Voxel size: 0.1m")
    print(f"  - Range filtering: 0-100m")
    print()

    # Ground segmentation
    ground, non_ground = fusion.segment_ground(fused_pc)
    print(f"Ground segmentation:")
    print(f"  - Ground points: {len(ground):,}")
    print(f"  - Non-ground (objects): {len(non_ground):,}")
    print()


def phase6_radar_fusion_example():
    """Phase 6: Multi-radar velocity fusion."""
    print("=" * 70)
    print("Phase 6: Radar Fusion")
    print("=" * 70)

    # Create synthetic radar detections (x, y, vx, vy)
    # Front + back radar (typical nuScenes setup)
    np.random.seed(42)

    radar_front = np.array([
        [10.0, 2.0, 5.0, -0.5],    # Car ahead, moving forward
        [8.0, -2.0, 3.0, -0.2],    # Another vehicle
    ], dtype=np.float32)

    radar_back = np.array([
        [-5.0, 1.0, -10.0, 0.1],   # Car behind, approaching
    ], dtype=np.float32)

    radar_detections = [radar_front, radar_back]

    # Sensor transforms
    T_front = np.eye(4)
    T_back = np.eye(4)
    T_back[0, 3] = -2.5  # Rear sensor offset

    transforms = [T_front, T_back]

    # Fuse radar
    fusion = RadarFusion(num_radars=2)
    fused_radar = fusion.fuse(radar_detections, transforms)

    print(f"Multi-radar fusion:")
    print(f"  - Front radar detections: {len(radar_front)}")
    print(f"  - Back radar detections: {len(radar_back)}")
    print(f"  - Fused detections: {len(fused_radar)}")
    print()

    # Show detections
    print(f"Fused radar detections (x, y, vx, vy):")
    for i, det in enumerate(fused_radar):
        dist = np.linalg.norm(det[:2])
        speed = np.linalg.norm(det[2:])
        print(f"  {i}: pos=({det[0]:6.1f}, {det[1]:6.1f}), vel=({det[2]:6.1f}, {det[3]:6.1f}) m/s, dist={dist:5.1f}m, speed={speed:5.1f}m/s")
    print()


def phase6_occupancy_grid_example():
    """Phase 6: Bayesian occupancy grid mapping."""
    print("=" * 70)
    print("Phase 6: Occupancy Grid Mapping")
    print("=" * 70)

    # Create synthetic sensor data
    np.random.seed(42)

    # Lidar points
    lidar_points = np.random.randn(500, 3).astype(np.float32) * 15
    lidar_points[:, 2] = np.abs(lidar_points[:, 2])  # Positive z

    # Radar detections
    radar_detections = np.random.randn(20, 4).astype(np.float32) * 10

    # Initialize occupancy grid
    # Vehicle: center at (0, 0)
    # Grid: 100m × 100m (-50 to +50m in x and y)
    # Resolution: 20cm cells → 500×500 grid
    occupancy = OccupancyGrid(
        size=(-50.0, 50.0),
        resolution=0.2,
        log_odds_threshold=0.5,
    )

    print(f"Occupancy grid setup:")
    print(f"  - Size: 100m × 100m (-50 to +50m)")
    print(f"  - Resolution: 0.2m (20cm cells)")
    print(f"  - Total cells: {occupancy.grid_size}×{occupancy.grid_size}")
    print()

    # Update with sensor data
    occupancy.update(
        lidar_points=lidar_points,
        radar_detections=radar_detections,
    )

    # Get occupancy map
    occ_map = occupancy.get_occupancy_map()

    # Statistics
    free_cells = np.sum(occ_map < 0.25)
    occupied_cells = np.sum(occ_map > 0.75)
    unknown_cells = np.sum((occ_map >= 0.25) & (occ_map <= 0.75))

    print(f"Occupancy map statistics:")
    print(f"  - Free cells (p < 0.25): {free_cells:,}")
    print(f"  - Occupied cells (p > 0.75): {occupied_cells:,}")
    print(f"  - Unknown cells (0.25 ≤ p ≤ 0.75): {unknown_cells:,}")
    print()


def phase6_full_perception_pipeline():
    """Phase 6: Full 3D perception pipeline."""
    print("=" * 70)
    print("Phase 6: Full 3D Perception Pipeline")
    print("=" * 70)

    # Simulate real-world autonomous driving scenario
    np.random.seed(42)

    # Step 1: Load video frames with stitching (Phase 1)
    layout = get_waymo_layout()
    stitcher = CylindricalStitcher(layout, blend_method="laplacian")

    # Create synthetic frames
    frames = {
        cam: (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
        for cam in layout.cameras.keys()
    }

    panorama = stitcher.stitch(frames)
    print(f"Step 1: Panoramic stitching")
    print(f"  - Input: 5 camera frames (720×1280)")
    print(f"  - Output: panorama ({panorama.shape[1]}×{panorama.shape[2]})")
    print()

    # Step 2: Fuse lidar point clouds
    point_clouds = [
        np.random.randn(20000, 4).astype(np.float32) * 30
        for _ in range(5)
    ]
    transforms = [np.eye(4) for _ in range(5)]

    lidar_fusion = LidarFusion(num_lidars=5, voxel_size=0.1)
    fused_lidar = lidar_fusion.fuse(point_clouds, transforms)

    ground, non_ground = lidar_fusion.segment_ground(fused_lidar)
    print(f"Step 2: Lidar fusion + ground segmentation")
    print(f"  - Fused points: {len(fused_lidar):,}")
    print(f"  - Ground: {len(ground):,}, Objects: {len(non_ground):,}")
    print()

    # Step 3: Fuse radar detections
    radar_front = np.random.randn(5, 4).astype(np.float32) * 10
    radar_back = np.random.randn(3, 4).astype(np.float32) * 10

    radar_fusion = RadarFusion(num_radars=2)
    fused_radar = radar_fusion.fuse(
        [radar_front, radar_back],
        [np.eye(4), np.eye(4)]
    )

    print(f"Step 3: Radar fusion")
    print(f"  - Fused detections: {len(fused_radar)}")
    print()

    # Step 4: Build occupancy grid
    occupancy_grid = OccupancyGrid(size=(-50, 50), resolution=0.5)
    occupancy_grid.update(
        lidar_points=fused_lidar[:, :3],
        radar_detections=fused_radar,
    )

    occ_map = occupancy_grid.get_occupancy_map()
    occupied = np.sum(occ_map > 0.75)

    print(f"Step 4: Occupancy grid mapping")
    print(f"  - Grid: {occupancy_grid.grid_size}×{occupancy_grid.grid_size} cells")
    print(f"  - Occupied cells: {occupied}")
    print()

    print("✓ Full 3D perception pipeline complete")
    print()


def main():
    """Run all Phase 5 & 6 examples."""
    print("\n" + "=" * 70)
    print("PyRoboFrames v0.5.2: Phase 5 & 6 Examples")
    print("Real-World Datasets + Advanced 3D Perception")
    print("=" * 70 + "\n")

    # Phase 5: Datasets
    phase5_waymo_example()
    phase5_nuscenes_example()
    phase5_kitti_example()

    # Phase 6: 3D Perception
    phase6_lidar_fusion_example()
    phase6_radar_fusion_example()
    phase6_occupancy_grid_example()

    # Full pipeline
    phase6_full_perception_pipeline()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
