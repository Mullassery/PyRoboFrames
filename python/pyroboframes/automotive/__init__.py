"""Automotive video stitching and 3D perception for autonomous driving.

Phase 1-6 features:
- Cylindrical panoramic stitching (Phase 1) ✓
- Advanced blending: Laplacian pyramid + graph-cut seams (Phase 2) ✓
- Bird's-eye-view projection for 3D perception (Phase 3) ✓
- GPU acceleration & temporal consistency (Phase 4) ✓
- Real-world dataset loaders: Waymo, nuScenes, KITTI (Phase 5) ✓
- Advanced 3D perception: Lidar fusion, radar fusion, occupancy grids (Phase 6) ✓

Example:
    ```python
    from pyroboframes.automotive import (
        CylindricalStitcher,
        BEVProjector,
        WaymoDatasetLoader,
        LidarFusion,
        OccupancyGrid,
        get_waymo_layout,
    )

    # Phase 1: Panoramic stitching
    layout = get_waymo_layout()
    stitcher = CylindricalStitcher(layout, blend_method="laplacian")
    panorama = stitcher.stitch(frames)

    # Phase 3: BEV projection for 3D perception
    bev_projector = BEVProjector(calibrations)
    bev = bev_projector.frames_to_bev(frames)

    # Phase 5: Real-world datasets
    waymo_loader = WaymoDatasetLoader("/path/to/waymo")
    for batch in waymo_loader:
        frames = batch["frames"]
        lidar = batch["lidar"]

    # Phase 6: 3D perception
    lidar_fusion = LidarFusion(num_lidars=5)
    fused = lidar_fusion.fuse(point_clouds, transforms)

    occupancy = OccupancyGrid(size=(-50, 50), resolution=0.1)
    occupancy.update(lidar_points)
    occupancy_map = occupancy.get_occupancy_map()
    ```
"""

from .bev import BEVProjector, create_bev_grid, warp_bev_to_panorama
from .blending import (
    blend_laplacian_pyramids,
    blend_with_seam,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    compensate_exposure,
    compute_blend_mask,
    find_optimal_seam,
)
from .camera_layouts import CAMERA_LAYOUTS, get_nuscenes_layout, get_waymo_layout
from .datasets import (
    KITTIDatasetLoader,
    NuScenesDatasetLoader,
    WaymoDatasetLoader,
)
from .perception_3d import (
    LidarFusion,
    OccupancyGrid,
    RadarFusion,
)
from .stitching import CylindricalStitcher

__all__ = [
    # Phase 1: Cylindrical stitching
    "CylindricalStitcher",
    # Phase 2: Advanced blending
    "build_gaussian_pyramid",
    "build_laplacian_pyramid",
    "blend_laplacian_pyramids",
    "find_optimal_seam",
    "blend_with_seam",
    "compensate_exposure",
    "compute_blend_mask",
    # Phase 3: BEV projection
    "BEVProjector",
    "create_bev_grid",
    "warp_bev_to_panorama",
    # Phase 5: Dataset loaders
    "WaymoDatasetLoader",
    "NuScenesDatasetLoader",
    "KITTIDatasetLoader",
    # Phase 6: 3D perception
    "LidarFusion",
    "RadarFusion",
    "OccupancyGrid",
    # Camera layouts
    "CAMERA_LAYOUTS",
    "get_waymo_layout",
    "get_nuscenes_layout",
]
