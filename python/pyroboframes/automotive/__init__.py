"""Automotive video stitching and 3D perception for autonomous driving.

Phase 1-3 features:
- Cylindrical panoramic stitching (Phase 1) ✓
- Advanced blending: Laplacian pyramid + graph-cut seams (Phase 2) ✓
- Bird's-eye-view projection for 3D perception (Phase 3) ✓

Example:
    ```python
    from pyroboframes.automotive import (
        CylindricalStitcher,
        BEVProjector,
        get_waymo_layout,
    )

    # Phase 1: Panoramic stitching
    layout = get_waymo_layout()
    stitcher = CylindricalStitcher(layout, blend_method="laplacian")
    panorama = stitcher.stitch(frames)

    # Phase 3: BEV projection for 3D perception
    bev_projector = BEVProjector(calibrations)
    bev = bev_projector.frames_to_bev(frames)
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
    # Camera layouts
    "CAMERA_LAYOUTS",
    "get_waymo_layout",
    "get_nuscenes_layout",
]
