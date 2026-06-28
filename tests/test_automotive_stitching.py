"""Tests for automotive video stitching (Phase 1)."""

import numpy as np
import pytest

from pyroboframes.automotive import (
    CylindricalStitcher,
    get_nuscenes_layout,
    get_waymo_layout,
)
from pyroboframes.automotive.camera_layouts import CAMERA_LAYOUTS
from pyroboframes.automotive.projection import (
    blend_seam_linear,
    compute_panorama_bounds,
    create_panorama_grid,
    project_image_to_cylinder,
    spherical_to_cylindrical,
)


class TestCameraLayouts:
    """Test camera layout definitions."""

    def test_waymo_layout(self):
        """Test Waymo layout has 5 cameras."""
        layout = get_waymo_layout()
        assert layout.name == "waymo"
        assert len(layout.cameras) == 5
        assert "FRONT" in layout.cameras
        assert "FRONT_LEFT" in layout.cameras
        assert "FRONT_RIGHT" in layout.cameras
        assert "SIDE_LEFT" in layout.cameras
        assert "SIDE_RIGHT" in layout.cameras

    def test_nuscenes_layout(self):
        """Test nuScenes layout has 6 cameras."""
        layout = get_nuscenes_layout()
        assert layout.name == "nuscenes"
        assert len(layout.cameras) == 6
        assert "CAM_FRONT" in layout.cameras
        assert "CAM_BACK" in layout.cameras

    def test_layout_required_fields(self):
        """Test all cameras have required calibration fields."""
        for layout_name, layout in CAMERA_LAYOUTS.items():
            for cam_name, params in layout.cameras.items():
                assert "yaw_deg" in params, f"{layout_name}/{cam_name} missing yaw_deg"
                assert "fx" in params, f"{layout_name}/{cam_name} missing fx"
                assert "fy" in params, f"{layout_name}/{cam_name} missing fy"
                assert "cx" in params, f"{layout_name}/{cam_name} missing cx"
                assert "cy" in params, f"{layout_name}/{cam_name} missing cy"
                assert "width" in params, f"{layout_name}/{cam_name} missing width"
                assert "height" in params, f"{layout_name}/{cam_name} missing height"

    def test_layout_repr(self):
        """Test layout string representation."""
        layout = get_waymo_layout()
        repr_str = repr(layout)
        assert "waymo" in repr_str
        assert "5" in repr_str  # 5 cameras


class TestProjection:
    """Test cylindrical projection functions."""

    def test_spherical_to_cylindrical_shape(self):
        """Test spherical to cylindrical conversion."""
        xyz = np.random.randn(2, 480, 640, 3).astype(np.float32)
        u_pan, v_pan = spherical_to_cylindrical(xyz)

        assert u_pan.shape == (2, 480, 640)
        assert v_pan.shape == (2, 480, 640)
        assert u_pan.min() >= 0.0 and u_pan.max() <= 1.0
        assert v_pan.min() >= 0.0 and v_pan.max() <= 1.0

    def test_project_image_to_cylinder(self):
        """Test image to cylinder projection."""
        image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
        extrinsics = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}

        (u_pan, v_pan), mask = project_image_to_cylinder(image, intrinsics, extrinsics)

        assert u_pan.shape == (480, 640)
        assert v_pan.shape == (480, 640)
        assert mask.shape == (480, 640)
        assert mask.dtype == bool

    def test_project_with_rotation(self):
        """Test projection with camera rotation."""
        image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}

        # Test multiple yaw angles
        for yaw in [0.0, 45.0, 90.0]:
            extrinsics = {"yaw_deg": yaw, "pitch_deg": 0.0, "roll_deg": 0.0}
            (u_pan, v_pan), mask = project_image_to_cylinder(image, intrinsics, extrinsics)

            assert u_pan.shape == (480, 640)
            assert np.any(mask)  # Some pixels should be valid

    def test_blend_seam_linear(self):
        """Test linear seam blending."""
        left = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        right = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        left_mask = np.ones((480, 640), dtype=bool)
        right_mask = np.ones((480, 640), dtype=bool)

        blended = blend_seam_linear(left, right, left_mask, right_mask)

        assert blended.shape == left.shape
        assert blended.dtype == left.dtype

    def test_create_panorama_grid(self):
        """Test panorama grid creation."""
        grid = create_panorama_grid(480, 3200, num_cameras=5, camera_fov=90.0)

        assert grid.shape == (480, 3200, 2)
        assert grid.dtype == np.float32

    def test_compute_panorama_bounds(self):
        """Test panorama bounds computation."""
        layout = get_waymo_layout()
        bounds = compute_panorama_bounds(layout.cameras, 480)

        assert bounds["num_cameras"] == 5
        assert bounds["height"] == 480
        assert bounds["width"] > 480
        assert bounds["coverage_degrees"] == 360.0
        assert bounds["estimated_pixel_per_degree"] > 0


class TestCylindricalStitcher:
    """Test main stitching class."""

    def test_stitcher_creation_waymo(self):
        """Test creating stitcher with Waymo layout."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        assert stitcher.layout.name == "waymo"
        assert stitcher.panorama_height == 480
        assert len(stitcher.camera_order) == 5

    def test_stitcher_creation_nuscenes(self):
        """Test creating stitcher with nuScenes layout."""
        layout = get_nuscenes_layout()
        stitcher = CylindricalStitcher(layout, panorama_height=480)

        assert stitcher.layout.name == "nuscenes"
        assert len(stitcher.camera_order) == 6

    def test_stitcher_camera_order(self):
        """Test cameras are ordered by yaw angle."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        # Extract yaw angles in order
        yaws = [layout.cameras[cam]["yaw_deg"] for cam in stitcher.camera_order]

        # Should be sorted (monotonically increasing)
        assert yaws == sorted(yaws)

    def test_stitch_single_camera_batch(self):
        """Test stitching with batch dimension."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "FRONT": (np.random.rand(8, 720, 1280, 3) * 255).astype(np.uint8),
            "FRONT_LEFT": (np.random.rand(8, 720, 1280, 3) * 255).astype(np.uint8),
        }

        panorama = stitcher.stitch(frames)

        assert panorama.ndim == 4
        assert panorama.shape[0] == 8  # Batch size
        assert panorama.shape[1] == stitcher.panorama_height
        assert panorama.shape[2] == stitcher.panorama_width
        assert panorama.shape[3] == 3
        assert panorama.dtype == np.uint8

    def test_stitch_no_batch_dimension(self):
        """Test stitching without batch dimension."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "FRONT": (np.random.rand(720, 1280, 3) * 255).astype(np.uint8),
            "FRONT_LEFT": (np.random.rand(720, 1280, 3) * 255).astype(np.uint8),
        }

        panorama = stitcher.stitch(frames)

        # Should add batch dimension automatically
        assert panorama.ndim == 4
        assert panorama.shape[0] == 1

    def test_stitch_all_waymo_cameras(self):
        """Test stitching with all 5 Waymo cameras."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            cam: (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
            for cam in layout.cameras.keys()
        }

        panorama = stitcher.stitch(frames)

        assert panorama.shape[0] == 1
        assert panorama.shape[3] == 3

    def test_stitch_partial_cameras(self):
        """Test stitching with subset of cameras."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "FRONT": (np.random.rand(720, 1280, 3) * 255).astype(np.uint8),
            "SIDE_LEFT": (np.random.rand(720, 1280, 3) * 255).astype(np.uint8),
        }

        panorama = stitcher.stitch(frames)
        assert panorama.shape[0] == 1

    def test_stitch_with_mask(self):
        """Test stitching with validity mask."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "FRONT": (np.random.rand(720, 1280, 3) * 255).astype(np.uint8),
        }

        panorama, mask = stitcher.stitch_with_mask(frames)

        assert panorama.shape[0] == 1
        assert mask.shape == (1, stitcher.panorama_height, stitcher.panorama_width)
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 1))

    def test_get_panorama_dims(self):
        """Test getting panorama dimensions."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout, panorama_height=480)

        height, width = stitcher.get_panorama_dims()
        assert height == 480
        assert width == stitcher.panorama_width

    def test_stitcher_repr(self):
        """Test stitcher string representation."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        repr_str = repr(stitcher)
        assert "CylindricalStitcher" in repr_str
        assert "waymo" in repr_str
        assert "linear" in repr_str

    def test_invalid_blend_method(self):
        """Test error on unsupported blend method."""
        layout = get_waymo_layout()

        with pytest.raises(ValueError, match="blend_method"):
            CylindricalStitcher(layout, blend_method="invalid")

    def test_empty_frames_error(self):
        """Test error on empty frames dict."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        with pytest.raises(ValueError, match="No frames"):
            stitcher.stitch({})

    def test_unknown_camera_error(self):
        """Test error on unknown camera name."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "UNKNOWN_CAM": (np.random.rand(720, 1280, 3) * 255).astype(np.uint8),
        }

        with pytest.raises(KeyError):
            stitcher.stitch(frames)

    def test_invalid_frame_shape_error(self):
        """Test error on invalid frame shape."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "FRONT": np.random.rand(720, 1280),  # Missing channel dimension
        }

        with pytest.raises(ValueError, match="Frame shape"):
            stitcher.stitch(frames)

    def test_batch_size_mismatch_error(self):
        """Test error on mismatched batch sizes."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "FRONT": (np.random.rand(8, 720, 1280, 3) * 255).astype(np.uint8),
            "FRONT_LEFT": (np.random.rand(16, 720, 1280, 3) * 255).astype(np.uint8),
        }

        with pytest.raises(ValueError, match="Batch size"):
            stitcher.stitch(frames)

    def test_different_panorama_heights(self):
        """Test stitching with different output heights."""
        layout = get_waymo_layout()

        for height in [240, 480, 720]:
            stitcher = CylindricalStitcher(layout, panorama_height=height)
            assert stitcher.panorama_height == height

            frames = {
                "FRONT": (np.random.rand(720, 1280, 3) * 255).astype(np.uint8),
            }

            panorama = stitcher.stitch(frames)
            assert panorama.shape[1] == height

    def test_stitch_consistency(self):
        """Test stitching produces consistent output."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "FRONT": (np.ones((720, 1280, 3)) * 128).astype(np.uint8),
        }

        pan1 = stitcher.stitch(frames)
        pan2 = stitcher.stitch(frames)

        # Same input should produce same output
        assert np.allclose(pan1, pan2)

    def test_all_zero_frames(self):
        """Test stitching all-zero frames."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "FRONT": np.zeros((720, 1280, 3), dtype=np.uint8),
        }

        panorama = stitcher.stitch(frames)
        assert panorama.dtype == np.uint8
        # Output should be mostly black with some boundary artifacts

    def test_all_white_frames(self):
        """Test stitching all-white frames."""
        layout = get_waymo_layout()
        stitcher = CylindricalStitcher(layout)

        frames = {
            "FRONT": np.full((720, 1280, 3), 255, dtype=np.uint8),
        }

        panorama = stitcher.stitch(frames)
        assert panorama.dtype == np.uint8
        # Output should be mostly white
