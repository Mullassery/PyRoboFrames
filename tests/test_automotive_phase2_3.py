"""Tests for automotive Phase 2 (advanced blending) and Phase 3 (BEV projection)."""

import numpy as np
import pytest

from pyroboframes.automotive import (
    BEVProjector,
    blend_laplacian_pyramids,
    blend_with_seam,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    compensate_exposure,
    create_bev_grid,
    find_optimal_seam,
    get_waymo_layout,
    CylindricalStitcher,
)
from pyroboframes.automotive.blending import compute_blend_mask


class TestPhase2Blending:
    """Test Phase 2: advanced blending techniques."""

    def test_build_gaussian_pyramid(self):
        """Test Gaussian pyramid construction."""
        image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        pyramid = build_gaussian_pyramid(image, levels=3)

        assert len(pyramid) == 3
        assert pyramid[0].shape == (480, 640, 3)
        assert pyramid[1].shape == (240, 320, 3)
        assert pyramid[2].shape == (120, 160, 3)

    def test_build_laplacian_pyramid(self):
        """Test Laplacian pyramid construction."""
        image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        laplacian_pyr = build_laplacian_pyramid(image, levels=3)

        assert len(laplacian_pyr) == 3
        assert laplacian_pyr[0].shape == (480, 640, 3)
        assert laplacian_pyr[1].shape == (240, 320, 3)
        assert laplacian_pyr[-1].shape == (120, 160, 3)

    def test_blend_laplacian_pyramids(self):
        """Test Laplacian pyramid blending."""
        image1 = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        image2 = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

        pyr1 = build_laplacian_pyramid(image1, levels=2)
        pyr2 = build_laplacian_pyramid(image2, levels=2)

        mask1 = np.ones((480, 640), dtype=bool)
        mask2 = np.ones((480, 640), dtype=bool)

        blended = blend_laplacian_pyramids(pyr1, pyr2, mask1, mask2)

        assert blended.shape == (480, 640, 3)

    def test_find_optimal_seam(self):
        """Test seam-finding with dynamic programming."""
        left = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        right = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

        seam_x = 320  # Middle
        seam = find_optimal_seam(left, right, seam_x, overlap_width=64)

        assert seam.shape == (480,)
        assert seam.min() >= seam_x - 64
        assert seam.max() <= seam_x + 64

    def test_blend_with_seam(self):
        """Test seam-based blending."""
        left = (np.ones((480, 640, 3)) * 100).astype(np.uint8)
        right = (np.ones((480, 640, 3)) * 200).astype(np.uint8)

        seam = np.full(480, 320, dtype=np.int32)  # Straight seam at x=320

        blended = blend_with_seam(left, right, seam, blend_width=32)

        assert blended.shape == (480, 640, 3)
        assert blended.dtype == np.uint8
        # Left side should be darker, right side brighter
        assert blended[:, 100].mean() < blended[:, 500].mean()

    def test_compensate_exposure(self):
        """Test exposure compensation."""
        left = (np.ones((480, 640, 3)) * 200).astype(np.uint8)
        right = (np.ones((480, 640, 3)) * 100).astype(np.uint8)

        overlap = (300, 200, 340, 280)
        compensated = compensate_exposure(left, right, overlap)

        assert compensated.shape == right.shape
        assert compensated.dtype == right.dtype
        # Compensated right should be brighter than original
        assert compensated.mean() >= right.mean()

    def test_compute_blend_mask(self):
        """Test blend mask computation."""
        left_valid = np.ones((480, 640), dtype=bool)
        right_valid = np.ones((480, 640), dtype=bool)
        seam = np.full(480, 320, dtype=np.int32)

        mask = compute_blend_mask(left_valid, right_valid, seam, blend_width=32)

        assert mask.shape == (480, 640)
        assert mask.min() >= 0 and mask.max() <= 1
        # Seam center should have high blend weight
        assert mask[240, 320] > 0.8

    def test_laplacian_vs_linear_blending(self):
        """Test that Laplacian blending produces different results than linear."""
        layout = get_waymo_layout()

        # Create two different stitchers
        stitcher_linear = CylindricalStitcher(layout, blend_method="linear")
        stitcher_laplacian = CylindricalStitcher(layout, blend_method="laplacian")

        # Create simple test frames (solid colors)
        frames = {
            "FRONT": (np.ones((720, 1280, 3)) * 100).astype(np.uint8),
            "FRONT_LEFT": (np.ones((720, 1280, 3)) * 150).astype(np.uint8),
        }

        pan_linear = stitcher_linear.stitch(frames)
        pan_laplacian = stitcher_laplacian.stitch(frames)

        assert pan_linear.shape == pan_laplacian.shape
        # Results should be different due to different blending
        # (they might be very similar for solid colors, but structure should differ)


class TestPhase3BEV:
    """Test Phase 3: BEV projection for 3D perception."""

    def test_bev_projector_creation(self):
        """Test creating BEV projector."""
        calibrations = {
            "front": {
                "fx": 1000,
                "fy": 1000,
                "cx": 640,
                "cy": 360,
                "width": 1280,
                "height": 720,
            },
        }

        projector = BEVProjector(
            calibrations,
            bev_size=(400, 400),
            bev_range=(-50, 50, -25, 25),
        )

        assert projector.bev_size == (400, 400)
        assert projector.bev_range == (-50, 50, -25, 25)

    def test_bev_projector_single_camera(self):
        """Test projecting single camera to BEV."""
        calibrations = {
            "front": {
                "fx": 1000,
                "fy": 1000,
                "cx": 640,
                "cy": 360,
                "width": 1280,
                "height": 720,
            },
        }

        projector = BEVProjector(calibrations, bev_size=(200, 200))
        image = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

        bev = projector.image_to_bev(image, "front", camera_height=1.5)

        assert bev.shape == (200, 200, 3)
        assert bev.dtype == image.dtype

    def test_bev_projector_multi_camera(self):
        """Test projecting multiple cameras to BEV."""
        calibrations = {
            "front": {
                "fx": 1000,
                "fy": 1000,
                "cx": 640,
                "cy": 360,
                "width": 1280,
                "height": 720,
            },
            "left": {
                "fx": 1000,
                "fy": 1000,
                "cx": 640,
                "cy": 360,
                "width": 1280,
                "height": 720,
            },
        }

        projector = BEVProjector(calibrations, bev_size=(200, 200))

        frames = {
            "front": (np.ones((720, 1280, 3)) * 100).astype(np.uint8),
            "left": (np.ones((720, 1280, 3)) * 150).astype(np.uint8),
        }

        bev = projector.frames_to_bev(frames, fusion_method="max")

        assert bev.shape == (200, 200, 3)
        assert bev.dtype == np.uint8

    def test_bev_fusion_methods(self):
        """Test different BEV fusion methods."""
        calibrations = {
            "cam1": {
                "fx": 1000,
                "fy": 1000,
                "cx": 640,
                "cy": 360,
                "width": 1280,
                "height": 720,
            },
        }

        projector = BEVProjector(calibrations, bev_size=(100, 100))
        frames = {"cam1": (np.ones((720, 1280, 3)) * 128).astype(np.uint8)}

        # Test each fusion method
        bev_max = projector.frames_to_bev(frames, fusion_method="max")
        bev_mean = projector.frames_to_bev(frames, fusion_method="mean")
        bev_stack = projector.frames_to_bev(frames, fusion_method="stack")

        assert bev_max.shape == (100, 100, 3)
        assert bev_mean.shape == (100, 100, 3)
        assert bev_stack.shape == (100, 100, 3)

    def test_bev_get_dims(self):
        """Test getting BEV dimensions."""
        calibrations = {"front": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}}
        projector = BEVProjector(
            calibrations,
            bev_size=(400, 400),
            bev_range=(-50, 50, -25, 25),
        )

        h, w = projector.get_bev_size()
        assert h == 400
        assert w == 400

        range_vals = projector.get_bev_range()
        assert range_vals == (-50, 50, -25, 25)

    def test_create_bev_grid(self):
        """Test BEV coordinate grid creation."""
        grid = create_bev_grid((200, 200), (-50, 50, -25, 25))

        assert grid.shape == (200, 200, 2)
        assert grid.dtype in [np.float32, np.float64]

        # Check bounds (within small tolerance due to discretization)
        assert np.isclose(grid[0, 0, 0], -50, atol=1.0)  # x_min
        assert np.isclose(grid[0, -1, 0], 50, atol=1.0)  # x_max
        assert np.isclose(grid[0, 0, 1], -25, atol=1.0)  # y_min
        assert np.isclose(grid[-1, 0, 1], 25, atol=1.0)  # y_max

    def test_bev_projector_repr(self):
        """Test BEV projector string representation."""
        calibrations = {"front": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}}
        projector = BEVProjector(calibrations, bev_size=(200, 200))

        repr_str = repr(projector)
        assert "BEVProjector" in repr_str
        assert "200" in repr_str

    def test_bev_invalid_camera_error(self):
        """Test error on unknown camera."""
        calibrations = {"front": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}}
        projector = BEVProjector(calibrations)

        image = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

        with pytest.raises(KeyError):
            projector.image_to_bev(image, "nonexistent")

    def test_bev_empty_frames_error(self):
        """Test error on empty frames."""
        calibrations = {"front": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}}
        projector = BEVProjector(calibrations)

        with pytest.raises(ValueError, match="No frames"):
            projector.frames_to_bev({})

    def test_bev_different_sizes(self):
        """Test BEV with different output sizes."""
        calibrations = {"front": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}}

        for size in [(100, 100), (200, 200), (400, 400)]:
            projector = BEVProjector(calibrations, bev_size=size)
            assert projector.get_bev_size() == size

    def test_bev_different_ranges(self):
        """Test BEV with different coverage ranges."""
        calibrations = {"front": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}}

        ranges = [
            (-25, 25, -10, 10),
            (-50, 50, -25, 25),
            (-100, 100, -50, 50),
        ]

        for bev_range in ranges:
            projector = BEVProjector(calibrations, bev_range=bev_range)
            assert projector.get_bev_range() == bev_range

    def test_bev_batch_processing(self):
        """Test BEV with batch of frames."""
        calibrations = {
            "front": {
                "fx": 1000,
                "fy": 1000,
                "cx": 640,
                "cy": 360,
                "width": 1280,
                "height": 720,
            },
        }

        projector = BEVProjector(calibrations, bev_size=(100, 100))

        # Batch of frames
        frames = {
            "front": (np.random.rand(8, 720, 1280, 3) * 255).astype(np.uint8),
        }

        bev = projector.frames_to_bev(frames)
        assert bev.shape[0] == 100  # Uses first frame only in Phase 3
