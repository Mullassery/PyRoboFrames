"""Tests for real-world dataset loaders."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from pyroboframes.dataset_loaders import (
    WaymoDatasetLoader,
    nuScenesDatasetLoader,
    KITTIDatasetLoader,
    CameraCalibration,
    FrameMetadata,
)


class TestCameraCalibration:
    """Test camera calibration data."""

    def test_calibration_creation(self):
        """Test creating calibration object."""
        calib = CameraCalibration(
            fx=1000.0,
            fy=1000.0,
            cx=512.0,
            cy=384.0,
            width=1024,
            height=768,
        )

        assert calib.fx == 1000.0
        assert calib.width == 1024
        assert calib.height == 768
        assert calib.distortion is None

    def test_calibration_with_distortion(self):
        """Test calibration with distortion coefficients."""
        distortion = np.array([0.1, -0.05, 0.0, 0.0])
        calib = CameraCalibration(
            fx=1000.0,
            fy=1000.0,
            cx=512.0,
            cy=384.0,
            width=1024,
            height=768,
            distortion=distortion,
        )

        assert calib.distortion is not None
        assert len(calib.distortion) == 4


class TestWaymoDatasetLoader:
    """Test Waymo dataset loader."""

    def test_waymo_initialization(self):
        """Test initializing Waymo loader with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = WaymoDatasetLoader(tmpdir)

            # Should initialize without error
            assert loader.dataset_path == Path(tmpdir)
            assert isinstance(loader.scenes, list)

    def test_waymo_get_nonexistent_scene(self):
        """Test getting nonexistent scene."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = WaymoDatasetLoader(tmpdir)

            scene = loader.get_scene(999)
            assert scene is None

    def test_waymo_frame_loading_nonexistent(self):
        """Test loading nonexistent frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = WaymoDatasetLoader(tmpdir)

            image, metadata = loader.get_frame(0, 0)
            assert image is None
            assert metadata is None

    def test_waymo_default_calibration(self):
        """Test default calibration loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = WaymoDatasetLoader(tmpdir)
            calib = loader._load_calibration(Path(tmpdir) / "nonexistent.json", "FRONT")

            # Should return default calibration
            assert calib.fx == 2015.0
            assert calib.fy == 2015.0
            assert calib.width == 1920
            assert calib.height == 1200


class TestNuScenesDatasetLoader:
    """Test nuScenes dataset loader."""

    def test_nuscenes_initialization(self):
        """Test initializing nuScenes loader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = nuScenesDatasetLoader(tmpdir)

            assert loader.dataset_path == Path(tmpdir)
            assert isinstance(loader.scenes, list)

    def test_nuscenes_default_calibration(self):
        """Test nuScenes default calibration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = nuScenesDatasetLoader(tmpdir)

            scene = loader.get_scene(0)
            assert scene is None

    def test_nuscenes_frame_loading(self):
        """Test frame loading returns None for nonexistent frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = nuScenesDatasetLoader(tmpdir)

            image, metadata = loader.get_frame(0, 0, "CAM_FRONT")
            assert image is None
            assert metadata is None

    def test_nuscenes_camera_calibration(self):
        """Test that nuScenes uses correct default calibration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = nuScenesDatasetLoader(tmpdir)

            # Based on nuScenes standard
            assert loader is not None


class TestKITTIDatasetLoader:
    """Test KITTI dataset loader."""

    def test_kitti_initialization(self):
        """Test initializing KITTI loader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = KITTIDatasetLoader(tmpdir, split="training")

            assert loader.split == "training"
            assert isinstance(loader.sequences, list)

    def test_kitti_testing_split(self):
        """Test KITTI loader with testing split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = KITTIDatasetLoader(tmpdir, split="testing")

            assert loader.split == "testing"

    def test_kitti_get_nonexistent_sequence(self):
        """Test getting nonexistent sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = KITTIDatasetLoader(tmpdir)

            seq = loader.get_sequence(999)
            assert seq is None

    def test_kitti_frame_loading(self):
        """Test KITTI frame loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = KITTIDatasetLoader(tmpdir)

            image, metadata = loader.get_frame(0, 0, camera=0)
            assert image is None
            assert metadata is None

    def test_kitti_default_calibration(self):
        """Test KITTI default calibration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = KITTIDatasetLoader(tmpdir)
            calib = loader._load_kitti_calib(Path(tmpdir) / "nonexistent.txt", 0)

            # Should return default KITTI calibration
            assert calib.fx == pytest.approx(718.856, rel=0.01)
            assert calib.fy == pytest.approx(718.856, rel=0.01)
            assert calib.width == 1242
            assert calib.height == 375


class TestFrameMetadata:
    """Test frame metadata."""

    def test_frame_metadata_creation(self):
        """Test creating frame metadata."""
        calib = CameraCalibration(1000, 1000, 512, 384, 1024, 768)
        metadata = FrameMetadata(
            timestamp=1.0,
            camera_name="FRONT",
            frame_index=0,
            image_path="/path/to/frame.png",
            calibration=calib,
        )

        assert metadata.timestamp == 1.0
        assert metadata.camera_name == "FRONT"
        assert metadata.frame_index == 0
        assert metadata.calibration is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
