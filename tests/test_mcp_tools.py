"""
PyRoboFrames MCP Tools Unit Tests
Tests for all 11 MCP metadata discovery and manipulation tools
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json


class TestSearchDatasets:
    """Test search_datasets MCP tool"""

    def test_search_by_query(self):
        """Search datasets by text query"""
        mock_metadata = Mock()
        mock_metadata.search_datasets.return_value = [
            {"name": "thermal_dataset_1", "modalities": ["thermal"]},
            {"name": "thermal_dataset_2", "modalities": ["thermal", "rgb"]},
        ]

        result = mock_metadata.search_datasets("thermal")
        assert len(result) == 2
        assert all("thermal" in d["modalities"] for d in result)

    def test_search_by_modalities(self):
        """Search datasets by modality filter"""
        mock_metadata = Mock()
        mock_metadata.search_datasets.return_value = [
            {"name": "rgbd_1", "modalities": ["rgb", "depth"]},
        ]

        result = mock_metadata.search_datasets(modalities=["rgb", "depth"])
        assert len(result) == 1
        assert set(result[0]["modalities"]) == {"rgb", "depth"}

    def test_search_empty_results(self):
        """Handle empty search results gracefully"""
        mock_metadata = Mock()
        mock_metadata.search_datasets.return_value = []

        result = mock_metadata.search_datasets("nonexistent")
        assert result == []
        assert len(result) == 0

    def test_search_with_tags(self):
        """Search datasets by tags"""
        mock_metadata = Mock()
        mock_metadata.search_datasets.return_value = [
            {"name": "dataset_1", "tags": ["production", "verified"]},
        ]

        result = mock_metadata.search_datasets(tags=["production"])
        assert len(result) == 1
        assert "production" in result[0]["tags"]


class TestListDatasets:
    """Test list_datasets MCP tool"""

    def test_list_all_datasets(self):
        """List all available datasets"""
        mock_metadata = Mock()
        datasets = [
            {"name": "ds1", "size_mb": 100},
            {"name": "ds2", "size_mb": 200},
            {"name": "ds3", "size_mb": 150},
        ]
        mock_metadata.list_datasets.return_value = datasets

        result = mock_metadata.list_datasets()
        assert len(result) == 3
        assert all("name" in d for d in result)

    def test_list_with_limit(self):
        """List datasets with limit"""
        mock_metadata = Mock()
        mock_metadata.list_datasets.return_value = [
            {"name": "ds1"},
            {"name": "ds2"},
        ]

        result = mock_metadata.list_datasets(limit=2)
        assert len(result) == 2

    def test_list_with_offset(self):
        """List datasets with pagination offset"""
        mock_metadata = Mock()
        mock_metadata.list_datasets.return_value = [
            {"name": "ds3"},
            {"name": "ds4"},
        ]

        result = mock_metadata.list_datasets(offset=2)
        assert len(result) == 2

    def test_list_sorted_by_size(self):
        """List datasets sorted by size"""
        mock_metadata = Mock()
        datasets = [
            {"name": "ds1", "size_mb": 100},
            {"name": "ds2", "size_mb": 50},
            {"name": "ds3", "size_mb": 200},
        ]
        mock_metadata.list_datasets.return_value = sorted(
            datasets, key=lambda x: x["size_mb"], reverse=True
        )

        result = mock_metadata.list_datasets(sort_by="size", descending=True)
        assert result[0]["size_mb"] >= result[1]["size_mb"]


class TestGetDatasetInfo:
    """Test get_dataset_info MCP tool"""

    def test_get_dataset_info_success(self):
        """Get full dataset information"""
        mock_metadata = Mock()
        mock_metadata.get_dataset_info.return_value = {
            "name": "test_dataset",
            "version": "1.0",
            "frames": 1000,
            "modalities": ["rgb", "depth"],
            "size_mb": 500,
            "description": "Test dataset",
        }

        result = mock_metadata.get_dataset_info("test_dataset")
        assert result["name"] == "test_dataset"
        assert result["frames"] == 1000
        assert "rgb" in result["modalities"]

    def test_get_dataset_info_not_found(self):
        """Handle dataset not found"""
        mock_metadata = Mock()
        mock_metadata.get_dataset_info.side_effect = ValueError(
            "Dataset not found"
        )

        with pytest.raises(ValueError):
            mock_metadata.get_dataset_info("nonexistent")

    def test_get_dataset_metadata_fields(self):
        """Verify all expected metadata fields present"""
        mock_metadata = Mock()
        info = {
            "name": "ds",
            "version": "1.0",
            "frames": 100,
            "modalities": ["rgb"],
            "size_mb": 50,
            "created": "2024-01-01",
            "updated": "2024-01-15",
        }
        mock_metadata.get_dataset_info.return_value = info

        result = mock_metadata.get_dataset_info("ds")
        expected_fields = {
            "name",
            "version",
            "frames",
            "modalities",
            "size_mb",
        }
        assert expected_fields.issubset(set(result.keys()))


class TestGetFrameMetadata:
    """Test get_frame_metadata MCP tool"""

    def test_get_frame_metadata_success(self):
        """Get metadata for specific frame"""
        mock_metadata = Mock()
        mock_metadata.get_frame_metadata.return_value = {
            "frame_id": 42,
            "timestamp": 1234567890,
            "modalities_present": ["rgb", "depth"],
            "annotations_count": 5,
            "size_kb": 256,
        }

        result = mock_metadata.get_frame_metadata("dataset_1", 42)
        assert result["frame_id"] == 42
        assert "rgb" in result["modalities_present"]

    def test_get_frame_by_timestamp(self):
        """Get frame by timestamp"""
        mock_metadata = Mock()
        mock_metadata.get_frame_metadata.return_value = {
            "frame_id": 100,
            "timestamp": 1704067200,
        }

        result = mock_metadata.get_frame_metadata("dataset_1", timestamp=1704067200)
        assert result["timestamp"] == 1704067200

    def test_get_frame_not_found(self):
        """Handle frame not found"""
        mock_metadata = Mock()
        mock_metadata.get_frame_metadata.side_effect = IndexError(
            "Frame not found"
        )

        with pytest.raises(IndexError):
            mock_metadata.get_frame_metadata("dataset_1", 99999)


class TestListAnnotations:
    """Test list_annotations MCP tool"""

    def test_list_annotations_by_dataset(self):
        """List all annotations in dataset"""
        mock_metadata = Mock()
        annotations = [
            {"id": "ann_1", "type": "bbox", "label": "object_1"},
            {"id": "ann_2", "type": "mask", "label": "segment_1"},
            {"id": "ann_3", "type": "keypoint", "label": "pose_1"},
        ]
        mock_metadata.list_annotations.return_value = annotations

        result = mock_metadata.list_annotations("dataset_1")
        assert len(result) == 3
        assert all("id" in a for a in result)

    def test_list_annotations_by_type(self):
        """List annotations filtered by type"""
        mock_metadata = Mock()
        mock_metadata.list_annotations.return_value = [
            {"id": "ann_1", "type": "bbox"},
            {"id": "ann_2", "type": "bbox"},
        ]

        result = mock_metadata.list_annotations("dataset_1", ann_type="bbox")
        assert all(a["type"] == "bbox" for a in result)

    def test_list_annotations_by_frame(self):
        """List annotations for specific frame"""
        mock_metadata = Mock()
        mock_metadata.list_annotations.return_value = [
            {"id": "ann_1", "frame_id": 42, "type": "bbox"},
        ]

        result = mock_metadata.list_annotations("dataset_1", frame_id=42)
        assert all(a.get("frame_id") == 42 for a in result)

    def test_list_annotations_empty(self):
        """Handle dataset with no annotations"""
        mock_metadata = Mock()
        mock_metadata.list_annotations.return_value = []

        result = mock_metadata.list_annotations("dataset_1")
        assert len(result) == 0


class TestGetAnnotation:
    """Test get_annotation MCP tool"""

    def test_get_annotation_by_id(self):
        """Get specific annotation details"""
        mock_metadata = Mock()
        mock_metadata.get_annotation.return_value = {
            "id": "ann_1",
            "type": "bbox",
            "label": "person",
            "frame_id": 42,
            "coordinates": [10, 20, 100, 200],
            "confidence": 0.95,
        }

        result = mock_metadata.get_annotation("dataset_1", "ann_1")
        assert result["id"] == "ann_1"
        assert result["type"] == "bbox"
        assert result["confidence"] == 0.95

    def test_get_annotation_not_found(self):
        """Handle annotation not found"""
        mock_metadata = Mock()
        mock_metadata.get_annotation.side_effect = KeyError(
            "Annotation not found"
        )

        with pytest.raises(KeyError):
            mock_metadata.get_annotation("dataset_1", "nonexistent")

    def test_get_annotation_with_metadata(self):
        """Annotation includes complete metadata"""
        mock_metadata = Mock()
        annotation = {
            "id": "ann_1",
            "type": "bbox",
            "label": "car",
            "frame_id": 100,
            "dataset_id": "dataset_1",
            "created_by": "annotator_1",
            "created_at": "2024-01-01T12:00:00",
        }
        mock_metadata.get_annotation.return_value = annotation

        result = mock_metadata.get_annotation("dataset_1", "ann_1")
        assert result["created_by"] == "annotator_1"
        assert result["dataset_id"] == "dataset_1"


class TestVerifyDatasetIntegrity:
    """Test verify_dataset_integrity MCP tool"""

    def test_verify_dataset_integrity_success(self):
        """Verify dataset passes integrity checks"""
        mock_metadata = Mock()
        mock_metadata.verify_dataset_integrity.return_value = {
            "status": "valid",
            "checks_passed": 8,
            "checks_failed": 0,
            "issues": [],
        }

        result = mock_metadata.verify_dataset_integrity("dataset_1")
        assert result["status"] == "valid"
        assert result["checks_passed"] > 0
        assert len(result["issues"]) == 0

    def test_verify_dataset_with_issues(self):
        """Detect integrity issues"""
        mock_metadata = Mock()
        mock_metadata.verify_dataset_integrity.return_value = {
            "status": "invalid",
            "checks_passed": 5,
            "checks_failed": 3,
            "issues": [
                "Missing frame metadata for frame_1",
                "Corrupted annotation file",
                "Size mismatch on frame_5",
            ],
        }

        result = mock_metadata.verify_dataset_integrity("dataset_1")
        assert result["status"] == "invalid"
        assert len(result["issues"]) == 3

    def test_verify_specific_checks(self):
        """Verify specific integrity checks"""
        mock_metadata = Mock()
        mock_metadata.verify_dataset_integrity.return_value = {
            "status": "valid",
            "frame_count_check": True,
            "modality_consistency_check": True,
            "annotation_integrity_check": True,
        }

        result = mock_metadata.verify_dataset_integrity("dataset_1")
        assert result["frame_count_check"] is True
        assert result["modality_consistency_check"] is True


class TestGetDatasetStats:
    """Test get_dataset_stats MCP tool"""

    def test_get_dataset_stats(self):
        """Get dataset statistics"""
        mock_metadata = Mock()
        mock_metadata.get_dataset_stats.return_value = {
            "total_frames": 1000,
            "total_size_mb": 500,
            "modalities": ["rgb", "depth", "thermal"],
            "annotations_count": 5000,
            "avg_frame_size_kb": 512,
            "fps": 30,
        }

        result = mock_metadata.get_dataset_stats("dataset_1")
        assert result["total_frames"] == 1000
        assert result["total_size_mb"] == 500
        assert len(result["modalities"]) == 3

    def test_stats_by_modality(self):
        """Get statistics broken down by modality"""
        mock_metadata = Mock()
        mock_metadata.get_dataset_stats.return_value = {
            "modality_stats": {
                "rgb": {"frames": 1000, "size_mb": 300},
                "depth": {"frames": 1000, "size_mb": 150},
                "thermal": {"frames": 1000, "size_mb": 50},
            }
        }

        result = mock_metadata.get_dataset_stats("dataset_1")
        assert "modality_stats" in result
        assert result["modality_stats"]["rgb"]["size_mb"] == 300

    def test_stats_by_annotation_type(self):
        """Get statistics by annotation type"""
        mock_metadata = Mock()
        mock_metadata.get_dataset_stats.return_value = {
            "annotation_stats": {
                "bbox": 3000,
                "mask": 1500,
                "keypoint": 500,
            }
        }

        result = mock_metadata.get_dataset_stats("dataset_1")
        assert result["annotation_stats"]["bbox"] == 3000


class TestSearchFrames:
    """Test search_frames MCP tool"""

    def test_search_frames_by_time(self):
        """Search frames by time range"""
        mock_metadata = Mock()
        mock_metadata.search_frames.return_value = [
            {"frame_id": 100, "timestamp": 1704067200},
            {"frame_id": 101, "timestamp": 1704067201},
        ]

        result = mock_metadata.search_frames(
            "dataset_1",
            start_time=1704067200,
            end_time=1704067300,
        )
        assert len(result) == 2

    def test_search_frames_by_condition(self):
        """Search frames by environmental conditions"""
        mock_metadata = Mock()
        mock_metadata.search_frames.return_value = [
            {"frame_id": 50, "lighting": "bright", "weather": "sunny"},
        ]

        result = mock_metadata.search_frames(
            "dataset_1",
            lighting="bright",
            weather="sunny",
        )
        assert len(result) >= 1

    def test_search_frames_by_content(self):
        """Search frames by content/objects"""
        mock_metadata = Mock()
        mock_metadata.search_frames.return_value = [
            {"frame_id": 42, "objects": ["person", "car", "bicycle"]},
        ]

        result = mock_metadata.search_frames(
            "dataset_1",
            objects=["person"],
        )
        assert all("person" in f["objects"] for f in result)

    def test_search_frames_empty_result(self):
        """Handle no matching frames"""
        mock_metadata = Mock()
        mock_metadata.search_frames.return_value = []

        result = mock_metadata.search_frames(
            "dataset_1",
            objects=["nonexistent"],
        )
        assert len(result) == 0


class TestExportDatasetManifest:
    """Test export_dataset_manifest MCP tool"""

    def test_export_as_json(self):
        """Export dataset manifest as JSON"""
        mock_metadata = Mock()
        manifest = {
            "name": "dataset_1",
            "version": "1.0",
            "frames": 1000,
            "modalities": ["rgb", "depth"],
        }
        mock_metadata.export_dataset_manifest.return_value = json.dumps(
            manifest
        )

        result = mock_metadata.export_dataset_manifest(
            "dataset_1", format="json"
        )
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["name"] == "dataset_1"

    def test_export_as_csv(self):
        """Export dataset manifest as CSV"""
        mock_metadata = Mock()
        mock_metadata.export_dataset_manifest.return_value = (
            "name,version,frames\ndataset_1,1.0,1000"
        )

        result = mock_metadata.export_dataset_manifest(
            "dataset_1", format="csv"
        )
        assert "dataset_1" in result
        assert "frames" in result

    def test_export_as_parquet(self):
        """Export dataset manifest as Parquet"""
        mock_metadata = Mock()
        mock_metadata.export_dataset_manifest.return_value = b"parquet_data"

        result = mock_metadata.export_dataset_manifest(
            "dataset_1", format="parquet"
        )
        assert isinstance(result, bytes)

    def test_export_with_filters(self):
        """Export manifest with specific fields"""
        mock_metadata = Mock()
        mock_metadata.export_dataset_manifest.return_value = (
            "name,modalities\ndataset_1,rgb|depth"
        )

        result = mock_metadata.export_dataset_manifest(
            "dataset_1",
            format="csv",
            fields=["name", "modalities"],
        )
        assert "modalities" in result


class TestDetectFormatCompatibility:
    """Test detect_format_compatibility MCP tool"""

    def test_compatibility_with_framework(self):
        """Check dataset compatibility with ML framework"""
        mock_metadata = Mock()
        mock_metadata.detect_format_compatibility.return_value = {
            "framework": "pytorch",
            "compatible": True,
            "required_transforms": [],
            "warnings": [],
        }

        result = mock_metadata.detect_format_compatibility(
            "dataset_1", framework="pytorch"
        )
        assert result["compatible"] is True
        assert result["framework"] == "pytorch"

    def test_compatibility_issues_detected(self):
        """Detect compatibility issues"""
        mock_metadata = Mock()
        mock_metadata.detect_format_compatibility.return_value = {
            "framework": "tensorflow",
            "compatible": False,
            "issues": [
                "Incompatible modality: thermal not supported by TensorFlow",
                "Annotation format mismatch",
            ],
            "required_transforms": ["convert_to_tfrecord", "normalize_modalities"],
        }

        result = mock_metadata.detect_format_compatibility(
            "dataset_1", framework="tensorflow"
        )
        assert result["compatible"] is False
        assert len(result["issues"]) > 0

    def test_compatibility_with_loaders(self):
        """Check compatibility with data loaders"""
        mock_metadata = Mock()
        mock_metadata.detect_format_compatibility.return_value = {
            "loaders": [
                {"name": "ROSBagLoader", "compatible": True},
                {"name": "MCAPLoader", "compatible": True},
                {"name": "MindDataLoader", "compatible": False},
            ]
        }

        result = mock_metadata.detect_format_compatibility(
            "dataset_1"
        )
        compatible = [
            l for l in result["loaders"] if l["compatible"]
        ]
        assert len(compatible) >= 1

    def test_get_required_conversions(self):
        """Get list of required data conversions"""
        mock_metadata = Mock()
        mock_metadata.detect_format_compatibility.return_value = {
            "required_conversions": [
                "depth_uint16_to_float32",
                "rgb_bgr_to_rgb",
            ],
            "conversion_time_estimate_seconds": 300,
        }

        result = mock_metadata.detect_format_compatibility(
            "dataset_1"
        )
        assert "required_conversions" in result
        assert len(result["required_conversions"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
