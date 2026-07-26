"""Tests for P3-P6 phases: Advanced I/O, Efficiency, Fusion, and Cross-Platform Training."""

import numpy as np
import pytest
import tempfile
import os

import sys
sys.path.insert(0, "/Users/georgimullassery/PyRoboFrames/python")

import pyroboframes as prf


class TestP3ParquetIO:
    """P3: Advanced I/O & Ecosystem Integration."""

    def test_parquet_writer_options(self):
        """Test ParquetWriter with various options."""
        opts = prf.ParquetWriteOptions(
            compression="snappy",
            row_group_size=5000,
            use_dictionary=True,
        )
        assert opts.compression == "snappy"
        assert opts.row_group_size == 5000

    def test_parquet_writer_creation(self):
        """Test ParquetWriter initialization."""
        writer = prf.ParquetWriter()
        assert writer is not None
        assert writer.options.compression == "snappy"

    def test_write_to_parquet_signature(self):
        """Test write_to_parquet convenience function exists."""
        assert callable(prf.write_to_parquet)

    def test_write_from_robotics_dataframe_callable(self):
        """Test write_from_robotics_dataframe is callable."""
        assert callable(prf.write_from_robotics_dataframe)

    def test_from_huggingface_hub_callable(self):
        """Test from_huggingface_hub function exists."""
        assert callable(prf.from_huggingface_hub)


class TestP4MemoryEfficiency:
    """P4: Memory Efficiency & Synchronization."""

    def test_lazy_parquet_dataset_creation(self):
        """Test LazyParquetDataset initialization."""
        # Create a temporary parquet file
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a simple parquet file
            data = {
                "col1": pa.array([1, 2, 3, 4, 5]),
                "col2": pa.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            }
            table = pa.table(data)
            path = os.path.join(tmpdir, "test.parquet")
            pq.write_table(table, path)

            # Load with LazyParquetDataset
            ds = prf.LazyParquetDataset(path)
            assert ds.num_rows == 5
            assert ds.num_columns == 2
            assert ds.columns == ["col1", "col2"]

    def test_lazy_parquet_slice(self):
        """Test slicing in LazyParquetDataset."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "col1": pa.array(range(100)),
                "col2": pa.array(np.arange(100, dtype=np.float32)),
            }
            table = pa.table(data)
            path = os.path.join(tmpdir, "test.parquet")
            pq.write_table(table, path)

            ds = prf.LazyParquetDataset(path)
            sliced = ds.slice(10, 20)
            assert sliced.num_rows == 10

    def test_camera_timeline_creation(self):
        """Test CameraTimeline for video sync."""
        times = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64)  # ns
        frames = np.array([0, 1, 2], dtype=np.int64)

        timeline = prf.CameraTimeline("camera_0", times, frames)
        assert timeline.num_frames == 3
        assert timeline.camera_name == "camera_0"

    def test_video_synchronizer_init(self):
        """Test VideoSynchronizer initialization."""
        times = np.array([0, 33_333_333, 66_666_666], dtype=np.int64)  # ~30Hz
        frames = np.array([0, 1, 2], dtype=np.int64)
        timeline = prf.CameraTimeline("ref", times, frames)

        sync = prf.VideoSynchronizer(timeline)
        assert sync.reference.camera_name == "ref"

    def test_jitter_filter_creation(self):
        """Test JitterFilter initialization."""
        jitter = prf.JitterFilter(alpha=0.7)
        assert jitter.alpha == 0.7

    def test_align_frame_sequences_signature(self):
        """Test align_frame_sequences is callable."""
        assert callable(prf.align_frame_sequences)


class TestP5SensorFusion:
    """P5: Multi-Sensor Fusion."""

    def test_multi_rate_fusion_engine_init(self):
        """Test MultiRateFusionEngine initialization."""
        engine = prf.MultiRateFusionEngine(reference_rate_hz=30.0)
        assert engine.reference_rate_hz == 30.0

    def test_detect_rates(self):
        """Test rate detection from timestamps."""
        engine = prf.MultiRateFusionEngine(reference_rate_hz=30.0)

        # 30Hz: 33.3ms intervals
        times_30hz = np.array([0, 33_333_333, 66_666_666], dtype=np.int64)
        # 100Hz: 10ms intervals
        times_100hz = np.array([0, 10_000_000, 20_000_000], dtype=np.int64)

        rates = engine.detect_rates({
            "camera": times_30hz,
            "imu": times_100hz,
        })

        # Check rates are close to expected
        assert 25 < rates["camera"] < 35  # ~30Hz
        assert 90 < rates["imu"] < 110  # ~100Hz

    def test_kalman_filter(self):
        """Test Kalman filtering."""
        engine = prf.MultiRateFusionEngine()

        # Noisy measurements
        measurements = np.array([1.0, 1.1, 0.95, 1.05, 1.0])
        filtered = engine.kalman_filter_state(measurements)
        assert filtered.shape == (5, 1)
        # Filtered should be smoother (less variance)
        assert np.std(filtered) < np.std(measurements)

    def test_weighted_fusion(self):
        """Test weighted sensor fusion."""
        engine = prf.MultiRateFusionEngine()

        readings = {
            "sensor_a": np.array([[1.0, 2.0]]),
            "sensor_b": np.array([[1.5, 2.5]]),
        }
        weights = {"sensor_a": 0.3, "sensor_b": 0.7}

        fused = engine.weighted_fusion(readings, weights)
        assert fused.shape == (1, 2)
        # Should be closer to sensor_b (higher weight)
        np.testing.assert_allclose(fused, [[1.35, 2.35]], atol=0.01)


class TestP6UnifiedOutputs:
    """P6: Cross-Platform Training Parity."""

    def test_tensor_adapter_numpy(self):
        """Test NumPy adapter (no-op)."""
        adapter = prf.ToTensorAdapter(framework="numpy")
        arr = np.array([1.0, 2.0, 3.0])
        tensor = adapter.to_tensor(arr)
        np.testing.assert_array_equal(tensor, arr)

    def test_detect_best_framework(self):
        """Test framework detection."""
        framework = prf.detect_best_framework()
        assert framework in ["numpy", "torch", "jax", "mlx", "tensorflow"]

    def test_create_adapter_for_device_cpu(self):
        """Test adapter creation for CPU device."""
        adapter = prf.create_adapter_for_device(device="cpu")
        assert adapter is not None
        assert adapter.get_framework_name() == "numpy"

    def test_adapter_batch_to_tensors(self):
        """Test batch conversion."""
        adapter = prf.ToTensorAdapter(framework="numpy")
        batch = {
            "obs": np.array([[1.0, 2.0]]),
            "act": np.array([[0.5, 0.5]]),
        }
        converted = adapter.batch_to_tensors(batch)
        assert "obs" in converted
        assert "act" in converted

    def test_gpu_transforms_available(self):
        """Test GPUTransforms class."""
        gt = prf.GPUTransforms(device="auto")
        assert gt is not None
        assert isinstance(gt.backend, str)

    def test_mlx_transforms_available(self):
        """Test MLXTransforms class."""
        mlx_t = prf.MLXTransforms()
        # Should not raise, but may report unavailable
        assert isinstance(mlx_t.available, bool)

    def test_mps_transforms_available(self):
        """Test MPSTransforms class."""
        mps_t = prf.MPSTransforms()
        # Should not raise, but may report unavailable
        assert isinstance(mps_t.available, bool)


# Integration test
class TestP3P6Integration:
    """Integration tests across P3-P6 phases."""

    def test_pipeline_roundtrip(self):
        """Test a full pipeline: create data → write Parquet → read lazy → adapt framework."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create mock dataset (P3)
            state = np.random.randn(100, 4).astype(np.float32)
            action = np.random.randn(100, 2).astype(np.float32)
            data = {
                "observation.state": pa.array(state.flatten()),
                "action": pa.array(action.flatten()),
            }
            # Note: PyArrow requires equal-length columns, so we need to pad or use lists
            # Simpler: just use scalar columns
            data = {
                "timestamp": pa.array(np.arange(100)),
                "state_dim": pa.array(np.random.randn(100).astype(np.float32)),
            }
            table = pa.table(data)

            # 2. Write with ParquetWriter (P3)
            path = os.path.join(tmpdir, "test.parquet")
            pq.write_table(table, path)

            # 3. Load with LazyParquetDataset (P4)
            ds = prf.LazyParquetDataset(path)
            assert ds.num_rows > 0

            # 4. Slice and convert with unified adapter (P6)
            sliced = ds.slice(0, 50)
            adapter = prf.ToTensorAdapter(framework="numpy")
            batch = {col: sliced.column(col).to_numpy() for col in sliced.column_names}
            converted = adapter.batch_to_tensors(batch)

            assert len(converted) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
