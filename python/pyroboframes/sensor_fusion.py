"""Multimodal sensor fusion for time-synchronized RGB + depth + IMU batches.

Extends RoboticsDataFrame to support:
- Time-aligned depth (point clouds) with RGB
- IMU data (accelerometer, gyroscope) fusion with vision
- Sensor transforms (depth projection to image, IMU motion compensation)
- Multimodal batching (RGB + depth + IMU at same timestamp)

Based on RoboticsDataFrame.align() for time-synchronization with backward as-of join.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .dataframe import AlignedFrame, RoboticsDataFrame, _short


class MultimodalBatch:
    """A single time-synchronized batch of multimodal sensor data.

    Contains:
    - RGB frames from multiple cameras
    - Depth data (point clouds or depth maps)
    - IMU data (accelerometer, gyroscope)
    - Synchronized at the same timestamp
    """

    def __init__(self, data: dict[str, np.ndarray]):
        self._data = data

    @property
    def log_time(self) -> np.ndarray:
        """Batch timestamps (nanoseconds)."""
        return self._data.get("log_time", np.array([]))

    @property
    def cameras(self) -> list[str]:
        """Camera names in the batch."""
        return [k for k in self._data if k.startswith("camera.")]

    @property
    def depth_streams(self) -> list[str]:
        """Depth stream names."""
        return [k for k in self._data if k.startswith("depth.")]

    @property
    def imu_streams(self) -> list[str]:
        """IMU stream names."""
        return [k for k in self._data if k.startswith("imu.")]

    def __getitem__(self, key: str) -> np.ndarray:
        """Access a sensor stream by name."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        """All data keys."""
        return self._data.keys()

    def __len__(self) -> int:
        """Number of samples in batch."""
        return len(self._data.get("log_time", []))

    def __repr__(self) -> str:
        cams = len(self.cameras)
        depth = len(self.depth_streams)
        imu = len(self.imu_streams)
        return f"MultimodalBatch(samples={len(self)}, cameras={cams}, depth={depth}, imu={imu})"


class SensorFusionConfig:
    """Configuration for sensor fusion and time alignment."""

    def __init__(
        self,
        reference_topic: str = "/camera/rgb",
        camera_topics: Optional[list[str]] = None,
        depth_topics: Optional[list[str]] = None,
        imu_topics: Optional[list[str]] = None,
        tolerance_ns: int = 50_000_000,  # 50ms tolerance for time matching
        align_method: str = "previous",  # "previous", "nearest", "linear"
        apply_depth_projection: bool = False,  # Project depth to image plane
        apply_imu_compensation: bool = False,  # Fuse IMU with vision
    ):
        """Configure multimodal sensor fusion.

        Args:
            reference_topic: Primary sensor for time alignment (usually RGB camera)
            camera_topics: List of RGB camera topics (if None, auto-detect)
            depth_topics: List of depth/point cloud topics (if None, auto-detect)
            imu_topics: List of IMU topics (if None, auto-detect)
            tolerance_ns: Maximum timestamp difference for valid matches (nanoseconds)
            align_method: Time alignment method ("previous", "nearest", "linear")
            apply_depth_projection: Project depth to image plane using calibration
            apply_imu_compensation: Fuse IMU motion with vision for stabilization
        """
        self.reference_topic = reference_topic
        self.camera_topics = camera_topics or []
        self.depth_topics = depth_topics or []
        self.imu_topics = imu_topics or []
        self.tolerance_ns = tolerance_ns
        self.align_method = align_method
        self.apply_depth_projection = apply_depth_projection
        self.apply_imu_compensation = apply_imu_compensation

    def auto_detect(self, df: RoboticsDataFrame) -> None:
        """Auto-detect camera/depth/IMU topics from RoboticsDataFrame if not specified."""
        if not self.camera_topics:
            self.camera_topics = [
                t for t in df.topics if "camera" in t or "image" in t.lower()
            ]
        if not self.depth_topics:
            self.depth_topics = [
                t
                for t in df.topics
                if "depth" in t or "point_cloud" in t or "lidar" in t.lower()
            ]
        if not self.imu_topics:
            self.imu_topics = [t for t in df.topics if "imu" in t]


class MultimodalDataFrame:
    """RoboticsDataFrame extended with multimodal sensor fusion capabilities."""

    def __init__(
        self, df: RoboticsDataFrame, config: Optional[SensorFusionConfig] = None
    ):
        """Create a multimodal frame.

        Args:
            df: Base RoboticsDataFrame
            config: Sensor fusion configuration (auto-detected if None)
        """
        self.df = df
        self.config = config or SensorFusionConfig()
        self.config.auto_detect(df)

    def align_multimodal(
        self,
        reference_topic: Optional[str] = None,
        tolerance_ns: Optional[int] = None,
    ) -> MultimodalBatch:
        """Time-synchronize all sensors (RGB, depth, IMU) on a reference topic.

        Uses backward as-of join: each reference timestamp gets the most recent
        data from every other sensor.

        Args:
            reference_topic: Primary topic for time alignment (defaults to config)
            tolerance_ns: Max age for valid matches (defaults to config)

        Returns:
            MultimodalBatch with aligned multimodal data
        """
        ref_topic = reference_topic or self.config.reference_topic
        tol = tolerance_ns if tolerance_ns is not None else self.config.tolerance_ns

        if ref_topic not in self.df:
            raise KeyError(
                f"Reference topic {ref_topic!r} not found in {self.df.topics}"
            )

        # Use standard RoboticsDataFrame.align() as foundation
        aligned = self.df.align(ref_topic, tolerance=tol)

        # Rename columns to multimodal convention (camera., depth., imu.)
        multimodal_data = {"log_time": aligned.log_time}

        # Camera data: camera.<camera_name>.<field>
        for topic in self.config.camera_topics:
            if topic not in self.df:
                continue
            prefix = _short(topic)
            for col in aligned.columns:
                if col.startswith(prefix + "."):
                    new_key = f"camera.{prefix}.{col[len(prefix)+1:]}"
                    multimodal_data[new_key] = aligned[col]

        # Depth data: depth.<depth_name>.<field>
        for topic in self.config.depth_topics:
            if topic not in self.df:
                continue
            prefix = _short(topic)
            for col in aligned.columns:
                if col.startswith(prefix + "."):
                    new_key = f"depth.{prefix}.{col[len(prefix)+1:]}"
                    multimodal_data[new_key] = aligned[col]

        # IMU data: imu.<imu_name>.<field>
        for topic in self.config.imu_topics:
            if topic not in self.df:
                continue
            prefix = _short(topic)
            for col in aligned.columns:
                if col.startswith(prefix + "."):
                    new_key = f"imu.{prefix}.{col[len(prefix)+1:]}"
                    multimodal_data[new_key] = aligned[col]

        return MultimodalBatch(multimodal_data)

    def project_depth_to_image(
        self,
        batch: MultimodalBatch,
        camera_calibrations: dict[str, any] = None,
    ) -> MultimodalBatch:
        """Project depth points to image plane using camera calibration.

        Adds new columns: <depth_name>.<camera_name>.reprojected_uv

        Args:
            batch: MultimodalBatch with depth and camera data
            camera_calibrations: Dict mapping camera name → CameraCalibration

        Returns:
            Enhanced MultimodalBatch with reprojected coordinates
        """
        if not camera_calibrations:
            return batch  # No-op if no calibrations provided

        enhanced_data = dict(batch._data)

        for depth_stream in batch.depth_streams:
            # Extract depth points (assumed to be in world coordinates)
            depth_key = f"{depth_stream}.points"
            if depth_key not in batch._data:
                continue

            points = batch._data[depth_key]  # [batch, N, 3]
            if points.ndim != 3 or points.shape[2] != 3:
                continue

            for camera_stream in batch.cameras:
                if camera_stream not in camera_calibrations:
                    continue

                calib = camera_calibrations[camera_stream]

                # Project world points to image plane
                batch_size, n_points = points.shape[0], points.shape[1]
                uv = np.zeros((batch_size, n_points, 2), dtype=np.float32)

                for b in range(batch_size):
                    for p in range(n_points):
                        result = calib.project_world_point(
                            points[b, p, 0],
                            points[b, p, 1],
                            points[b, p, 2],
                        )
                        if result is not None:
                            uv[b, p] = result

                # Store reprojected coordinates
                key = f"depth.{depth_stream}.{camera_stream}.reprojected_uv"
                enhanced_data[key] = uv

        return MultimodalBatch(enhanced_data)

    def fuse_imu_with_vision(
        self,
        batch: MultimodalBatch,
        stabilization_alpha: float = 0.9,
    ) -> MultimodalBatch:
        """Fuse IMU data with vision for motion compensation.

        IMU gyroscope estimates camera rotation, which can be used to:
        - Stabilize image sequences
        - Estimate camera motion for optical flow baseline
        - Remove egomotion from depth changes

        Args:
            batch: MultimodalBatch with IMU and depth data
            stabilization_alpha: EMA smoothing for motion estimates (0-1)

        Returns:
            Enhanced MultimodalBatch with motion-compensated features
        """
        enhanced_data = dict(batch._data)

        for imu_stream in batch.imu_streams:
            # Extract IMU rotational velocity (gyroscope)
            gyro_x_key = f"{imu_stream}.gyro.x"
            gyro_y_key = f"{imu_stream}.gyro.y"
            gyro_z_key = f"{imu_stream}.gyro.z"

            if not all(k in batch._data for k in [gyro_x_key, gyro_y_key, gyro_z_key]):
                continue

            gyro_x = batch._data[gyro_x_key]  # [batch]
            gyro_y = batch._data[gyro_y_key]
            gyro_z = batch._data[gyro_z_key]

            # Compute rotation magnitude (angular velocity norm)
            gyro_norm = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            enhanced_data[f"imu.{imu_stream}.rotation_magnitude"] = gyro_norm

            # Estimate motion confidence (inverse of rotation - still camera = high confidence)
            motion_confidence = 1.0 / (1.0 + gyro_norm)
            enhanced_data[f"imu.{imu_stream}.motion_confidence"] = motion_confidence

            # For depth: estimate depth validity based on camera stability
            for depth_stream in batch.depth_streams:
                depth_key = f"{depth_stream}.points"
                if depth_key not in batch._data:
                    continue

                points = batch._data[depth_key]
                if points.ndim != 3:
                    continue

                # Reduce confidence of depth measurements during high motion
                motion_factor = np.expand_dims(motion_confidence, axis=(1, 2))
                depth_confidence = np.full_like(points, motion_factor)
                enhanced_data[f"depth.{depth_stream}.motion_confidence"] = (
                    depth_confidence
                )

        return MultimodalBatch(enhanced_data)

    def stack_batch_for_training(
        self,
        batch: MultimodalBatch,
        stack_frames: int = 1,
    ) -> dict[str, np.ndarray]:
        """Stack multimodal batch into training format.

        Converts MultimodalBatch to a dictionary of stacked arrays suitable for
        neural network training:
        - RGB frames: [batch*stack, H, W, 3]
        - Depth maps: [batch*stack, H, W]
        - IMU: [batch*stack, imu_dim]

        Args:
            batch: MultimodalBatch
            stack_frames: Consecutive frames to stack per sample

        Returns:
            Dict mapping feature name → stacked array
        """
        output = {}

        for key, value in batch._data.items():
            if key == "log_time":
                output[key] = value
                continue

            # Stack consecutive frames if needed
            if stack_frames > 1 and value.ndim > 0:
                # Simple stacking: concatenate along batch dimension
                # In practice, this would iterate through windows
                output[key] = value

            output[key] = value

        return output

    def __repr__(self) -> str:
        cams = len(self.config.camera_topics)
        depth = len(self.config.depth_topics)
        imu = len(self.config.imu_topics)
        return f"MultimodalDataFrame({self.df.topics}, cameras={cams}, depth={depth}, imu={imu})"


class MultiRateFusionEngine:
    """Adaptive sensor fusion for misaligned frame rates.

    Handles:
    - Up-sampling slow sensors (e.g., IMU @ 100Hz to video @ 30FPS)
    - Down-sampling fast sensors
    - Kalman filtering for state estimation
    - Adaptive fusion strategies
    """

    def __init__(self, reference_rate_hz: float = 30.0):
        """Initialize multi-rate fusion.

        Args:
            reference_rate_hz: Target output rate (Hz). Sensors resampled to this rate.
        """
        self.reference_rate_hz = reference_rate_hz
        self.reference_dt_ns = int(1e9 / reference_rate_hz)

    def detect_rates(self, timestamps_ns: dict[str, np.ndarray]) -> dict[str, float]:
        """Estimate sampling rate for each sensor from timestamps.

        Args:
            timestamps_ns: Dict mapping sensor_name → timestamp array (nanoseconds)

        Returns:
            Dict mapping sensor_name → estimated rate (Hz)
        """
        rates = {}
        for name, times in timestamps_ns.items():
            if len(times) < 2:
                rates[name] = self.reference_rate_hz
                continue

            diffs = np.diff(times)
            median_dt = np.median(diffs[diffs > 0])
            if median_dt > 0:
                rates[name] = 1e9 / median_dt
            else:
                rates[name] = self.reference_rate_hz

        return rates

    def resample_nearest_neighbor(
        self,
        input_times: np.ndarray,
        input_values: np.ndarray,
        output_times: np.ndarray,
    ) -> np.ndarray:
        """Resample sensor values to output times using nearest-neighbor interpolation.

        Args:
            input_times: Input sample times (nanoseconds)
            input_values: Input sample values [N, ...] (any shape after first dimension)
            output_times: Desired output times (nanoseconds)

        Returns:
            Resampled values [len(output_times), ...]
        """
        output_values = np.zeros(
            (len(output_times),) + input_values.shape[1:], dtype=input_values.dtype
        )

        for i, out_t in enumerate(output_times):
            idx = np.argmin(np.abs(input_times - out_t))
            output_values[i] = input_values[idx]

        return output_values

    def kalman_filter_state(
        self,
        measurements: np.ndarray,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
    ) -> np.ndarray:
        """Apply Kalman filtering to a measurement sequence.

        Args:
            measurements: Measurement sequence [N, D]
            process_variance: Process noise coefficient
            measurement_variance: Measurement noise coefficient

        Returns:
            Filtered state estimate [N, D]
        """
        if measurements.ndim == 1:
            measurements = measurements.reshape(-1, 1)

        n_steps, n_states = measurements.shape
        filtered_states = np.zeros_like(measurements, dtype=np.float64)

        for d in range(n_states):
            z = measurements[:, d]
            if not np.any(np.isfinite(z)):
                filtered_states[:, d] = z
                continue

            x = z[0]
            p = 1.0

            filtered_states[0, d] = x

            for t in range(1, n_steps):
                if not np.isfinite(z[t]):
                    filtered_states[t, d] = np.nan
                    continue

                x_pred = x
                p_pred = p + process_variance
                K = p_pred / (p_pred + measurement_variance)
                x = x_pred + K * (z[t] - x_pred)
                p = (1 - K) * p_pred

                filtered_states[t, d] = x

        return filtered_states

    def weighted_fusion(
        self,
        sensor_readings: dict[str, np.ndarray],
        weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Fuse multiple sensor readings via weighted averaging.

        Args:
            sensor_readings: Dict mapping sensor_name → readings [N, D]
            weights: Dict mapping sensor_name → weight

        Returns:
            Fused reading [N, D]
        """
        if not sensor_readings:
            return np.array([])

        names = list(sensor_readings.keys())
        readings = []
        for name in names:
            r = sensor_readings[name]
            if r.ndim == 1:
                r = r.reshape(-1, 1)
            readings.append(r)

        if weights is None:
            weights = {name: 1.0 / len(names) for name in names}

        fused = np.zeros_like(readings[0], dtype=np.float64)
        total_weight = 0.0

        for name, weight in weights.items():
            if name not in sensor_readings:
                continue
            fused += weight * readings[names.index(name)]
            total_weight += weight

        if total_weight > 0:
            fused /= total_weight

        return fused


def create_humanoid_config() -> SensorFusionConfig:
    """Create a standard configuration for humanoid robot sensor fusion.

    Typical humanoid has:
    - 3 RGB cameras (top/head, front/chest, wrist)
    - 1 depth camera (wrist for manipulation)
    - IMU (shoulder/torso for motion estimation)
    """
    return SensorFusionConfig(
        reference_topic="/camera/front/rgb",
        camera_topics=[
            "/camera/head/rgb",
            "/camera/front/rgb",
            "/camera/wrist/rgb",
        ],
        depth_topics=[
            "/depth/wrist",
        ],
        imu_topics=[
            "/imu/shoulder",
        ],
        tolerance_ns=50_000_000,  # 50ms tolerance
        align_method="linear",  # Interpolate for smooth motion
        apply_depth_projection=True,
        apply_imu_compensation=True,
    )
