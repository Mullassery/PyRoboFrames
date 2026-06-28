# Multimodal Sensor Fusion in PyRoboFrames v0.4.2

Time-synchronized RGB + depth + IMU sensor fusion for humanoid robot learning.

## Overview

Modern humanoid robots are equipped with multiple sensors:
- **RGB Cameras** (3×): head, front/chest, wrist (for vision-language models)
- **Depth Cameras** (1+): wrist (for manipulation, grasping)
- **IMU** (1+): shoulder/torso (for motion estimation, stability)

PyRoboFrames v0.4.2 provides APIs to:
1. **Time-align** sensors at different sampling rates (RGB 30Hz, depth 15Hz, IMU 300Hz)
2. **Project** depth to image plane using camera calibration
3. **Fuse** IMU motion with vision for egomotion compensation
4. **Create training batches** with synchronized multimodal data

## Quick Start

### Basic Multimodal Loading

```python
import pyroboframes as prf
from pyroboframes.sensor_fusion import MultimodalDataFrame, create_humanoid_config

# Load dataset
ds = prf.RoboFrameDataset.from_path("humanoid_dataset/")

# Create multimodal dataframe with humanoid config
config = create_humanoid_config()
mdf = MultimodalDataFrame(ds, config)

# Get time-synchronized batch
batch = mdf.align_multimodal(
    reference_topic="/camera/front/rgb",
    tolerance_ns=50_000_000,  # 50ms tolerance
)

# Access synchronized data
print(batch.cameras)          # ['camera.head', 'camera.front', 'camera.wrist']
print(batch.depth_streams)    # ['depth.wrist']
print(batch.imu_streams)      # ['imu.shoulder']
```

### Depth Projection

Project depth points to image plane using camera calibration:

```python
# Create camera calibrations
calibrations = {
    "camera.wrist": prf.CameraCalibration(
        name="wrist",
        fx=600, fy=600, cx=320, cy=240,
        width=640, height=480,
    ),
}

# Project depth to image
enhanced_batch = mdf.project_depth_to_image(batch, calibrations)

# Access reprojected coordinates
uv = enhanced_batch["depth.wrist.camera.wrist.reprojected_uv"]  # [batch, N_points, 2]
```

### IMU Fusion

Fuse IMU gyroscope data with vision for motion compensation:

```python
# Fuse IMU motion with vision
stabilized = mdf.fuse_imu_with_vision(batch, stabilization_alpha=0.9)

# Access motion estimates
rotation_mag = stabilized["imu.shoulder.rotation_magnitude"]  # Angular velocity norm
motion_conf = stabilized["imu.shoulder.motion_confidence"]    # 1.0 = still, 0.0 = fast motion

# For depth: confidence during camera motion
depth_conf = stabilized["depth.wrist.motion_confidence"]  # Down-weight during fast motion
```

## Architecture

### Time Alignment Strategy

The core innovation is **backward as-of join** — each reference timestamp gets the most recent (≤) sample from every other sensor:

```
RGB timestamps:    |-----|-----|-----|-----|  (30 Hz)
Depth timestamps:    |---------|---------|    (15 Hz, 50ms lag)
IMU timestamps:    |-|-|-|-|-|-|-|-|-|-|-|-|  (300 Hz)

Aligned result (on RGB grid):
  t1 → {rgb@t1, depth@t0 (previous), imu@t1}
  t2 → {rgb@t2, depth@t1 (newer),    imu@t2}
```

Tolerance filtering removes stale matches:

```python
tolerance_ns = 50_000_000  # 50ms
# Only use depth sample if age(depth) <= 50ms
```

### Multimodal Batch Structure

After alignment, data is organized as `MultimodalBatch` with prefixed keys:

```
camera.<camera_name>.<field>    # RGB data
depth.<depth_name>.<field>      # Depth/point cloud
imu.<imu_name>.<field>          # IMU data
```

Example:

```python
batch = mdf.align_multimodal()

# Access by topic
rgb_front = batch["camera.front.frame"]
depth_wrist = batch["depth.wrist.points"]
imu_gyro = batch["imu.shoulder.gyro"]

# Iterate all cameras
for cam in batch.cameras:
    print(f"Camera: {cam}")

# Iterate all depth streams
for depth in batch.depth_streams:
    print(f"Depth: {depth}")
```

## Configuration

### Humanoid Robot Setup (Default)

```python
from pyroboframes.sensor_fusion import create_humanoid_config

config = create_humanoid_config()
# reference_topic: "/camera/front/rgb"
# cameras: ["/camera/head/rgb", "/camera/front/rgb", "/camera/wrist/rgb"]
# depth: ["/depth/wrist"]
# imu: ["/imu/shoulder"]
# tolerance: 50ms
# alignment: linear interpolation
```

### Custom Configuration

```python
from pyroboframes.sensor_fusion import SensorFusionConfig

config = SensorFusionConfig(
    reference_topic="/camera/front/rgb",
    camera_topics=["/camera/head/rgb", "/camera/front/rgb"],
    depth_topics=["/depth/wrist"],
    imu_topics=["/imu/base", "/imu/shoulder"],
    tolerance_ns=50_000_000,  # 50ms
    align_method="linear",    # "previous", "nearest", "linear"
    apply_depth_projection=True,
    apply_imu_compensation=True,
)

mdf = MultimodalDataFrame(df, config)
```

## Sensor Modalities

### RGB Cameras

**Typical specs:**
- Resolution: 640×480 or 1280×960
- Frame rate: 30 Hz
- Latency: <10ms
- Use: Vision-language models, object detection, hand tracking

**Access:**
```python
for cam in batch.cameras:
    rgb = batch[f"{cam}.frame"]  # [batch, H, W, 3]
```

### Depth Cameras

**Formats:**
- Point clouds: `.xyz`, `.ply`, `.pcd` (ROS standard)
- Depth maps: `[H, W]` images (meters)
- Structured light: stereo matching, time-of-flight

**Typical specs:**
- Resolution: 480×640 (VGA) or 1080×1920 (HD)
- Frame rate: 15-30 Hz
- Latency: 50-100ms (typically behind RGB)
- Range: 0.1-10 meters (depth-dependent)

**Access:**
```python
for depth in batch.depth_streams:
    points = batch[f"{depth}.points"]  # [batch, N_points, 3] meters
    # OR
    depth_map = batch[f"{depth}.depth_map"]  # [batch, H, W] meters
```

### IMU (Accelerometer + Gyroscope)

**Data:**
- Accelerometer: [ax, ay, az] (m/s²) — includes gravity
- Gyroscope: [gx, gy, gz] (rad/s) — rotational velocity
- Sampling: 100-300 Hz (much higher than vision)

**Uses:**
- Motion estimation (odometry)
- Motion compensation (de-rotation, stabilization)
- Camera shake detection
- Kinematic model supervision

**Access:**
```python
for imu in batch.imu_streams:
    accel = batch[f"{imu}.accel"]  # [batch, 3] m/s²
    gyro = batch[f"{imu}.gyro"]    # [batch, 3] rad/s
    
# After fusion:
motion_conf = batch[f"{imu}.motion_confidence"]  # [batch] (0-1)
```

## Training Workflows

### Vision-Only

```python
# Ignore depth and IMU, use only RGB
batch = mdf.align_multimodal()

# Stack RGB from 3 cameras
rgb_stack = np.concatenate([
    batch["camera.head.frame"],
    batch["camera.front.frame"],
    batch["camera.wrist.frame"],
], axis=-1)  # [batch, H, W, 9] (3 cameras × 3 channels)

# Feed to VLA model
action = model(rgb_stack)
```

### Vision + Depth (Manipulation)

```python
# Use RGB + depth for manipulation tasks
batch = mdf.align_multimodal()

rgb = batch["camera.wrist.frame"]      # [batch, H, W, 3]
depth = batch["depth.wrist.depth_map"]  # [batch, H, W]

# Concatenate
rgbd = np.concatenate([rgb, depth[:, :, :, np.newaxis]], axis=-1)  # [batch, H, W, 4]

# Prediction
grasp = model(rgbd)  # Grasp point in image coordinates
```

### Egomotion Compensation

```python
# Use IMU to estimate camera motion
batch = mdf.align_multimodal()

gyro = batch["imu.shoulder.gyro"]       # [batch, 3] rad/s
motion_conf = batch["imu.shoulder.motion_confidence"]  # [batch]

# Down-weight depth measurements during high motion
depth = batch["depth.wrist.depth_map"]
depth_conf = batch["depth.wrist.motion_confidence"]  # [batch, H, W]

# Only use high-confidence depth
valid_depth = depth * depth_conf  # Zero out during motion
```

### Full Multimodal

```python
# Combine RGB + depth + IMU + state for full humanoid learning
batch = mdf.align_multimodal()

# Image modality: 3× RGB + 1× depth
rgb_head = batch["camera.head.frame"]
rgb_chest = batch["camera.front.frame"]
rgb_wrist = batch["camera.wrist.frame"]
depth = batch["depth.wrist.depth_map"]

# Proprioception: robot state from actuators
state = batch["observation.state"]  # Joint angles, velocities

# Motion: IMU for acceleration context
accel = batch["imu.shoulder.accel"]

# Stack all modalities
features = {
    "vision": (rgb_head, rgb_chest, rgb_wrist, depth),
    "proprioception": state,
    "motion": accel,
}

# Train multimodal transformer
action, confidence = model(features)
```

## API Reference

### MultimodalDataFrame

```python
class MultimodalDataFrame:
    def __init__(
        self,
        df: RoboticsDataFrame,
        config: Optional[SensorFusionConfig] = None,
    ) -> None:
        """Create multimodal frame from RoboticsDataFrame."""

    def align_multimodal(
        self,
        reference_topic: Optional[str] = None,
        tolerance_ns: Optional[int] = None,
    ) -> MultimodalBatch:
        """Time-synchronize all sensors (backward as-of join)."""

    def project_depth_to_image(
        self,
        batch: MultimodalBatch,
        camera_calibrations: dict[str, CameraCalibration] = None,
    ) -> MultimodalBatch:
        """Project depth to image plane using calibration."""

    def fuse_imu_with_vision(
        self,
        batch: MultimodalBatch,
        stabilization_alpha: float = 0.9,
    ) -> MultimodalBatch:
        """Fuse IMU with vision for motion compensation."""

    def stack_batch_for_training(
        self,
        batch: MultimodalBatch,
        stack_frames: int = 1,
    ) -> dict[str, np.ndarray]:
        """Convert to training-ready format."""
```

### MultimodalBatch

```python
class MultimodalBatch:
    def __init__(self, data: dict[str, np.ndarray]) -> None:
        """Container for aligned multimodal data."""

    @property
    def log_time(self) -> np.ndarray:
        """Timestamps (nanoseconds)."""

    @property
    def cameras(self) -> list[str]:
        """Camera stream names."""

    @property
    def depth_streams(self) -> list[str]:
        """Depth stream names."""

    @property
    def imu_streams(self) -> list[str]:
        """IMU stream names."""

    def __getitem__(self, key: str) -> np.ndarray:
        """Access sensor stream by name."""

    def __len__(self) -> int:
        """Number of samples."""
```

## Performance

**Typical throughput (on Apple Silicon M3):**
- Time alignment: ~100K timestamps/second
- Depth projection: ~50K points/second
- IMU fusion: negligible (<1%)

**Memory overhead:**
- Per-sample overhead: ~100 bytes (metadata)
- Depth batch [8, 480, 640]: ~10 MB
- RGB batch [8, 480, 640, 3]: ~6 MB per camera

## Limitations & Future Work

**v0.4.2 Limitations:**
- No compression for point clouds (raw storage)
- Single reference topic (no multi-leader alignment)
- Linear interpolation only for resampling
- No automatic outlier filtering for depth

**Planned (v0.5.0+):**
- Point cloud compression (lossy quantization)
- Multi-camera calibration (extrinsics)
- Depth hole-filling and inpainting
- Temporal coherence constraints
- Automatic motion detection

## Examples

See `examples/humanoid_multimodal_fusion.py` for a complete humanoid robot example including:
- Loading multimodal data
- Time alignment with tolerance
- Depth projection to image
- Training loop setup
- Batch stacking for neural networks

## References

- OpenCV Camera Calibration
- ROS Perception: tf + calibration
- LeRobot v3.0 dataset format
- Open X-Embodiment multimodal specs
