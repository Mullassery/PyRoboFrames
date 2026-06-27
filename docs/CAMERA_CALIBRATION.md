# Camera Calibration in PyRoboFrames

PyRoboFrames provides camera calibration APIs for storing intrinsic parameters (focal length, principal point), distortion models, and camera poses. This enables proper 3D reconstruction from depth data and multi-camera alignment.

## Concepts

### Intrinsic Parameters

The **intrinsic matrix** (K matrix) maps 3D camera-frame points to 2D image pixels:

```
[u]   [fx  0 cx] [X]
[v] = [ 0 fy cy] [Y]
[1]   [ 0  0  1] [Z]
```

Where:
- **fx, fy**: Focal lengths (in pixels). Larger = more zoom, smaller FOV.
- **cx, cy**: Principal point (image center, in pixels). Typically near the image center (e.g., cx=320 for 640-pixel width).
- **X, Y, Z**: 3D point in camera frame (meters).
- **u, v**: Projected pixel coordinates.

### Distortion Model

Camera lenses introduce radial and tangential distortion:

```
Radial:      1 + k1*r² + k2*r⁴ + k3*r⁶
Tangential:  p1, p2 terms
```

Most cameras need only k1 and k2. High-quality cameras may have k1, k2, k3.

### Camera Pose

The **extrinsic matrix** (pose) stores the camera's position and orientation in world coordinates:

```
[R|t]  where R is 3×3 rotation, t is 3×1 translation
```

Transforms 3D world points to camera frame: **p_camera = R @ p_world + t**

## Quick Start

### Creating Camera Intrinsics

```python
import pyroboframes as prf

# Create intrinsics (no distortion, identity pose)
intr = prf.CameraIntrinsics(
    fx=500.0,      # focal length x (pixels)
    fy=500.0,      # focal length y (pixels)
    cx=320.0,      # principal point x (pixels)
    cy=240.0,      # principal point y (pixels)
    width=640,     # image width (pixels)
    height=480,    # image height (pixels)
)

# Get K matrix (3×3)
k_matrix = intr.k_matrix()
print(k_matrix)
# [[500.0,   0.0, 320.0],
#  [  0.0, 500.0, 240.0],
#  [  0.0,   0.0,   1.0]]

# Project a 3D point to image
u, v = intr.project(x=1.0, y=1.0, z=2.0)  # 3D point in camera frame
print(f"Pixel: ({u:.1f}, {v:.1f})")
# Pixel: (570.0, 490.0)

# Unproject a pixel to 3D ray direction (unit vector)
direction = intr.unproject_direction(u=320.0, v=240.0)  # principal point
print(direction)
# [0.0, 0.0, 1.0]  (looking along +Z axis)
```

### Creating Camera Calibration (with distortion & pose)

```python
# Basic calibration (intrinsics only, identity pose)
calib = prf.CameraCalibration(
    name="observation.images.top",
    fx=500.0,
    fy=500.0,
    cx=320.0,
    cy=240.0,
    width=640,
    height=480,
)

# Project world 3D point (with pose transform)
u, v = calib.project_world_point(x=1.0, y=1.0, z=2.0)

# Unproject pixel to world ray (origin + direction)
origin, direction = calib.unproject_to_world_ray(u=320.0, v=240.0)
print(f"Camera at: {origin}")
print(f"Ray direction: {direction}")
```

## Multi-Camera Setup

Typical robots have 3-4 cameras with different focal lengths and positions:

```python
import pyroboframes as prf

# Top camera (wide angle, used for navigation)
top_cam = prf.CameraCalibration(
    name="observation.images.top",
    fx=480.0, fy=480.0,    # wider FOV
    cx=320.0, cy=240.0,
    width=640, height=480,
)

# Front camera (higher quality, used for manipulation)
front_cam = prf.CameraCalibration(
    name="observation.images.front",
    fx=600.0, fy=600.0,    # tighter FOV, more details
    cx=352.0, cy=288.0,
    width=704, height=576,
)

# Wrist camera (on end-effector)
wrist_cam = prf.CameraCalibration(
    name="observation.images.wrist",
    fx=600.0, fy=600.0,
    cx=320.0, cy=240.0,
    width=640, height=480,
)

# Store in dataset metadata (future version)
calibrations = [top_cam, front_cam, wrist_cam]
```

## Obtaining Calibration Parameters

### From Manufacturer Datasheets

Most depth cameras (RealSense, Oak-D, etc.) include calibration data:

```bash
# RealSense D435
# Datasheet: focal_length_x = 384.0, focal_length_y = 384.0
# Principal point: 320.5, 240.5
# Image: 640×480

intr = prf.CameraIntrinsics(384.0, 384.0, 320.5, 240.5, 640, 480)
```

### Using OpenCV Calibration

If you have checkerboard calibration results from OpenCV:

```python
import cv2
import numpy as np
import pyroboframes as prf

# Load calibration from OpenCV
with np.load("calibration.npz") as data:
    K = data["K"]  # intrinsic matrix
    D = data["D"]  # distortion coefficients

# Extract parameters from K
fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# Create PyRoboFrames calibration
calib = prf.CameraCalibration(
    name="camera",
    fx=fx, fy=fy,
    cx=cx, cy=cy,
    width=640, height=480,
)
```

### Using RealSense SDK

```python
import pyrealsense2 as rs
import pyroboframes as prf

# Get intrinsics from RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)
pipeline.start(config)

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
intr = color_frame.profile.as_video_stream_profile().intrinsics

# Create PyRoboFrames calibration
calib = prf.CameraCalibration(
    name="realsense_color",
    fx=intr.fx,
    fy=intr.fy,
    cx=intr.ppx,
    cy=intr.ppy,
    width=intr.width,
    height=intr.height,
)
```

## Depth Map to 3D Point Cloud

Once you have calibration, convert depth maps to point clouds:

```python
import numpy as np
import cv2
import pyroboframes as prf
from pyroboframes.depth_io import load_point_cloud_from_depth_map

# Read depth map (e.g., from RealSense)
depth_frame = pipeline.wait_for_frames().get_depth_frame()
depth_array = np.asanyarray(depth_frame.get_data())  # shape: [480, 640]

# Calibration parameters
fx, fy = 384.0, 384.0
cx, cy = 320.5, 240.5

# Convert to point cloud
cloud = load_point_cloud_from_depth_map(
    depth_array,
    fx=fx, fy=fy,
    cx=cx, cy=cy,
)

print(f"Generated {len(cloud)} points from {depth_array.shape[0]}×{depth_array.shape[1]} depth map")
```

## API Reference

### CameraIntrinsics

```python
class CameraIntrinsics:
    def __init__(
        self,
        fx: float,      # focal length x (pixels)
        fy: float,      # focal length y (pixels)
        cx: float,      # principal point x (pixels)
        cy: float,      # principal point y (pixels)
        width: int,     # image width (pixels)
        height: int,    # image height (pixels)
    ) -> None: ...

    def k_matrix() -> np.ndarray:  # [3, 3]
        """Get the 3×3 K intrinsic matrix."""

    def project(
        self,
        x: float,       # 3D x in camera frame
        y: float,       # 3D y in camera frame
        z: float,       # 3D z in camera frame
    ) -> Tuple[float, float] | None:
        """Project 3D point to image pixel. Returns None if behind camera."""

    def unproject_direction(
        self,
        u: float,       # pixel x
        v: float,       # pixel y
    ) -> np.ndarray:    # [3] unit direction
        """Unproject pixel to 3D ray direction (unit vector)."""

    # Properties
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
```

### CameraCalibration

```python
class CameraCalibration:
    def __init__(
        self,
        name: str,      # camera name (e.g., "observation.images.top")
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ) -> None: ...

    def project_world_point(
        self,
        x: float,       # 3D x in world frame
        y: float,       # 3D y in world frame
        z: float,       # 3D z in world frame
    ) -> Tuple[float, float] | None:
        """Project world 3D point to image (with pose transform)."""

    def unproject_to_world_ray(
        self,
        u: float,       # pixel x
        v: float,       # pixel y
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject pixel to world ray. Returns (origin, direction)."""

    # Properties
    name: str
    intrinsics: CameraIntrinsics
```

## depth_io Utilities

```python
from pyroboframes.depth_io import (
    load_point_cloud_from_numpy,
    load_point_cloud_from_depth_map,
    downsample_point_cloud,
    filter_point_cloud,
    align_point_clouds_icp,
)

# Load from NumPy array
points = np.random.randn(1000, 3).astype(np.float32)
cloud = load_point_cloud_from_numpy(points)

# Load from depth map with intrinsics
depth = np.random.rand(480, 640).astype(np.float32)
cloud = load_point_cloud_from_depth_map(depth, fx=500, fy=500, cx=320, cy=240)

# Downsample
cloud_sparse = downsample_point_cloud(cloud, factor=4)

# Filter by depth range
cloud_clean = filter_point_cloud(cloud, min_depth=0.1, max_depth=5.0)

# Align two clouds (ICP)
R, t = align_point_clouds_icp(source_cloud, target_cloud)
```

## Multi-Camera 3D Alignment

Store camera poses relative to a common world frame:

```python
import numpy as np

# Rotation: identity (no rotation)
R = np.eye(3)

# Translation: camera 2 is 1 meter to the right of camera 1
t = np.array([1.0, 0.0, 0.0])

# Later: transform 3D points from camera 2 frame to camera 1 frame
p_camera2 = np.array([1.0, 1.0, 1.0])
p_camera1 = R @ p_camera2 + t
```

## Troubleshooting

### "Point projects to infinity"

Camera focal length (fx, fy) is too small or the 3D point is too close to the camera.

**Fix:** Verify focal length from calibration source; ensure depth is in meters.

### "Principal point outside image"

cx > width or cy > height.

**Fix:** Principal point should be near the image center. For 640×480 image, expect cx ≈ 320, cy ≈ 240.

### "No points after projection"

All points are behind the camera (z < 0 in camera frame).

**Fix:** Verify camera pose (rotation/translation) is correct for transforming world points.

## Performance

- **Projection:** ~1 μs per point
- **Unprojection:** ~1 μs per pixel
- **Depth to point cloud:** ~10-100 ms for 640×480 depth map (NumPy conversion)

## Next Steps (v0.4.2+)

- [ ] Save/load calibration from JSON (dataset metadata)
- [ ] Camera pose estimation from checkerboard images
- [ ] Undistortion utilities (remove lens distortion from images)
- [ ] Multi-camera 3D reconstruction
- [ ] Epipolar geometry for stereo matching

## References

- [OpenCV Camera Calibration](https://docs.opencv.org/master/dc/dbb/tutorial_calibration.html)
- [RealSense Calibration](https://github.com/IntelRealSense/librealsense/wiki/Projection-in-RealSense-SDK-2.0)
- [Pinhole Camera Model](https://en.wikipedia.org/wiki/Pinhole_camera_model)
