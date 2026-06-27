# Depth Camera Support in PyRoboFrames v0.4.0

PyRoboFrames now supports reading and storing depth camera data, including point clouds from depth sensors. This guide explains how to load and integrate depth data with RGB video in your robot datasets.

## Point Cloud Formats

PyRoboFrames supports multiple point cloud formats for flexibility with different depth sensors and tools:

| Format | Extension | Pros | Cons |
|--------|-----------|------|------|
| **XYZ** | `.xyz` | Simple, human-readable, easy to generate | No metadata, larger files |
| **PLY** | `.ply` | Flexible, supports colors and normals | Larger (ASCII), verbose |
| **PCD** | `.pcd` | ROS standard, widely used in robotics | Verbose header |
| **NPY** | `.npy` | Compact (NumPy binary), natural for Python | Requires NumPy, less portable (v0.5.0+) |

## Quick Start

### Loading a Point Cloud

```python
import pyroboframes as prf
import numpy as np

# Load point cloud from file
cloud = prf.PointCloud.load("depth_frame.xyz")

# Get the point positions as a NumPy array
points = cloud.points()  # shape: [N, 3], dtype: float32 (x, y, z in meters)

print(f"Loaded {len(cloud)} points")
```

### Supported Formats

#### XYZ Format (Simple Text)

**File format:** One point per line, whitespace-separated coordinates.

```
0.0 0.0 0.0
1.5 2.3 0.5
# Comments are allowed
2.1 1.2 3.4
```

**Loading:**
```python
cloud = prf.PointCloud.load("scan.xyz")
```

#### PLY Format (Polygon File Format)

**File format:** Text header + ASCII vertex data.

```
ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.5 2.3 0.5
2.1 1.2 3.4
```

**Loading:**
```python
cloud = prf.PointCloud.load("scan.ply")
```

**Note:** Binary PLY is not yet supported; use ASCII PLY format.

#### PCD Format (Point Cloud Data - ROS Standard)

**File format:** Header + ASCII point data.

```
VERSION 0.7
FIELDS X Y Z
SIZE 4 4 4
TYPE f f f
COUNT 1 1 1
WIDTH 3
HEIGHT 1
POINTS 3
DATA ascii
0.0 0.0 0.0
1.5 2.3 0.5
2.1 1.2 3.4
```

**Loading:**
```python
cloud = prf.PointCloud.load("scan.pcd")
```

**Note:** Binary and compressed PCD formats are not yet supported; use ASCII PCD.

## Integrating Depth with RGB Video

Depth cameras and RGB cameras often run on different time schedules. PyRoboFrames uses the same `delta_timestamps` mechanism as video to align depth with RGB:

```python
import pyroboframes as prf

ds = prf.RoboFrameDataset.from_path("my_dataset/")

# Load RGB frames at 30 Hz and depth at 15 Hz (0.033s apart)
loader = ds.loader(
    batch_size=32,
    cameras=["observation.images.rgb"],
    # Align depth to RGB with 50ms lag (typical for USB depth sensors)
    delta_timestamps={
        "observation.depth": [-0.05],  # 50ms in the past
        "observation.images.rgb": [0.0],
    },
    tolerance_s=0.01,  # 10ms tolerance for timestamp matching
)

for batch in loader:
    rgb_frames = batch["observation.images.rgb"]  # [32, H, W, 3] uint8
    depth_frames = batch["observation.depth"]  # [32, H, W] float32 (meters)
    # Use rgb_frames and depth_frames together in training
```

## API Reference

### PointCloud Class

```python
class PointCloud:
    """A 3D point cloud from a depth camera."""
    
    @staticmethod
    def load(path: str) -> PointCloud:
        """Load a point cloud from file (.xyz, .ply, .pcd, .npy).
        
        Args:
            path: Path to point cloud file
            
        Returns:
            Loaded PointCloud object
            
        Raises:
            ValueError: If file format is unsupported or file is malformed
        """
    
    def len() -> int:
        """Return the number of points in the cloud."""
    
    def is_empty() -> bool:
        """Check if the point cloud is empty."""
    
    def points() -> np.ndarray:
        """Get point positions as NumPy array [N, 3] (x, y, z in meters)."""
    
    def __len__() -> int:
        """Support len(cloud)."""
    
    def __repr__() -> str:
        """String representation: PointCloud(points=N, colors=bool, normals=bool)."""
```

## Depth Sensor Integration Examples

### Oak-D (USB Depth Camera)

Oak-D sensors typically output depth at 30 Hz with 50-100ms latency. Store depth as PCD or XYZ files:

```bash
# Example: Export Oak-D depth as point cloud
oak_d_device.get_stereo_depth_frame() -> convert to .pcd and save
```

### RealSense (D435/D455)

RealSense cameras output depth + RGB at up to 30 Hz. Export to PCD or XYZ:

```bash
# librealsense provides .pcd export
realsense-viewer -> right-click -> Export Point Cloud
```

### Generic Depth Sensors

For any depth camera that outputs:
1. **Depth map (H×W)** → Convert to XYZ or PCD format:
   ```python
   # Simple depth-to-XYZ conversion
   depth = load_depth_map()  # [H, W] meters
   x, y = np.meshgrid(np.arange(W), np.arange(H))
   # Using camera intrinsics K:
   points_xyz = np.stack([
       (x - K[0,2]) * depth / K[0,0],
       (y - K[1,2]) * depth / K[1,1],
       depth
   ], axis=-1).reshape(-1, 3)
   # Save to XYZ or PCD
   ```

2. **Point cloud directly** → Use `.pcd` or `.ply` export from the sensor SDK

## Depth Camera Calibration

Camera calibration (intrinsics, distortion) is critical for:
- 3D reconstruction from depth
- Reprojecting between camera frames
- Multi-camera alignment

**Coming in v0.4.1:** Camera calibration APIs. For now:
- Store calibration in dataset metadata (custom JSON)
- Use external calibration tools (OpenCV, ROS calibration)
- Keep intrinsics [K matrix] with your depth processing code

## Storage Considerations

**Point clouds in datasets:**
- XYZ/PLY/PCD files can be stored in a `depth/` subdirectory per episode
- Typical size: 100KB–1MB per frame (depending on resolution)
- For 10,000 frames at 30 fps: ~100 MB–1 GB uncompressed

**Compression options (future):**
- PNG 16-bit (lossless, ~50KB per frame for 480p)
- WebP lossless (20–30KB per frame)
- Custom quantization (trade quality for size)

## Troubleshooting

### "Point cloud file not found"
Ensure the file exists and is readable:
```python
import os
assert os.path.exists("depth.xyz"), "File not found"
```

### "Invalid point cloud format"
Check the file format is one of: `.xyz`, `.ply`, `.pcd`, `.npy`
```python
# Check file extension
path = "depth.xyz"  # supported
# path = "depth.bin"  # not supported
```

### "Points have inconsistent dimensions"
Ensure the point cloud file has valid numeric data:
```
# ✓ Valid XYZ (3 columns)
0.0 0.0 0.0
1.0 1.0 1.0

# ✗ Invalid (2 columns)
0.0 0.0
```

### "Expected N points, got M"
The PCD header declares N points but the data section has M:
- Count: `POINTS N` in the PCD header must match the number of data lines
- Check the file isn't truncated or corrupted

## Performance

Typical performance on Apple Silicon:
- **Load 480p point cloud:** <1 ms (from .xyz or .pcd)
- **Load batch of 32 clouds:** ~30 ms
- **Memory per cloud:** ~5 MB (50k points × 3 floats)

## Next Steps

### v0.4.1 (Planned)
- Camera calibration registry (`Camera` struct with K, D, extrinsics)
- Depth reprojection APIs
- Multi-camera 3D alignment
- NPY format support

### v0.5.0 (Planned)
- Depth compression (PNG 16-bit, WebP)
- Depth inpainting (fill holes)
- Stereo matching (generate depth from RGB)
- 3D mesh generation from point clouds

## References

- [PLY (Polygon File Format)](http://paulbourke.net/dataformats/ply/)
- [PCD (Point Cloud Data)](https://pointclouds.org/documentation/tutorials/pcd_file_format.html)
- [PCL (Point Cloud Library)](http://pointclouds.org/)
- [Open3D (3D Data)](http://www.open3d.org/)
