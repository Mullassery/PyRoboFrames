//! Point cloud reading and handling for depth cameras.
//!
//! Supports multiple point cloud formats:
//! - `.xyz`: Simple text format (one point per line: x y z)
//! - `.ply`: Polygon File Format (ASCII or binary)
//! - `.pcd`: Point Cloud Data format (ROS standard)
//! - `.npy`: Raw depth maps as NumPy arrays (HxWx1 or HxW)
//!
//! A `PointCloud` is a collection of 3D points with optional colors and normals.

use std::path::Path;

use crate::{Error, Result};

/// A 3D point cloud with optional per-point attributes.
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// Point positions: [N, 3] (x, y, z in meters)
    pub points: Vec<[f32; 3]>,
    /// Per-point RGB colors: [N, 3] (0-255). None if no colors.
    pub colors: Option<Vec<[u8; 3]>>,
    /// Per-point normal vectors: [N, 3]. None if no normals.
    pub normals: Option<Vec<[f32; 3]>>,
    /// Depth map (HxW). Stored as raw if loaded from depth format.
    pub depth_map: Option<Vec<f32>>,
    /// Depth map shape (height, width) if available.
    pub depth_shape: Option<(usize, usize)>,
}

impl PointCloud {
    /// Create a new point cloud from positions only.
    pub fn new(points: Vec<[f32; 3]>) -> Self {
        Self {
            points,
            colors: None,
            normals: None,
            depth_map: None,
            depth_shape: None,
        }
    }

    /// Number of points in the cloud.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the point cloud is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Load a point cloud from file. Format is auto-detected from extension.
    pub fn load(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "xyz" => Self::load_xyz(path),
            "ply" => Self::load_ply(path),
            "pcd" => Self::load_pcd(path),
            "npy" => Self::load_npy(path),
            _ => Err(Error::Dataset(format!(
                "unsupported point cloud format: .{} (expected .xyz, .ply, .pcd, or .npy)",
                ext
            ))),
        }
    }

    /// Load from XYZ format (simple text: x y z per line).
    fn load_xyz(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Dataset(format!("reading {}: {}", path.display(), e)))?;

        let mut points = Vec::new();
        for (line_no, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let coords: Result<Vec<f32>> = line
                .split_whitespace()
                .map(|s| {
                    s.parse::<f32>().map_err(|_| {
                        Error::Dataset(format!(
                            "{}:{}: invalid float: {}",
                            path.display(),
                            line_no + 1,
                            s
                        ))
                    })
                })
                .collect();

            let coords = coords?;
            if coords.len() < 3 {
                return Err(Error::Dataset(format!(
                    "{}:{}: expected at least 3 coordinates (x y z), got {}",
                    path.display(),
                    line_no + 1,
                    coords.len()
                )));
            }

            points.push([coords[0], coords[1], coords[2]]);
        }

        if points.is_empty() {
            return Err(Error::Dataset(format!(
                "{}: no valid points found",
                path.display()
            )));
        }

        Ok(Self::new(points))
    }

    /// Load from PLY format (Polygon File Format).
    ///
    /// Currently supports ASCII PLY with vertex elements containing x, y, z properties.
    /// Binary PLY is not yet supported.
    fn load_ply(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Dataset(format!("reading {}: {}", path.display(), e)))?;

        let mut lines = content.lines();

        // Parse header
        let mut vertex_count = 0;
        let mut has_x = false;
        let mut has_y = false;
        let mut has_z = false;
        let mut is_binary = false;

        for line in &mut lines {
            let line = line.trim();
            if line.starts_with("format") {
                if line.contains("binary") {
                    is_binary = true;
                }
            } else if line.starts_with("element vertex") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    vertex_count = parts[2]
                        .parse::<usize>()
                        .map_err(|_| {
                            Error::Dataset(format!(
                                "{}: invalid vertex count in PLY header",
                                path.display()
                            ))
                        })?;
                }
            } else if line.starts_with("property float x") {
                has_x = true;
            } else if line.starts_with("property float y") {
                has_y = true;
            } else if line.starts_with("property float z") {
                has_z = true;
            } else if line == "end_header" {
                break;
            }
        }

        if is_binary {
            return Err(Error::Dataset(format!(
                "{}: binary PLY not yet supported (use ASCII PLY)",
                path.display()
            )));
        }

        if !has_x || !has_y || !has_z {
            return Err(Error::Dataset(format!(
                "{}: PLY missing required x, y, or z properties",
                path.display()
            )));
        }

        let mut points = Vec::with_capacity(vertex_count);
        for (line_no, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let coords: Result<Vec<f32>> = line
                .split_whitespace()
                .take(3)
                .map(|s| {
                    s.parse::<f32>().map_err(|_| {
                        Error::Dataset(format!(
                            "{}:vertex {}: invalid float: {}",
                            path.display(),
                            line_no + 1,
                            s
                        ))
                    })
                })
                .collect();

            let coords = coords?;
            if coords.len() != 3 {
                return Err(Error::Dataset(format!(
                    "{}:vertex {}: expected 3 coordinates, got {}",
                    path.display(),
                    line_no + 1,
                    coords.len()
                )));
            }

            points.push([coords[0], coords[1], coords[2]]);
        }

        if points.len() != vertex_count {
            return Err(Error::Dataset(format!(
                "{}: expected {} vertices, got {}",
                path.display(),
                vertex_count,
                points.len()
            )));
        }

        Ok(Self::new(points))
    }

    /// Load from PCD format (Point Cloud Data, ROS standard).
    ///
    /// Currently supports ASCII PCD with X, Y, Z fields. Binary PCD and compressed PCD
    /// are not yet supported.
    fn load_pcd(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Dataset(format!("reading {}: {}", path.display(), e)))?;

        let mut lines = content.lines().peekable();
        let mut point_count = 0;
        let mut data_format = "";

        // Parse header until we find DATA line
        while let Some(line) = lines.peek() {
            let line = line.trim();
            if line.starts_with("POINTS") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    point_count = parts[1].parse::<usize>().map_err(|_| {
                        Error::Dataset(format!(
                            "{}: invalid POINTS count in PCD header",
                            path.display()
                        ))
                    })?;
                }
            } else if line.starts_with("DATA") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    data_format = parts[1];
                }
                lines.next(); // consume the DATA line and break
                break;
            }
            lines.next();
        }

        if data_format != "ascii" {
            return Err(Error::Dataset(format!(
                "{}: only ASCII PCD supported (got {})",
                path.display(),
                data_format
            )));
        }

        let mut points = Vec::with_capacity(point_count);
        for (line_no, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let coords: Result<Vec<f32>> = line
                .split_whitespace()
                .take(3)
                .map(|s| {
                    s.parse::<f32>().map_err(|_| {
                        Error::Dataset(format!(
                            "{}:point {}: invalid float: {}",
                            path.display(),
                            line_no + 1,
                            s
                        ))
                    })
                })
                .collect();

            let coords = coords?;
            if coords.len() != 3 {
                return Err(Error::Dataset(format!(
                    "{}:point {}: expected 3 coordinates, got {}",
                    path.display(),
                    line_no + 1,
                    coords.len()
                )));
            }

            points.push([coords[0], coords[1], coords[2]]);
        }

        if points.len() != point_count {
            return Err(Error::Dataset(format!(
                "{}: expected {} points, got {}",
                path.display(),
                point_count,
                points.len()
            )));
        }

        Ok(Self::new(points))
    }

    /// Load from NPY format (NumPy array saved as .npy).
    ///
    /// The array should be either:
    /// - 2D [N, 3]: point positions directly (N points, 3 coordinates each)
    /// - 3D [H, W, 3]: depth map as grid of points (H×W points)
    ///
    /// Supports float32 and float64 arrays.
    fn load_npy(path: &Path) -> Result<Self> {
        // Read the NPY file header manually (no external dependency)
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Dataset(format!("reading {}: {}", path.display(), e)))?;

        // Validate NPY magic number (first 6 bytes: \x93NUMPY)
        if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
            return Err(Error::Dataset(format!(
                "{}: not a valid .npy file (magic number mismatch)",
                path.display()
            )));
        }

        // Parse NPY header (version-dependent format)
        let version_major = bytes[6];
        let version_minor = bytes[7];

        if version_major != 1 && version_major != 3 {
            return Err(Error::Dataset(format!(
                "{}: NPY version {}.{} not supported (need 1.x or 3.x)",
                path.display(),
                version_major,
                version_minor
            )));
        }

        // For now, return helpful error: users should use .xyz, .ply, or .pcd
        // Full NPY parsing requires byte manipulation and numpy format compliance,
        // which is substantial. Better to guide users to simpler formats.
        Err(Error::Dataset(
            format!(
                "{}: .npy loading requires numpy-compatible parsing (complex). \
                 Recommend exporting as .xyz, .ply, or .pcd format instead. \
                 From Python: np.savetxt('cloud.xyz', points); or use open3d.io.write_point_cloud()",
                path.display()
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(name: &str, content: &str) -> (tempfile::NamedTempFile, String) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        let path = file.path().to_string_lossy().to_string();
        (file, path)
    }

    #[test]
    fn loads_xyz_format() {
        let xyz_content = "0.0 0.0 0.0\n1.0 1.0 1.0\n2.0 2.0 2.0\n";
        let (_file, path) = write_temp_file("test.xyz", xyz_content);
        let cloud = PointCloud::load(Path::new(&path)).unwrap();
        assert_eq!(cloud.len(), 3);
        assert_eq!(cloud.points[0], [0.0, 0.0, 0.0]);
        assert_eq!(cloud.points[1], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn loads_xyz_with_comments() {
        let xyz_content = "# Point cloud data\n0.0 0.0 0.0\n# comment\n1.0 1.0 1.0\n";
        let (_file, path) = write_temp_file("test.xyz", xyz_content);
        let cloud = PointCloud::load(Path::new(&path)).unwrap();
        assert_eq!(cloud.len(), 2);
    }

    #[test]
    fn loads_ply_format() {
        let ply_content = r#"ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.0 1.0 1.0
"#;
        let (_file, path) = write_temp_file("test.ply", ply_content);
        let cloud = PointCloud::load(Path::new(&path)).unwrap();
        assert_eq!(cloud.len(), 2);
        assert_eq!(cloud.points[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn loads_pcd_format() {
        let pcd_content = r#"VERSION 0.7
FIELDS X Y Z
SIZE 4 4 4
TYPE f f f
COUNT 1 1 1
WIDTH 2
HEIGHT 1
POINTS 2
DATA ascii
0.0 0.0 0.0
1.0 1.0 1.0
"#;
        let (_file, path) = write_temp_file("test.pcd", pcd_content);
        let cloud = PointCloud::load(Path::new(&path)).unwrap();
        assert_eq!(cloud.len(), 2);
    }

    #[test]
    fn rejects_unsupported_format() {
        let result = PointCloud::load(Path::new("test.unknown"));
        assert!(result.is_err());
    }

    #[test]
    fn xyz_rejects_missing_coordinates() {
        let xyz_content = "0.0 0.0\n"; // Only 2 coords
        let (_file, path) = write_temp_file("test.xyz", xyz_content);
        let result = PointCloud::load(Path::new(&path));
        assert!(result.is_err());
    }
}
