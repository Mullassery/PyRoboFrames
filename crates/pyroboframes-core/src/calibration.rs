//! Camera calibration: intrinsics, extrinsics, and distortion parameters.
//!
//! Stores calibration data for cameras in a robot dataset, enabling:
//! - Undistortion of images
//! - 3D reconstruction from depth
//! - Multi-camera alignment
//! - Camera pose estimation
//!
//! Based on OpenCV calibration format for compatibility with existing tools.

use serde::{Deserialize, Serialize};

/// Camera intrinsic matrix (3×3 K matrix).
///
/// Maps 3D world points to 2D image coordinates:
/// ```text
/// [u]   [fx  0 cx] [X]
/// [v] = [ 0 fy cy] [Y]
/// [1]   [ 0  0  1] [Z]
/// ```
///
/// Where (X, Y, Z) are world coordinates and (u, v) are image pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    /// Focal length in x (pixels)
    pub fx: f64,
    /// Focal length in y (pixels)
    pub fy: f64,
    /// Principal point x coordinate (pixels)
    pub cx: f64,
    /// Principal point y coordinate (pixels)
    pub cy: f64,
    /// Image width (pixels)
    pub width: usize,
    /// Image height (pixels)
    pub height: usize,
}

impl CameraIntrinsics {
    /// Create intrinsics from focal length and principal point.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: usize, height: usize) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    /// Get the 3×3 K matrix as a flat array (row-major).
    pub fn k_matrix(&self) -> [f64; 9] {
        [
            self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0,
        ]
    }

    /// Project a 3D point onto the image plane (without distortion).
    ///
    /// Returns (u, v) pixel coordinates, or None if point is behind camera.
    pub fn project(&self, x: f64, y: f64, z: f64) -> Option<(f64, f64)> {
        if z <= 0.0 {
            return None; // Point behind camera
        }
        let u = self.fx * (x / z) + self.cx;
        let v = self.fy * (y / z) + self.cy;
        Some((u, v))
    }

    /// Unproject a pixel to a 3D ray (direction only, no depth).
    ///
    /// Returns the unit direction vector (normalized).
    pub fn unproject_direction(&self, u: f64, v: f64) -> [f64; 3] {
        let x = (u - self.cx) / self.fx;
        let y = (v - self.cy) / self.fy;
        let z = 1.0;
        let norm = (x * x + y * y + z * z).sqrt();
        [x / norm, y / norm, z / norm]
    }
}

/// Lens distortion parameters (OpenCV radial + tangential).
///
/// Supports radial distortion (k1, k2, k3) and tangential distortion (p1, p2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DistortionModel {
    /// Radial distortion coefficient 1
    pub k1: f64,
    /// Radial distortion coefficient 2
    pub k2: f64,
    /// Radial distortion coefficient 3 (usually 0 for standard cameras)
    pub k3: f64,
    /// Tangential distortion coefficient 1
    pub p1: f64,
    /// Tangential distortion coefficient 2
    pub p2: f64,
}

impl Default for DistortionModel {
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

impl DistortionModel {
    /// Create a distortion model from coefficients.
    pub fn new(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self { k1, k2, k3, p1, p2 }
    }

    /// Apply distortion to normalized image coordinates.
    ///
    /// Input: (x, y) normalized coordinates (in units of focal length, origin at principal point).
    /// Output: (x', y') distorted coordinates.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        // Radial distortion: 1 + k1*r^2 + k2*r^4 + k3*r^6
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        // Tangential distortion
        let x_tan = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let y_tan = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;

        let x_dist = x * radial + x_tan;
        let y_dist = y * radial + y_tan;

        (x_dist, y_dist)
    }

    /// Check if distortion is significant (any coefficient > 0.001).
    pub fn is_significant(&self) -> bool {
        self.k1.abs() > 0.001
            || self.k2.abs() > 0.001
            || self.k3.abs() > 0.001
            || self.p1.abs() > 0.001
            || self.p2.abs() > 0.001
    }
}

/// Camera pose: position and orientation relative to world frame.
///
/// Stored as 3×4 extrinsic matrix [R|t] where:
/// - R is the 3×3 rotation matrix (camera -> world)
/// - t is the 3×1 translation vector (camera -> world)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraPose {
    /// 3×3 rotation matrix (row-major), camera frame to world frame
    pub rotation: [f64; 9],
    /// 3×1 translation vector, camera frame to world frame
    pub translation: [f64; 3],
}

impl CameraPose {
    /// Identity pose: camera at origin, looking along +Z axis.
    pub fn identity() -> Self {
        Self {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// Create pose from rotation matrix and translation.
    pub fn new(rotation: [f64; 9], translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Get the extrinsic matrix as a 3×4 flat array.
    pub fn extrinsic_matrix(&self) -> [f64; 12] {
        [
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.translation[0],
            self.rotation[3],
            self.rotation[4],
            self.rotation[5],
            self.translation[1],
            self.rotation[6],
            self.rotation[7],
            self.rotation[8],
            self.translation[2],
        ]
    }

    /// Transform a 3D point from world to camera frame.
    pub fn world_to_camera(&self, p: [f64; 3]) -> [f64; 3] {
        // p_camera = R * p_world + t
        [
            self.rotation[0] * p[0]
                + self.rotation[1] * p[1]
                + self.rotation[2] * p[2]
                + self.translation[0],
            self.rotation[3] * p[0]
                + self.rotation[4] * p[1]
                + self.rotation[5] * p[2]
                + self.translation[1],
            self.rotation[6] * p[0]
                + self.rotation[7] * p[1]
                + self.rotation[8] * p[2]
                + self.translation[2],
        ]
    }

    /// Transform a 3D point from camera to world frame.
    pub fn camera_to_world(&self, p: [f64; 3]) -> [f64; 3] {
        // Inverse transform: p_world = R^T * (p_camera - t)
        let p_translated = [
            p[0] - self.translation[0],
            p[1] - self.translation[1],
            p[2] - self.translation[2],
        ];
        [
            self.rotation[0] * p_translated[0]
                + self.rotation[3] * p_translated[1]
                + self.rotation[6] * p_translated[2],
            self.rotation[1] * p_translated[0]
                + self.rotation[4] * p_translated[1]
                + self.rotation[7] * p_translated[2],
            self.rotation[2] * p_translated[0]
                + self.rotation[5] * p_translated[1]
                + self.rotation[8] * p_translated[2],
        ]
    }
}

/// Complete camera calibration: intrinsics, distortion, and pose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCalibration {
    /// Camera name (e.g., "observation.images.top")
    pub name: String,
    /// Intrinsic parameters
    pub intrinsics: CameraIntrinsics,
    /// Distortion model (optional)
    #[serde(default)]
    pub distortion: DistortionModel,
    /// Pose relative to world frame (optional, identity if not specified)
    #[serde(default = "CameraPose::identity")]
    pub pose: CameraPose,
}

impl CameraCalibration {
    /// Create a new calibration with intrinsics only (no distortion or pose).
    pub fn new(name: String, intrinsics: CameraIntrinsics) -> Self {
        Self {
            name,
            intrinsics,
            distortion: DistortionModel::default(),
            pose: CameraPose::identity(),
        }
    }

    /// Create a new calibration with all parameters.
    pub fn with_all(
        name: String,
        intrinsics: CameraIntrinsics,
        distortion: DistortionModel,
        pose: CameraPose,
    ) -> Self {
        Self {
            name,
            intrinsics,
            distortion,
            pose,
        }
    }

    /// Project a 3D world point to image pixel coordinates.
    ///
    /// Applies pose (world -> camera), then intrinsic projection, then distortion.
    pub fn project_world_point(&self, p: [f64; 3]) -> Option<(f64, f64)> {
        // Transform to camera frame
        let p_cam = self.pose.world_to_camera(p);

        // Project using intrinsics (without distortion yet)
        let (u_undist, v_undist) = self.intrinsics.project(p_cam[0], p_cam[1], p_cam[2])?;

        // Apply distortion if significant
        if self.distortion.is_significant() {
            let x_norm = (u_undist - self.intrinsics.cx) / self.intrinsics.fx;
            let y_norm = (v_undist - self.intrinsics.cy) / self.intrinsics.fy;
            let (x_dist, y_dist) = self.distortion.apply(x_norm, y_norm);
            let u = x_dist * self.intrinsics.fx + self.intrinsics.cx;
            let v = y_dist * self.intrinsics.fy + self.intrinsics.cy;
            Some((u, v))
        } else {
            Some((u_undist, v_undist))
        }
    }

    /// Unproject a pixel to a 3D ray in world frame.
    ///
    /// Returns the ray origin (camera position) and direction (unit vector).
    pub fn unproject_to_world_ray(&self, u: f64, v: f64) -> ([f64; 3], [f64; 3]) {
        // Get ray direction in camera frame (undistorted)
        let dir_cam = self.intrinsics.unproject_direction(u, v);

        // Transform ray origin and direction to world frame
        let origin = self.pose.camera_to_world([0.0, 0.0, 0.0]);

        // Rotate direction (R^T for camera to world rotation)
        let dir_world = [
            self.pose.rotation[0] * dir_cam[0]
                + self.pose.rotation[3] * dir_cam[1]
                + self.pose.rotation[6] * dir_cam[2],
            self.pose.rotation[1] * dir_cam[0]
                + self.pose.rotation[4] * dir_cam[1]
                + self.pose.rotation[7] * dir_cam[2],
            self.pose.rotation[2] * dir_cam[0]
                + self.pose.rotation[5] * dir_cam[1]
                + self.pose.rotation[8] * dir_cam[2],
        ];

        (origin, dir_world)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsics_k_matrix() {
        let intr = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let k = intr.k_matrix();
        assert_eq!(k[0], 500.0); // fx
        assert_eq!(k[4], 500.0); // fy
        assert_eq!(k[2], 320.0); // cx
        assert_eq!(k[5], 240.0); // cy
        assert_eq!(k[8], 1.0); // bottom-right
    }

    #[test]
    fn intrinsics_project() {
        let intr = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        // Point at (1, 1, 2) meters
        let (u, v) = intr.project(1.0, 1.0, 2.0).unwrap();
        // u = 500 * (1/2) + 320 = 570
        // v = 500 * (1/2) + 240 = 490
        assert!((u - 570.0).abs() < 0.01);
        assert!((v - 490.0).abs() < 0.01);
    }

    #[test]
    fn intrinsics_rejects_behind_camera() {
        let intr = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        assert!(intr.project(1.0, 1.0, -1.0).is_none()); // Behind camera
        assert!(intr.project(1.0, 1.0, 0.0).is_none()); // On camera plane
    }

    #[test]
    fn distortion_no_distortion() {
        let dist = DistortionModel::default();
        let (x, y) = dist.apply(0.1, 0.2);
        assert!((x - 0.1).abs() < 0.001);
        assert!((y - 0.2).abs() < 0.001);
    }

    #[test]
    fn distortion_is_significant() {
        let dist_none = DistortionModel::default();
        assert!(!dist_none.is_significant());

        let dist_some = DistortionModel::new(0.01, 0.0, 0.0, 0.0, 0.0);
        assert!(dist_some.is_significant());
    }

    #[test]
    fn pose_identity() {
        let pose = CameraPose::identity();
        let p = [1.0, 2.0, 3.0];
        let p_cam = pose.world_to_camera(p);
        assert!((p_cam[0] - 1.0).abs() < 0.001);
        assert!((p_cam[1] - 2.0).abs() < 0.001);
        assert!((p_cam[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn pose_roundtrip() {
        let pose = CameraPose::identity();
        let p = [1.0, 2.0, 3.0];
        let p_cam = pose.world_to_camera(p);
        let p_back = pose.camera_to_world(p_cam);
        assert!((p_back[0] - p[0]).abs() < 0.001);
        assert!((p_back[1] - p[1]).abs() < 0.001);
        assert!((p_back[2] - p[2]).abs() < 0.001);
    }
}
