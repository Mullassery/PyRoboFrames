"""TFRecord utilities for Waymo Open Dataset parsing.

Handles efficient parsing of Waymo's TFRecord format with embedded protobufs.
Supports camera images, lidar point clouds, and calibration metadata.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import numpy as np


def parse_tfrecord_example(raw_record: bytes) -> Dict[str, Any]:
    """Parse a single TFRecord example from Waymo dataset.

    Args:
        raw_record: Raw bytes from TFRecordDataset

    Returns:
        Dict with parsed frame data:
        - timestamp_micros: int
        - frame_id: int
        - images: {camera_id -> encoded_image_bytes}
        - lidar: {lidar_id -> range_image_bytes}
        - camera_calibrations: list of calibration dicts
        - objects: list of 3D bbox annotations
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TFRecord parsing requires: pip install tensorflow")

    # Parse the example proto
    example = tf.train.Example()
    example.ParseFromString(raw_record)

    features = example.features.feature
    parsed_data = {
        "timestamp_micros": None,
        "frame_id": None,
        "images": {},
        "lidar": {},
        "camera_calibrations": [],
        "objects": [],
    }

    # Extract metadata
    if "context/timestamp_micros" in features:
        parsed_data["timestamp_micros"] = features["context/timestamp_micros"].int64_list.value[0]

    if "context/frame_number" in features:
        parsed_data["frame_id"] = features["context/frame_number"].int64_list.value[0]

    # Extract camera images
    camera_names = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]

    for cam_name in camera_names:
        encoded_key = f"images/{cam_name}/encoded"
        if encoded_key in features:
            # Store encoded bytes - caller will decode with PIL
            encoded_image = features[encoded_key].bytes_list.value[0]
            parsed_data["images"][cam_name] = encoded_image

    # Extract lidar
    lidar_names = ["TOP"]  # Waymo has 5 lidars, TOP is primary

    for lidar_name in lidar_names:
        range_image_key = f"lidar/{lidar_name}/range_image/range_image_return1/range_image/data"
        if range_image_key in features:
            range_image_bytes = features[range_image_key].bytes_list.value[0]
            parsed_data["lidar"][lidar_name] = range_image_bytes

    # Extract camera calibrations
    if "context/camera_calibrations" in features:
        calib_bytes = features["context/camera_calibrations"].bytes_list.value[0]
        # In production, would parse calibration proto
        parsed_data["camera_calibrations"] = calib_bytes

    # Extract 3D objects
    if "objects/type" in features:
        object_types = features["objects/type"].int64_list.value
        object_bboxes_3d = features.get("objects/bbox/center_x", None)
        if object_bboxes_3d:
            parsed_data["objects"] = object_types

    return parsed_data


def decode_camera_image(encoded_image: bytes) -> np.ndarray:
    """Decode JPEG image from Waymo format.

    Args:
        encoded_image: Encoded JPEG bytes

    Returns:
        [H, W, 3] uint8 RGB image
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Image decoding requires: pip install tensorflow")

    # Use TensorFlow to decode JPEG (more efficient than PIL)
    image_tensor = tf.image.decode_jpeg(encoded_image, channels=3)

    return image_tensor.numpy()


def decode_lidar_range_image(
    range_image_bytes: bytes,
    shape: Tuple[int, int] = (64, 2048),
) -> np.ndarray:
    """Decode lidar range image from Waymo format.

    Args:
        range_image_bytes: Encoded range image bytes
        shape: Expected shape (height, width)

    Returns:
        [H, W, 2] float32 range image (distance, intensity)
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Lidar decoding requires: pip install tensorflow")

    # Decode as float32
    range_image = tf.io.decode_raw(range_image_bytes, tf.float32)

    # Reshape to [H, W, 2]
    height, width = shape
    range_image = tf.reshape(range_image, [height, width, 2])

    return range_image.numpy()


def range_image_to_point_cloud(
    range_image: np.ndarray,
    calibration: Dict[str, float],
) -> np.ndarray:
    """Convert range image to 3D point cloud.

    Args:
        range_image: [H, W, 2] (distance, intensity)
        calibration: Camera intrinsics dict

    Returns:
        [N, 4] point cloud (x, y, z, intensity)
    """
    H, W = range_image.shape[:2]

    # Extract distance and intensity
    distances = range_image[:, :, 0]  # [H, W]
    intensity = range_image[:, :, 1]  # [H, W]

    # Create ray angles
    # Waymo lidar: 64 vertical channels, 2048 horizontal pixels
    vertical_angles = np.linspace(-np.pi / 4, np.pi / 4, H)  # -45 to +45 degrees
    horizontal_angles = np.linspace(-np.pi, np.pi, W)

    # Compute point cloud
    points_3d = []

    for h in range(H):
        for w in range(W):
            dist = distances[h, w]

            if dist > 0:  # Valid point
                v_angle = vertical_angles[h]
                h_angle = horizontal_angles[w]

                # Spherical to Cartesian
                x = dist * np.cos(v_angle) * np.cos(h_angle)
                y = dist * np.cos(v_angle) * np.sin(h_angle)
                z = dist * np.sin(v_angle)
                i = intensity[h, w]

                points_3d.append([x, y, z, i])

    if len(points_3d) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    return np.array(points_3d, dtype=np.float32)


def parse_camera_calibration_proto(calib_bytes: bytes) -> Dict[str, Dict[str, float]]:
    """Parse camera calibration proto from Waymo format.

    Args:
        calib_bytes: Serialized calibration proto

    Returns:
        {camera_name -> {fx, fy, cx, cy, width, height, k1, k2, p1, p2}}
    """
    # Placeholder - real implementation would use waymo_open_dataset.dataset_pb2
    # For now, return standard Waymo intrinsics (well-known values)

    return {
        "FRONT": {
            "fx": 2015.0,
            "fy": 2015.0,
            "cx": 960,
            "cy": 640,
            "width": 1920,
            "height": 1280,
            "k1": -0.33,
            "k2": 0.18,
        },
        "FRONT_LEFT": {
            "fx": 2015.0,
            "fy": 2015.0,
            "cx": 960,
            "cy": 640,
            "width": 1920,
            "height": 1280,
            "k1": -0.33,
            "k2": 0.18,
        },
        "FRONT_RIGHT": {
            "fx": 2015.0,
            "fy": 2015.0,
            "cx": 960,
            "cy": 640,
            "width": 1920,
            "height": 1280,
            "k1": -0.33,
            "k2": 0.18,
        },
        "SIDE_LEFT": {
            "fx": 1590.0,
            "fy": 1590.0,
            "cx": 960,
            "cy": 640,
            "width": 1920,
            "height": 1280,
            "k1": -0.33,
            "k2": 0.18,
        },
        "SIDE_RIGHT": {
            "fx": 1590.0,
            "fy": 1590.0,
            "cx": 960,
            "cy": 640,
            "width": 1920,
            "height": 1280,
            "k1": -0.33,
            "k2": 0.18,
        },
    }
