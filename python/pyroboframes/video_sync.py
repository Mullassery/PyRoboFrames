"""Temporal alignment of multi-camera video sequences with variable frame rates.

Handles jitter, dropped frames, and out-of-order decode via windowed matching
against a reference camera timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class FrameTimestamp:
    """A single frame's timing metadata."""

    frame_index: int
    log_time: int  # nanoseconds
    camera_name: str
    is_valid: bool = True


@dataclass
class CameraTimeline:
    """Timeline of frames from a single camera."""

    camera_name: str
    timestamps: np.ndarray  # int64 array of nanosecond timestamps
    frame_indices: np.ndarray  # corresponding frame indices
    fps: float | None = None

    def __post_init__(self):
        """Validate and sort by time."""
        self.timestamps = np.asarray(self.timestamps, dtype=np.int64)
        self.frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        if len(self.timestamps) != len(self.frame_indices):
            raise ValueError("timestamps and frame_indices must have same length")

        # Sort by time
        sort_idx = np.argsort(self.timestamps)
        self.timestamps = self.timestamps[sort_idx]
        self.frame_indices = self.frame_indices[sort_idx]

        if self.fps is None and len(self.timestamps) > 1:
            # Estimate FPS from median frame interval
            diffs = np.diff(self.timestamps)
            median_dt = np.median(diffs[diffs > 0])
            if median_dt > 0:
                self.fps = 1e9 / median_dt

    @property
    def num_frames(self) -> int:
        """Number of frames."""
        return len(self.timestamps)

    @property
    def time_range(self) -> tuple[int, int]:
        """(min_time, max_time) in nanoseconds."""
        return (int(self.timestamps.min()), int(self.timestamps.max()))


class VideoSynchronizer:
    """Synchronize multiple camera timelines to a reference camera.

    Uses a sliding window approach to match frames from secondary cameras to the reference
    timeline, handling jitter and dropped frames gracefully.
    """

    def __init__(
        self,
        reference_camera: CameraTimeline,
        window_size_ns: int | None = None,
        max_reorder_frames: int = 10,
    ):
        """Initialize synchronizer with a reference camera.

        Args:
            reference_camera: CameraTimeline to sync other cameras to
            window_size_ns: Time window for frame matching (default: 2x median frame interval)
            max_reorder_frames: Max frames we'll reorder to fix out-of-order delivery
        """
        self.reference = reference_camera
        self.max_reorder_frames = max_reorder_frames

        # Auto-detect window size
        if window_size_ns is None:
            if len(reference_camera.timestamps) > 1:
                diffs = np.diff(reference_camera.timestamps)
                median_dt = np.median(diffs[diffs > 0])
                window_size_ns = int(median_dt * 2)
            else:
                window_size_ns = 100_000_000  # 100ms default

        self.window_size_ns = window_size_ns

    def sync_camera(self, other: CameraTimeline) -> dict[int, int]:
        """Sync a secondary camera to the reference timeline.

        Returns:
            Mapping of reference frame_index → secondary frame_index (-1 if no match found)
        """
        result = {}

        for ref_idx, ref_time in zip(self.reference.frame_indices, self.reference.timestamps):
            # Find frames within window
            window_start = ref_time - self.window_size_ns // 2
            window_end = ref_time + self.window_size_ns // 2

            candidates = np.where((other.timestamps >= window_start) & (other.timestamps <= window_end))[0]

            if len(candidates) == 0:
                # No match
                result[int(ref_idx)] = -1
            else:
                # Pick closest frame
                closest_idx = candidates[np.argmin(np.abs(other.timestamps[candidates] - ref_time))]
                result[int(ref_idx)] = int(other.frame_indices[closest_idx])

        return result

    def sync_multi_camera(
        self,
        cameras: dict[str, CameraTimeline],
    ) -> dict[str, dict[int, int]]:
        """Sync multiple cameras to the reference.

        Args:
            cameras: Dict mapping camera_name → CameraTimeline

        Returns:
            Dict mapping camera_name → sync result (reference_idx → camera_idx)
        """
        return {name: self.sync_camera(timeline) for name, timeline in cameras.items()}


class JitterFilter:
    """Smooth frame delivery jitter via exponential moving average.

    Detects and corrects timing jitter (high-frequency noise in frame intervals)
    while preserving intentional tempo changes (e.g., slow-motion).
    """

    def __init__(self, alpha: float = 0.7):
        """Initialize jitter filter.

        Args:
            alpha: EMA smoothing factor (0.5-0.9). Higher = more aggressive smoothing
        """
        self.alpha = alpha

    def filter_timestamps(
        self,
        timestamps: np.ndarray,
        variance_threshold_ns: int = 1_000_000,
    ) -> np.ndarray:
        """Smooth frame timestamps via EMA.

        Args:
            timestamps: Input timestamps (nanoseconds, assumed sorted)
            variance_threshold_ns: Only smooth if variance exceeds this threshold

        Returns:
            Smoothed timestamps (same shape as input)
        """
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if len(timestamps) < 2:
            return timestamps.astype(np.int64)

        # Compute frame intervals
        intervals = np.diff(timestamps)

        # Check if we need to filter
        variance = np.var(intervals)
        if variance < variance_threshold_ns**2:
            return timestamps.astype(np.int64)

        # EMA-smooth intervals
        smoothed_intervals = np.zeros_like(intervals, dtype=np.float64)
        smoothed_intervals[0] = intervals[0]
        for i in range(1, len(intervals)):
            smoothed_intervals[i] = (
                self.alpha * intervals[i] + (1.0 - self.alpha) * smoothed_intervals[i - 1]
            )

        # Reconstruct timestamps
        smoothed_times = [timestamps[0]]
        for interval in smoothed_intervals:
            smoothed_times.append(smoothed_times[-1] + interval)

        return np.array(smoothed_times, dtype=np.int64)


def align_frame_sequences(
    reference_times: np.ndarray,
    secondary_times: np.ndarray,
    reference_frames: np.ndarray,
    secondary_frames: np.ndarray,
    window_size_ns: int = 100_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple windowed alignment of two frame sequences.

    Args:
        reference_times: Reference camera timestamps (ns)
        secondary_times: Secondary camera timestamps (ns)
        reference_frames: Reference camera frame data
        secondary_frames: Secondary camera frame data
        window_size_ns: Time window for matching

    Returns:
        (aligned_times, aligned_ref_frames, aligned_sec_frames) tuples aligned to reference
    """
    ref_timeline = CameraTimeline("ref", reference_times, np.arange(len(reference_times)))
    sec_timeline = CameraTimeline("sec", secondary_times, np.arange(len(secondary_times)))

    sync = VideoSynchronizer(ref_timeline, window_size_ns=window_size_ns)
    mapping = sync.sync_camera(sec_timeline)

    # Filter valid matches
    valid_ref_indices = []
    valid_sec_indices = []
    for ref_idx, sec_idx in mapping.items():
        if sec_idx >= 0:  # Match found
            valid_ref_indices.append(ref_idx)
            valid_sec_indices.append(sec_idx)

    if not valid_ref_indices:
        return np.array([], dtype=np.int64), np.array([]), np.array([])

    valid_ref_indices = np.array(valid_ref_indices, dtype=np.int64)
    valid_sec_indices = np.array(valid_sec_indices, dtype=np.int64)

    return (
        reference_times[valid_ref_indices],
        reference_frames[valid_ref_indices],
        secondary_frames[valid_sec_indices],
    )
