"""Deep dataset validation: missing frames, codec errors, and temporal gap detection.

Goes beyond metadata-level checks to probe actual video files and timestamp sequences.

```python
from pyroboframes.validation import DatasetValidator

validator = DatasetValidator(ds, sample_rate=0.1)
report = validator.validate()
print(report.summary())
report.raise_if_errors()
```
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from ._core import RoboFrameDataset


@dataclass
class ValidationIssue:
    """A single validation finding."""

    severity: Literal["error", "warning", "info"]
    category: str  # "missing_frames" | "codec_error" | "temporal_gap" | "metadata"
    message: str
    episode: Optional[int] = None
    camera: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.upper()}]"]
        if self.episode is not None:
            parts.append(f"episode={self.episode}")
        if self.camera:
            parts.append(f"camera={self.camera!r}")
        parts.append(f"({self.category})")
        parts.append(self.message)
        return " ".join(parts)


@dataclass
class FullValidationReport:
    """Aggregated report from DatasetValidator.validate()."""

    issues: list[ValidationIssue] = field(default_factory=list)
    episodes_checked: int = 0
    cameras_checked: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        lines = [
            f"Validation complete in {self.duration_s:.2f}s — "
            f"{self.episodes_checked} episodes, {len(self.cameras_checked)} cameras",
            f"  {len(self.errors)} errors, {len(self.warnings)} warnings",
        ]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)

    def raise_if_errors(self) -> None:
        if self.errors:
            msgs = "\n".join(str(e) for e in self.errors)
            raise ValueError(f"Dataset validation failed with {len(self.errors)} error(s):\n{msgs}")


class TemporalGapChecker:
    """Detects timestamp gaps larger than 2× the frame period."""

    def __init__(self, fps: float):
        self.fps = fps
        self._max_gap = 2.0 / max(fps, 1e-6)

    def check(self, episode_index: int, timestamps: np.ndarray) -> list[ValidationIssue]:
        if len(timestamps) < 2:
            return []
        diffs = np.diff(timestamps)
        gap_mask = diffs > self._max_gap
        issues = []
        for i, (gap, is_gap) in enumerate(zip(diffs, gap_mask)):
            if is_gap:
                issues.append(ValidationIssue(
                    severity="warning",
                    category="temporal_gap",
                    episode=episode_index,
                    message=(
                        f"gap of {gap:.4f}s between frames {i} and {i+1} "
                        f"(threshold: {self._max_gap:.4f}s at {self.fps}fps)"
                    ),
                ))
        return issues


class MissingFrameChecker:
    """Compares expected frame count (from metadata) vs. decodable frames via ffprobe."""

    def __init__(self) -> None:
        self._has_ffprobe = shutil.which("ffprobe") is not None

    def check(
        self, episode_index: int, camera: str, video_path: str, expected_frames: int
    ) -> list[ValidationIssue]:
        if not self._has_ffprobe:
            return [ValidationIssue(
                severity="info",
                category="missing_frames",
                episode=episode_index,
                camera=camera,
                message="ffprobe not found; skipping frame count check",
            )]

        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-count_packets",
                    "-show_entries", "stream=nb_read_packets",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return [ValidationIssue(
                    severity="error",
                    category="missing_frames",
                    episode=episode_index,
                    camera=camera,
                    message=f"ffprobe failed for {video_path!r}: {result.stderr.strip()}",
                )]
            actual = int(result.stdout.strip())
            if actual != expected_frames:
                severity = "error" if actual < expected_frames else "warning"
                return [ValidationIssue(
                    severity=severity,
                    category="missing_frames",
                    episode=episode_index,
                    camera=camera,
                    message=(
                        f"expected {expected_frames} frames but video has {actual} "
                        f"({'missing' if actual < expected_frames else 'extra'} frames)"
                    ),
                )]
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as exc:
            return [ValidationIssue(
                severity="warning",
                category="missing_frames",
                episode=episode_index,
                camera=camera,
                message=f"frame count check failed: {exc}",
            )]
        return []


class CodecHealthChecker:
    """Probes N random frames per video file to detect decode errors."""

    def __init__(self, n_samples: int = 5) -> None:
        self.n_samples = n_samples
        self._has_ffmpeg = shutil.which("ffmpeg") is not None

    def check(self, episode_index: int, camera: str, video_path: str) -> list[ValidationIssue]:
        if not self._has_ffmpeg:
            return []

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-v", "error",
                    "-i", video_path,
                    "-vf", f"select='not(mod(n,{max(1, self.n_samples)}))'",
                    "-vsync", "vfr",
                    "-f", "null", "-",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            stderr = result.stderr.strip()
            issues = []
            if result.returncode != 0:
                issues.append(ValidationIssue(
                    severity="error",
                    category="codec_error",
                    episode=episode_index,
                    camera=camera,
                    message=f"codec probe failed for {video_path!r}: {stderr[:200]}",
                ))
            elif "error" in stderr.lower() or "invalid" in stderr.lower():
                issues.append(ValidationIssue(
                    severity="warning",
                    category="codec_error",
                    episode=episode_index,
                    camera=camera,
                    message=f"possible decode errors in {video_path!r}: {stderr[:200]}",
                ))
            return issues
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return [ValidationIssue(
                severity="warning",
                category="codec_error",
                episode=episode_index,
                camera=camera,
                message=f"codec check timed out or failed: {exc}",
            )]


class DatasetValidator:
    """Full dataset validation: metadata + temporal gaps + missing frames + codec health.

    Args:
        dataset: RoboFrameDataset to validate.
        check_frames: Probe video frame counts via ffprobe.
        check_temporal: Detect timestamp gaps in Parquet data.
        check_codec: Probe video decode health via ffmpeg.
        sample_rate: Fraction of episodes to probe for video checks (0.0–1.0).
    """

    def __init__(
        self,
        dataset: "RoboFrameDataset",
        *,
        check_frames: bool = True,
        check_temporal: bool = True,
        check_codec: bool = True,
        sample_rate: float = 0.1,
    ) -> None:
        self.dataset = dataset
        self.check_frames = check_frames
        self.check_temporal = check_temporal
        self.check_codec = check_codec
        self.sample_rate = max(0.0, min(1.0, sample_rate))

    def validate(self) -> FullValidationReport:
        """Run all enabled checks and return a full report."""
        import time

        t0 = time.monotonic()
        report = FullValidationReport()

        # Rust-side metadata validation
        rust_report = self.dataset.validate()
        for msg in rust_report.errors:
            report.issues.append(ValidationIssue(
                severity="error", category="metadata", message=msg
            ))
        for msg in rust_report.warnings:
            report.issues.append(ValidationIssue(
                severity="warning", category="metadata", message=msg
            ))

        num_episodes = self.dataset.num_episodes()
        fps = self.dataset.fps()
        cameras = self.dataset.cameras()
        report.cameras_checked = cameras

        # Select episodes to deep-check
        rng = np.random.default_rng(42)
        all_eps = list(range(num_episodes))
        n_sample = max(1, int(len(all_eps) * self.sample_rate))
        sampled_eps = sorted(rng.choice(all_eps, size=min(n_sample, len(all_eps)), replace=False))

        temporal_checker = TemporalGapChecker(fps) if self.check_temporal else None
        frame_checker = MissingFrameChecker() if self.check_frames else None
        codec_checker = CodecHealthChecker() if self.check_codec else None

        for ep_idx in sampled_eps:
            report.issues.extend(self.validate_episode(
                ep_idx,
                temporal_checker=temporal_checker,
                frame_checker=frame_checker,
                codec_checker=codec_checker,
            ))

        report.episodes_checked = len(sampled_eps)
        report.duration_s = time.monotonic() - t0
        return report

    def validate_episode(
        self,
        episode_index: int,
        *,
        temporal_checker: Optional[TemporalGapChecker] = None,
        frame_checker: Optional[MissingFrameChecker] = None,
        codec_checker: Optional[CodecHealthChecker] = None,
    ) -> list[ValidationIssue]:
        """Validate a single episode. Returns list of issues found."""
        import os

        issues: list[ValidationIssue] = []
        dataset_path = self.dataset.path()
        episodes = self.dataset.episodes()
        if episode_index >= len(episodes):
            return [ValidationIssue(
                severity="error",
                category="metadata",
                episode=episode_index,
                message=f"episode {episode_index} out of range (total {len(episodes)})",
            )]

        ep = episodes[episode_index]

        # Temporal gap check — read timestamps from Parquet
        if temporal_checker is not None:
            try:
                import pyarrow.parquet as pq
                data_file = os.path.join(dataset_path, ep.get("data_file", ""))
                if data_file and os.path.exists(data_file):
                    table = pq.read_table(data_file, columns=["timestamp"])
                    timestamps = table["timestamp"].to_pylist()
                    if timestamps:
                        issues.extend(temporal_checker.check(
                            episode_index, np.array(timestamps, dtype=float)
                        ))
            except Exception as exc:
                issues.append(ValidationIssue(
                    severity="info",
                    category="temporal_gap",
                    episode=episode_index,
                    message=f"could not read timestamps: {exc}",
                ))

        # Video checks
        for cam in self.dataset.cameras():
            video_rel = ep.get("videos", {}).get(cam, {}).get("path", "")
            if not video_rel:
                continue
            video_path = os.path.join(dataset_path, video_rel)
            if not os.path.exists(video_path):
                issues.append(ValidationIssue(
                    severity="error",
                    category="missing_frames",
                    episode=episode_index,
                    camera=cam,
                    message=f"video file missing: {video_path!r}",
                ))
                continue

            if frame_checker is not None:
                expected = ep.get("length", 0)
                issues.extend(frame_checker.check(episode_index, cam, video_path, expected))

            if codec_checker is not None:
                issues.extend(codec_checker.check(episode_index, cam, video_path))

        return issues
