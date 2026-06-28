"""Tests for the data validation toolkit (DatasetValidator, TemporalGapChecker, etc.)."""

import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pyroboframes.validation import (
    CodecHealthChecker,
    DatasetValidator,
    FullValidationReport,
    MissingFrameChecker,
    TemporalGapChecker,
    ValidationIssue,
)


# ---------------------------------------------------------------------------
# TemporalGapChecker
# ---------------------------------------------------------------------------


def test_temporal_gap_checker_no_gaps():
    fps = 30.0
    checker = TemporalGapChecker(fps)
    timestamps = np.arange(100) / fps
    issues = checker.check(episode_index=0, timestamps=timestamps)
    assert issues == []


def test_temporal_gap_checker_detects_gap():
    fps = 10.0
    checker = TemporalGapChecker(fps)
    # 0.5s gap between frame 4 and frame 5 → exceeds 2/fps = 0.2s threshold
    timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.9, 1.0, 1.1])
    issues = checker.check(episode_index=2, timestamps=timestamps)
    assert len(issues) == 1
    assert issues[0].category == "temporal_gap"
    assert issues[0].episode == 2
    assert "0.5" in issues[0].message or "gap" in issues[0].message.lower()


def test_temporal_gap_checker_single_frame_no_issues():
    checker = TemporalGapChecker(fps=30.0)
    issues = checker.check(episode_index=0, timestamps=np.array([0.0]))
    assert issues == []


def test_temporal_gap_checker_severity_is_warning():
    checker = TemporalGapChecker(fps=10.0)
    timestamps = np.array([0.0, 1.0])
    issues = checker.check(episode_index=0, timestamps=timestamps)
    assert len(issues) == 1
    assert issues[0].severity == "warning"


# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------


def test_validation_issue_str_includes_severity_and_category():
    issue = ValidationIssue(
        severity="error",
        category="missing_frames",
        message="video file missing",
        episode=3,
        camera="top",
    )
    s = str(issue)
    assert "ERROR" in s
    assert "missing_frames" in s
    assert "episode=3" in s
    assert "top" in s


def test_validation_issue_optional_fields_omitted():
    issue = ValidationIssue(severity="info", category="metadata", message="ok")
    s = str(issue)
    assert "episode" not in s
    assert "camera" not in s


# ---------------------------------------------------------------------------
# FullValidationReport
# ---------------------------------------------------------------------------


def test_full_report_ok_when_no_errors():
    report = FullValidationReport()
    report.issues.append(
        ValidationIssue(severity="warning", category="metadata", message="minor")
    )
    assert report.ok is True


def test_full_report_not_ok_with_errors():
    report = FullValidationReport()
    report.issues.append(
        ValidationIssue(severity="error", category="metadata", message="bad")
    )
    assert report.ok is False


def test_full_report_raise_if_errors():
    report = FullValidationReport()
    report.issues.append(
        ValidationIssue(severity="error", category="codec_error", message="decode failed")
    )
    with pytest.raises(ValueError, match="validation failed"):
        report.raise_if_errors()


def test_full_report_raise_if_no_errors_passes():
    report = FullValidationReport()
    report.raise_if_errors()  # Should not raise.


def test_full_report_summary_contains_counts():
    report = FullValidationReport(episodes_checked=5, cameras_checked=["top"])
    report.issues.append(ValidationIssue(severity="error", category="metadata", message="e"))
    report.issues.append(ValidationIssue(severity="warning", category="metadata", message="w"))
    summary = report.summary()
    assert "1 error" in summary
    assert "1 warning" in summary


# ---------------------------------------------------------------------------
# MissingFrameChecker (no ffprobe required for these tests)
# ---------------------------------------------------------------------------


def test_missing_frame_checker_no_ffprobe_returns_info(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    checker = MissingFrameChecker()
    issues = checker.check(0, "top", "/nonexistent/video.mp4", expected_frames=30)
    assert len(issues) == 1
    assert issues[0].severity == "info"
    assert "ffprobe" in issues[0].message.lower()


def test_missing_frame_checker_missing_file_returns_no_issue_without_ffprobe(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    checker = MissingFrameChecker()
    # Without ffprobe the checker returns info (not an error about missing file)
    issues = checker.check(0, "cam", "/no/such/file.mp4", 10)
    assert all(i.severity in ("info", "warning") for i in issues)


# ---------------------------------------------------------------------------
# DatasetValidator (metadata path — no ffmpeg/ffprobe needed)
# ---------------------------------------------------------------------------


def _make_minimal_dataset(path: str, n_episodes: int = 2, length: int = 10) -> None:
    """Write a minimal LeRobot v3.0 layout for validation testing."""
    import pyroboframes as prf

    arrays = {
        "observation.state": np.random.rand(n_episodes * length, 4).astype(np.float32),
        "action": np.random.rand(n_episodes * length, 4).astype(np.float32),
    }
    prf.write_lerobot_dataset(path, arrays, [length] * n_episodes)


def test_dataset_validator_clean_dataset_ok(tmp_path):
    import pyroboframes as prf

    ds_path = str(tmp_path / "ds")
    _make_minimal_dataset(ds_path)
    ds = prf.RoboFrameDataset.from_path(ds_path)
    validator = DatasetValidator(
        ds, check_frames=False, check_codec=False, check_temporal=False
    )
    report = validator.validate()
    assert report.ok, report.summary()


def test_dataset_validator_report_has_episodes_checked(tmp_path):
    import pyroboframes as prf

    ds_path = str(tmp_path / "ds")
    _make_minimal_dataset(ds_path, n_episodes=4, length=5)
    ds = prf.RoboFrameDataset.from_path(ds_path)
    validator = DatasetValidator(
        ds, check_frames=False, check_codec=False, check_temporal=False, sample_rate=1.0
    )
    report = validator.validate()
    assert report.episodes_checked >= 1
    assert report.duration_s >= 0.0


def test_dataset_validator_single_episode(tmp_path):
    import pyroboframes as prf

    ds_path = str(tmp_path / "ds")
    _make_minimal_dataset(ds_path, n_episodes=1, length=8)
    ds = prf.RoboFrameDataset.from_path(ds_path)
    validator = DatasetValidator(
        ds, check_frames=False, check_codec=False, check_temporal=False
    )
    report = validator.validate()
    assert isinstance(report, FullValidationReport)
