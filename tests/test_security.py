"""Tests for pyroboframes.security path validation and its wiring into loader entry points."""

import os

import pytest

from pyroboframes.security import validate_dataset_path
from test_loader import make_dataset

import pyroboframes as prf


def test_validate_dataset_path_rejects_traversal_component():
    with pytest.raises(ValueError, match="Directory traversal"):
        validate_dataset_path("some/../escape")


def test_validate_dataset_path_rejects_leading_traversal():
    with pytest.raises(ValueError, match="Directory traversal"):
        validate_dataset_path("../escape")


def test_validate_dataset_path_allows_dotdot_free_relative_path(tmp_path):
    # No ".." component and no base_dir: just resolves to an absolute path.
    resolved = validate_dataset_path(tmp_path / "sub" / "file.h5")
    assert resolved.is_absolute()


def test_validate_dataset_path_allows_contained_path(tmp_path):
    inside = tmp_path / "data" / "file.h5"
    inside.parent.mkdir()
    resolved = validate_dataset_path(inside, base_dir=tmp_path)
    assert resolved == inside.resolve()


def test_validate_dataset_path_rejects_escape_from_base_dir(tmp_path):
    base = tmp_path / "sandbox"
    base.mkdir()
    outside = tmp_path / "outside" / "file.h5"
    outside.parent.mkdir()
    with pytest.raises(ValueError, match="must be within"):
        validate_dataset_path(outside, base_dir=base)


def test_robo_frame_dataset_from_path_base_dir_allows_contained(tmp_path):
    root = tmp_path / "datasets" / "ds"
    os.makedirs(root)
    make_dataset(str(root), episodes=1, length=4)

    ds = prf.RoboFrameDataset.from_path(str(root), base_dir=str(tmp_path))
    assert ds.num_episodes == 1


def test_robo_frame_dataset_from_path_base_dir_rejects_escape(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside" / "ds"
    os.makedirs(outside)
    make_dataset(str(outside), episodes=1, length=4)

    with pytest.raises(ValueError, match="escapes base_dir"):
        prf.RoboFrameDataset.from_path(str(outside), base_dir=str(sandbox))


def test_robo_frame_dataset_from_path_without_base_dir_is_unrestricted(tmp_path):
    # No base_dir: preserves pre-existing unrestricted behavior.
    root = tmp_path / "ds"
    os.makedirs(root)
    make_dataset(str(root), episodes=1, length=4)

    ds = prf.RoboFrameDataset.from_path(str(root))
    assert ds.num_episodes == 1


def test_convert_mcap_base_dir_rejects_escape(tmp_path):
    mcap_writer = pytest.importorskip("mcap.writer")

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    mcap_path = tmp_path / "run.mcap"
    with open(mcap_path, "wb") as fh:
        w = mcap_writer.Writer(fh)
        w.start()
        w.finish()

    with pytest.raises(ValueError, match="escapes base_dir"):
        prf.convert_mcap(str(mcap_path), str(tmp_path / "out"), base_dir=str(sandbox))
