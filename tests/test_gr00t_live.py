"""Opt-in validation against a real GR00T-compatible dataset."""

from __future__ import annotations

import os

import pytest
from pyroboframes import validate_gr00t_dataset

GR00T_DATASET = os.environ.get("GR00T_DATASET")


@pytest.mark.skipif(not GR00T_DATASET, reason="set GR00T_DATASET to a real dataset")
def test_live_gr00t_dataset():
    report = validate_gr00t_dataset(GR00T_DATASET)
    report.raise_if_errors()

    assert report.episodes_checked > 0
