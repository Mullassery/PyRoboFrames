"""Tests for cross-dataset quality scoring (DatasetQualityProfile, CrossDatasetComparator)."""

import numpy as np
import pytest
from pyroboframes.quality import (
    CrossDatasetComparator,
    DatasetQualityProfile,
    EpisodeScorer,
)


def _make_scores(n: int, mean: float = 0.5, std: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        i: {
            "diversity": float(rng.normal(mean, std)),
            "sharpness": float(rng.normal(mean, std)),
            "state_variance": float(rng.normal(mean, std)),
            "action_magnitude": float(rng.normal(mean, std)),
            "motion_smoothness": float(rng.normal(mean, std)),
            "quality_score": float(rng.normal(mean, std)),
        }
        for i in range(n)
    }


def test_profile_from_scores_has_correct_episode_count():
    scores = _make_scores(20)
    profile = DatasetQualityProfile.from_scores("test", scores)
    assert profile.episode_count == 20


def test_profile_from_scores_percentiles_ordered():
    scores = _make_scores(100, mean=0.5, std=0.15)
    profile = DatasetQualityProfile.from_scores("test", scores)
    for metric, stats in profile.per_metric_stats.items():
        assert stats["p25"] <= stats["p50"] <= stats["p75"] <= stats["p90"], (
            f"Percentiles not ordered for {metric}: {stats}"
        )


def test_profile_empty_scores():
    profile = DatasetQualityProfile.from_scores("empty", {})
    assert profile.episode_count == 0
    assert profile.per_metric_stats == {}


def test_profile_summary_contains_name():
    scores = _make_scores(5)
    profile = DatasetQualityProfile.from_scores("my_dataset", scores)
    summary = profile.summary()
    assert "my_dataset" in summary


def test_comparator_rank_episode_returns_0_to_100():
    scores = _make_scores(100)
    profile = DatasetQualityProfile.from_scores("ref", scores)
    comparator = CrossDatasetComparator(reference=profile)

    episode_scores = {
        "diversity": 0.5,
        "sharpness": 0.8,
        "state_variance": 0.3,
        "quality_score": 0.5,  # should be skipped
    }
    ranks = comparator.rank_episode(episode_scores)
    assert "quality_score" not in ranks
    for metric, rank in ranks.items():
        assert 0.0 <= rank <= 100.0, f"Rank out of range for {metric}: {rank}"


def test_comparator_rank_high_value_above_50():
    """A clearly above-average episode should rank above 50th percentile."""
    rng = np.random.default_rng(1)
    scores = {i: {"diversity": float(rng.normal(0.3, 0.05))} for i in range(100)}
    profile = DatasetQualityProfile.from_scores("ref", scores)
    comparator = CrossDatasetComparator(reference=profile)

    # Value 3 std above mean → should be near 100th percentile.
    ranks = comparator.rank_episode({"diversity": 0.45})
    assert ranks["diversity"] > 80.0


def test_comparator_compare_cohens_d_sign():
    """Better dataset should have positive Cohen's d."""
    low_scores = _make_scores(50, mean=0.3, std=0.05, seed=0)
    high_scores = _make_scores(50, mean=0.7, std=0.05, seed=1)

    ref = DatasetQualityProfile.from_scores("low", low_scores)
    other = DatasetQualityProfile.from_scores("high", high_scores)
    comparator = CrossDatasetComparator(reference=ref)
    result = comparator.compare(other)

    for metric, stats in result.items():
        if metric == "quality_score":
            continue
        assert stats["cohens_d"] > 0, f"Expected positive Cohen's d for {metric}"
        assert 0.0 <= stats["pct_overlap"] <= 1.0


def test_comparator_recommend_mixing_ratio_favors_better():
    low_scores = _make_scores(50, mean=0.3, std=0.05, seed=0)
    high_scores = _make_scores(50, mean=0.7, std=0.05, seed=1)
    ref = DatasetQualityProfile.from_scores("low", low_scores)
    other = DatasetQualityProfile.from_scores("high", high_scores)
    comparator = CrossDatasetComparator(reference=ref)
    ratio = comparator.recommend_mixing_ratio(other)
    # Higher-quality `other` should get > 50% weight.
    assert ratio > 0.5


def test_comparator_equal_datasets_ratio_is_half():
    scores = _make_scores(50, mean=0.5, std=0.1, seed=42)
    profile = DatasetQualityProfile.from_scores("ds", scores)
    comparator = CrossDatasetComparator(reference=profile)
    # Same profile as reference → should be ~0.5
    ratio = comparator.recommend_mixing_ratio(profile)
    assert abs(ratio - 0.5) < 0.01
