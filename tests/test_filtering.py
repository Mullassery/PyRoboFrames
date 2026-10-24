"""Tests for episode filtering over the real episode-metadata index."""

import pytest

from pyroboframes.filtering import EpisodeFilter, EpisodeFilterBuilder


EPISODES = [
    {"episode_index": 0, "length": 10, "task": "pick", "success": True, "quality_score": 0.4},
    {"episode_index": 1, "length": 200, "task": "place", "success": False, "quality_score": 0.9},
    {"episode_index": 2, "length": 300, "task": "pick", "success": True, "quality_score": 0.8},
]


def test_range_and_equality_filters_chain_correctly():
    filt = EpisodeFilter(EPISODES).where(length_min=100).where(task="pick")
    assert filt.episode_indices == [2]
    assert filt.count() == 1


def test_set_membership_filter():
    filt = EpisodeFilter(EPISODES).where(task=["pick", "place"])
    assert filt.episode_indices == [0, 1, 2]


def test_unknown_field_is_rejected_instead_of_matching_everything():
    with pytest.raises(ValueError, match="unknown episode field 'difficulty'"):
        EpisodeFilter(EPISODES).where(difficulty="easy")


def test_reset_restores_all_episodes():
    filt = EpisodeFilter(EPISODES).where(success=True)
    assert filt.episode_indices == [0, 2]
    assert filt.reset().episode_indices == [0, 1, 2]


def test_builder_composes_conditions():
    indices = (
        EpisodeFilterBuilder(EPISODES)
        .success_only()
        .min_length(100)
        .min_quality(0.5)
        .build()
    )
    assert indices == [2]


def test_dataset_with_episodes_attribute_is_accepted():
    class FakeDataset:
        episodes = [
            {"episode_index": 7, "length": 5, "from_index": 0, "to_index": 5},
            {"episode_index": 8, "length": 50, "from_index": 5, "to_index": 55},
        ]

    filt = EpisodeFilter(FakeDataset()).where(length_min=10)
    assert filt.episode_indices == [8]


def test_dataframe_without_episode_index_gets_a_clear_error():
    class NotADataset:
        pass

    with pytest.raises(TypeError, match="episodes"):
        EpisodeFilter(NotADataset())
