"""Episode filtering and querying API.

Filters an episode index (``RoboFrameDataset.episodes`` or an explicit list of
episode-metadata mappings) for curriculum learning, task-conditional training,
and data curation.

```python
from pyroboframes import EpisodeFilter, RoboFrameDataset

ds = RoboFrameDataset.from_path("…/my_lerobot_dataset")
long_episodes = EpisodeFilter(ds).where(length_min=100, length_max=500)
loader = ds.loader(episodes=long_episodes.episode_indices)
```

The real ``RoboFrameDataset`` episode index currently exposes
``episode_index``, ``length``, ``from_index``, and ``to_index``. Additional
keys are filtered only when they are actually present in the supplied episode
metadata; asking for a field that does not exist raises ``ValueError`` instead
of silently matching everything.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dataframe import RoboticsDataFrame  # noqa: F401  (documented rejection)

EpisodeMetadata = Mapping[str, Any]


def _normalize_episodes(source: Any) -> list[dict[str, Any]]:
    """Accept a dataset (anything with an ``episodes`` attribute) or an explicit
    sequence of episode-metadata mappings."""
    if isinstance(source, Mapping) or isinstance(source, (str, bytes)):
        raise TypeError(
            "EpisodeFilter expects a dataset with an `episodes` index or a "
            "sequence of episode-metadata mappings"
        )
    if isinstance(source, Sequence):
        return [dict(episode) for episode in source]
    episodes = getattr(source, "episodes", None)
    if episodes is None:
        raise TypeError(
            "EpisodeFilter requires an object with an `episodes` episode index "
            "(e.g. RoboFrameDataset) or an explicit sequence of episode metadata; "
            f"got {type(source).__name__}"
        )
    if callable(episodes):
        episodes = episodes()
    return [dict(episode) for episode in episodes]


class EpisodeFilter:
    """Build and apply filters on an episode metadata index."""

    def __init__(self, source: Any):
        self._episodes = _normalize_episodes(source)
        self._episode_indices = [
            int(episode.get("episode_index", position))
            for position, episode in enumerate(self._episodes)
        ]
        self._by_index = {
            episode_index: episode
            for episode_index, episode in zip(self._episode_indices, self._episodes)
        }
        self._conditions: list[tuple[str, str, Any]] = []

    @property
    def fields(self) -> set[str]:
        """Every metadata key present across the episode index."""
        return {key for episode in self._episodes for key in episode}

    def where(self, **kwargs) -> EpisodeFilter:
        """Apply filter conditions (SQL WHERE-clause semantics).

        Supports equality, ``*_min``/``*_max`` ranges, and set membership:

        - ``where(length=100)``
        - ``where(length_min=50, length_max=200)``
        - ``where(task=["pick", "place"])``

        The field must exist in the episode metadata; unknown fields raise
        ``ValueError`` rather than being silently ignored.
        """
        available = self.fields
        for key, value in kwargs.items():
            if key.endswith("_min"):
                column, operator = key[: -len("_min")], ">="
            elif key.endswith("_max"):
                column, operator = key[: -len("_max")], "<="
            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                column, operator = key, "in"
                value = list(value)
            else:
                column, operator = key, "=="

            if column not in available:
                raise ValueError(
                    f"unknown episode field {column!r}; available fields: "
                    f"{', '.join(sorted(available)) or '(none)'}"
                )
            self._conditions.append((column, operator, value))

        self._apply_conditions()
        return self

    def _apply_conditions(self) -> None:
        remaining = []
        for episode_index in self._episode_indices:
            episode = self._by_index[episode_index]
            if all(
                self._check_condition(episode, column, operator, value)
                for column, operator, value in self._conditions
            ):
                remaining.append(episode_index)
        self._episode_indices = remaining

    @staticmethod
    def _check_condition(
        episode: Mapping[str, Any], column: str, operator: str, value: Any
    ) -> bool:
        column_value = episode.get(column)
        if operator == "==":
            return column_value == value
        if operator == "!=":
            return column_value != value
        if operator == "in":
            return column_value in value
        if operator == ">=":
            return column_value >= value
        if operator == "<=":
            return column_value <= value
        if operator == ">":
            return column_value > value
        if operator == "<":
            return column_value < value
        raise ValueError(f"unsupported filter operator {operator!r}")

    @property
    def episode_indices(self) -> list[int]:
        """Episode indices passing all filters."""
        return self._episode_indices

    def count(self) -> int:
        """Number of episodes passing filters."""
        return len(self._episode_indices)

    def reset(self) -> EpisodeFilter:
        """Reset all filters."""
        self._episode_indices = list(self._by_index)
        self._conditions = []
        return self

    def __repr__(self) -> str:
        return (
            f"EpisodeFilter(episodes={self.count()}, "
            f"conditions={len(self._conditions)})"
        )


class EpisodeFilterBuilder:
    """Fluent builder for filter chains over episode metadata."""

    def __init__(self, source: Any):
        self._filter = EpisodeFilter(source)

    def success_only(self) -> EpisodeFilterBuilder:
        """Keep only episodes whose metadata has ``success=True``."""
        self._filter.where(success=True)
        return self

    def by_task(self, *tasks: str) -> EpisodeFilterBuilder:
        """Filter to specific task(s)."""
        self._filter.where(task=list(tasks) if len(tasks) > 1 else tasks[0])
        return self

    def by_difficulty(self, *difficulties: str) -> EpisodeFilterBuilder:
        """Filter by difficulty level(s)."""
        self._filter.where(
            difficulty=list(difficulties) if len(difficulties) > 1 else difficulties[0]
        )
        return self

    def min_length(self, length: int) -> EpisodeFilterBuilder:
        """Keep episodes with at least this many frames."""
        self._filter.where(length_min=length)
        return self

    def max_length(self, length: int) -> EpisodeFilterBuilder:
        """Keep episodes with at most this many frames."""
        self._filter.where(length_max=length)
        return self

    def min_quality(self, score: float) -> EpisodeFilterBuilder:
        """Keep episodes with ``quality_score >= score``."""
        self._filter.where(quality_score_min=score)
        return self

    def build(self) -> list[int]:
        """Return the filtered episode indices."""
        return self._filter.episode_indices

    def count(self) -> int:
        """Count filtered episodes."""
        return self._filter.count()

    def __repr__(self) -> str:
        return f"EpisodeFilterBuilder({self._filter.count()} episodes)"
