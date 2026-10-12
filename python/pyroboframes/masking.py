"""Sparse/masked data support for handling sensor failures and missing values.

Real robots have sensor outages, network dropouts, and calibration failures.
This module handles those gracefully over a :class:`~pyroboframes.dataframe.RoboticsDataFrame`
(each topic is a time-indexed table; a *missing* sample is a ``NaN`` in a numeric column,
``None`` in a nested/object column, or anything matching ``sentinel_value``).

```python
from pyroboframes.masking import MaskedDataFrame, interpolate_missing

mdf = MaskedDataFrame(df, mask_key="sensor_valid")  # optional per-topic boolean column
print(mdf.coverage_report())                         # coverage % per topic.column

df_filled = interpolate_missing(df, method="forward_fill", columns=["imu.a"])
```

Every function here is pure NumPy/Arrow (no pandas) and returns a new
:class:`RoboticsDataFrame` unless ``inplace=True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa

from .dataframe import RoboticsDataFrame as _RoboticsDataFrame
from .dataframe import _short

if TYPE_CHECKING:
    from .dataframe import RoboticsDataFrame

_VALID_METHODS = ("forward_fill", "backward_fill", "linear", "nearest")


def _is_nan_sentinel(value: object) -> bool:
    return isinstance(value, float) and np.isnan(value)


def _column_valid(col: np.ndarray, sentinel: float = np.nan) -> np.ndarray:
    """Boolean validity of a column: ``False`` where the value is missing.

    Numeric columns are missing where ``NaN`` (or equal to a non-NaN
    ``sentinel``); object columns where the element is ``None`` or ``NaN``.
    """
    if col.dtype == object:
        return np.array(
            [v is not None and not _is_nan_sentinel(v) for v in col], dtype=bool
        )
    if np.issubdtype(col.dtype, np.floating):
        valid = ~np.isnan(col)
        if not _is_nan_sentinel(sentinel):
            valid = valid & (col != sentinel)
        return valid
    # Integers/bools/strings cannot hold NaN -> always valid.
    return np.ones(len(col), dtype=bool)


def _fill_1d(values: np.ndarray, method: str) -> np.ndarray:
    """Fill ``NaN`` gaps in a 1-D numeric array by ``method`` (returns a new float64 array).

    Leading/trailing positions with no adjacent valid sample stay ``NaN`` (no
    extrapolation). A column with no valid values at all is returned unchanged.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    n = out.size
    if n == 0:
        return out

    valid = ~np.isnan(out)
    if valid.all() or not valid.any():
        return out

    idx = np.arange(n)
    valid_idx = idx[valid]

    if method == "forward_fill":
        source = np.where(valid, idx, -1)
        source = np.maximum.accumulate(source)
        fill = ~valid & (source >= 0)
        out[fill] = out[source[fill]]
    elif method == "backward_fill":
        source = np.where(valid, idx, n)
        source = np.minimum.accumulate(source[::-1])[::-1]
        fill = ~valid & (source < n)
        out[fill] = out[source[fill]]
    elif method == "nearest":
        pos = np.searchsorted(valid_idx, idx)
        left = np.clip(pos - 1, 0, valid_idx.size - 1)
        right = np.clip(pos, 0, valid_idx.size - 1)
        choose_left = np.abs(idx - valid_idx[left]) <= np.abs(valid_idx[right] - idx)
        nearest = np.where(choose_left, valid_idx[left], valid_idx[right])
        out[~valid] = out[nearest[~valid]]
    elif method == "linear":
        out[~valid] = np.interp(idx[~valid], valid_idx, out[valid])
        # np.interp clamps outside the valid range; suppress that extrapolation.
        out[idx < valid_idx[0]] = np.nan
        out[idx > valid_idx[-1]] = np.nan
    else:
        raise ValueError(
            f"unknown interpolation method {method!r}; expected one of {_VALID_METHODS}"
        )

    return out


def _select_columns(
    frame_columns: list[str], topic: str, wanted: list[str] | None
) -> list[str]:
    """Which of a topic's data columns to act on, matching either the bare column
    name (``"x"``) or the qualified ``"topic.x"`` form."""
    if wanted is None:
        return list(frame_columns)
    wanted_set = set(wanted)
    short = _short(topic)
    return [
        c for c in frame_columns if c in wanted_set or f"{short}.{c}" in wanted_set
    ]


class MaskedDataFrame:
    """Wrapper for a :class:`RoboticsDataFrame` that tracks valid data coverage.

    Coverage is reported per ``topic.column``; a value is *missing* if it is
    ``NaN``/``None``/``sentinel_value`` (or, when ``mask_key`` is given and the
    topic has a column of that name, wherever that boolean column is falsy).
    """

    def __init__(
        self,
        dataframe: RoboticsDataFrame,
        mask_key: str | None = None,
        sentinel_value: float = np.nan,
    ):
        self._df = dataframe
        self.mask_key = mask_key
        self.sentinel = sentinel_value

    def _topic_validity(self, topic: str, col_name: str) -> np.ndarray:
        """Validity mask for one ``topic.column`` (combining the optional mask_key)."""
        frame = self._df[topic]
        col = frame.column(col_name)
        valid = _column_valid(col, self.sentinel)

        if self.mask_key is not None and self.mask_key in frame.columns:
            mask_col = frame.column(self.mask_key)
            valid = valid & np.asarray(mask_col).astype(bool)

        return valid

    def coverage_by_column(self, topic: str | None = None) -> dict[str, float]:
        """Valid-data coverage (0-1) for each data column.

        Args:
            topic: restrict to one topic; ``None`` (default) reports every topic,
                keyed ``"topic.column"`` (bare ``"column"`` when a single topic is
                requested).
        """
        topics = [topic] if topic is not None else self._df.topics
        coverage: dict[str, float] = {}

        for t in topics:
            if t not in self._df:
                continue
            for col_name in self._df[t].columns:
                if col_name == self.mask_key:
                    continue
                valid = self._topic_validity(t, col_name)
                key = col_name if topic is not None else f"{_short(t)}.{col_name}"
                coverage[key] = (
                    float(np.sum(valid) / len(valid)) if len(valid) > 0 else 0.0
                )

        return coverage

    def coverage_report(self) -> str:
        """Human-readable coverage report for every topic."""
        report_lines = ["Coverage Report:", "================"]

        for topic in self._df.topics:
            report_lines.append(f"\nTopic {topic}:")
            coverage = self.coverage_by_column(topic)
            if not coverage:
                report_lines.append("  (no data columns)")
            for col, cov in coverage.items():
                pct = cov * 100
                filled = int(pct / 10)
                bar = "█" * filled + "░" * (10 - filled)
                report_lines.append(f"  {col:20} {bar} {pct:5.1f}%")

        return "\n".join(report_lines)

    def get_valid_indices(
        self, topic: str, columns: list[str] | None = None
    ) -> np.ndarray:
        """Boolean mask over ``topic``'s rows: ``True`` where every requested column is valid.

        Args:
            topic: the topic to check.
            columns: data columns to require valid (default: all of the topic's data columns).
        """
        frame = self._df[topic]
        columns = frame.columns if columns is None else columns

        valid = np.ones(len(frame), dtype=bool)
        for col_name in columns:
            if col_name not in frame.columns:
                continue
            valid = valid & self._topic_validity(topic, col_name)

        return valid


def _replace_columns(
    dataframe: RoboticsDataFrame,
    transformer,
    columns: list[str] | None,
    inplace: bool,
) -> RoboticsDataFrame:
    """Apply ``transformer(topic, col_name, values) -> new_values`` to selected numeric columns."""
    frames: dict[str, object] = {} if not inplace else None

    for topic in dataframe.topics:
        frame = dataframe[topic]
        table = frame.table
        changed = False
        for col_name in _select_columns(frame.columns, topic, columns):
            col = frame.column(col_name)
            if not (np.issubdtype(col.dtype, np.number) and col.dtype != np.bool_):
                continue
            new_values = transformer(topic, col_name, np.asarray(col, dtype=np.float64))
            if new_values is None:
                continue
            table = table.set_column(
                table.column_names.index(col_name), col_name, pa.array(new_values)
            )
            changed = True

        new_frame = frame if not changed else type(frame)(topic, table)
        if inplace:
            dataframe._frames[topic] = new_frame
        else:
            frames[topic] = new_frame

    if inplace:
        return dataframe
    return _RoboticsDataFrame(frames, dataframe.metadata)


def interpolate_missing(
    dataframe: RoboticsDataFrame,
    method: Literal["forward_fill", "backward_fill", "linear", "nearest"] = "forward_fill",
    columns: list[str] | None = None,
    inplace: bool = False,
) -> RoboticsDataFrame:
    """Interpolate missing values in numeric columns.

    Args:
        dataframe: RoboticsDataFrame with missing (``NaN``) values.
        method: ``"forward_fill"`` (last valid value), ``"backward_fill"`` (next valid
            value), ``"linear"`` (linear interpolation between valid points), or
            ``"nearest"`` (closest valid value). Leading/trailing gaps with no adjacent
            valid sample remain ``NaN`` (no extrapolation).
        columns: columns to interpolate, matching bare (``"x"``) or qualified
            (``"topic.x"``) names (default: every numeric column).
        inplace: modify ``dataframe`` in place and return it (default: return a new one).

    Returns:
        A RoboticsDataFrame with gaps filled.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"unknown interpolation method {method!r}; expected one of {_VALID_METHODS}"
        )
    return _replace_columns(
        dataframe,
        lambda _topic, _col, values: _fill_1d(values, method),
        columns,
        inplace,
    )


def fill_with_zeros(
    dataframe: RoboticsDataFrame,
    columns: list[str] | None = None,
) -> RoboticsDataFrame:
    """Fill missing values with zeros (e.g., a zero-velocity/zero-action assumption).

    Args:
        dataframe: RoboticsDataFrame with missing (``NaN``) values.
        columns: columns to fill, matching bare or qualified names. By default, fills
            every column whose name contains ``"action"`` if any exist (the "
            zero-action" use case), otherwise every numeric column.

    Returns:
        A RoboticsDataFrame with ``NaN`` → ``0``.
    """
    if columns is None:
        action_columns = [
            f"{_short(t)}.{c}"
            for t in dataframe.topics
            for c in dataframe[t].columns
            if "action" in c
        ]
        if action_columns:
            columns = action_columns
    return _replace_columns(
        dataframe,
        lambda _topic, _col, values: np.where(np.isnan(values), 0.0, values),
        columns,
        inplace=False,
    )


class SensorHealthMonitor:
    """Track sensor (topic.column) health and failure patterns over a dataset."""

    def __init__(self, dataframe: RoboticsDataFrame):
        self._df = dataframe

    def failure_rate_by_column(self) -> dict[str, float]:
        """Fraction of missing values per ``topic.column`` across the whole dataset."""
        coverage = MaskedDataFrame(self._df).coverage_by_column()
        return {col: 1.0 - cov for col, cov in coverage.items()}

    def failure_topics(self, threshold: float = 0.1) -> list[str]:
        """Topics whose *mean* column failure rate exceeds ``threshold``.

        Args:
            threshold: failure-rate threshold (e.g. ``0.1`` = 10% missing) above which
                a topic is reported for exclusion from training.
        """
        masked = MaskedDataFrame(self._df)
        failed: list[str] = []
        for topic in self._df.topics:
            coverage = masked.coverage_by_column(topic)
            if not coverage:
                continue
            mean_failure = float(np.mean([1.0 - c for c in coverage.values()]))
            if mean_failure > threshold:
                failed.append(topic)
        return failed
