"""Robotics DataFrame — a typed, time-indexed, multi-sensor view over converted robotics logs.

The ingest converters (``convert_mcap`` / ``convert_ros2_bag``) write one Parquet table per topic
plus a ``metadata.json`` manifest. :class:`RoboticsDataFrame` loads that output and gives it a
robotics-aware API:

- per-topic access (each topic is a time-indexed :class:`TopicFrame` with a ``log_time`` column),
- time slicing across every topic at once (:meth:`RoboticsDataFrame.slice`),
- **time-synchronized alignment** — an as-of join that snaps every other sensor onto a reference
  topic's timestamps (:meth:`RoboticsDataFrame.align`), the basis for multi-sensor fusion.

This is the abstraction that makes PyRoboFrames a data *platform* rather than just a loader: raw
heterogeneous logs become one coherent, queryable, time-aligned table.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _short(topic: str) -> str:
    """Topic name → a column prefix: ``/imu/data`` → ``imu.data``."""
    return topic.strip("/").replace("/", ".") or "topic"


class TopicFrame:
    """One topic's messages as a time-indexed table (a ``log_time`` column plus data columns)."""

    def __init__(self, name: str, table: pa.Table):
        self.name = name
        self.table = table

    @property
    def log_time(self) -> np.ndarray:
        """Message timestamps (nanoseconds), in stored order."""
        return self.table.column("log_time").to_numpy(zero_copy_only=False)

    @property
    def columns(self) -> list[str]:
        """Data column names (everything except ``log_time``)."""
        return [c for c in self.table.column_names if c != "log_time"]

    def column(self, name: str) -> np.ndarray:
        return self.table.column(name).to_numpy(zero_copy_only=False)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Every column (including ``log_time``) as NumPy arrays."""
        return {c: self.table.column(c).to_numpy(zero_copy_only=False) for c in self.table.column_names}

    def __len__(self) -> int:
        return self.table.num_rows

    def __repr__(self) -> str:
        return f"TopicFrame({self.name!r}, rows={len(self)}, columns={self.columns})"


class AlignedFrame:
    """Result of :meth:`RoboticsDataFrame.align`: a single table on the reference topic's
    timestamps, with each other topic's columns snapped on (as-of) and prefixed by topic name.
    Missing matches are ``NaN`` (numeric) or ``None`` (other)."""

    def __init__(self, data: dict[str, np.ndarray]):
        self._data = data

    @property
    def log_time(self) -> np.ndarray:
        return self._data["log_time"]

    @property
    def columns(self) -> list[str]:
        return [c for c in self._data if c != "log_time"]

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def __len__(self) -> int:
        return len(self._data["log_time"])

    def to_pandas(self):
        import pandas as pd  # optional dependency

        return pd.DataFrame(self._data)

    def __repr__(self) -> str:
        return f"AlignedFrame(rows={len(self)}, columns={self.columns})"


class RoboticsDataFrame:
    """A multi-topic, time-indexed view over a converted robotics log."""

    def __init__(self, frames: dict[str, TopicFrame], metadata: dict | None = None):
        self._frames = frames
        self.metadata = metadata or {}

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_converted(cls, path: str) -> "RoboticsDataFrame":
        """Load a directory written by ``convert_mcap`` / ``convert_ros2_bag`` (Parquet tables +
        ``metadata.json``). Falls back to globbing ``*.parquet`` if no manifest is present."""
        meta_path = os.path.join(path, "metadata.json")
        frames: dict[str, TopicFrame] = {}
        metadata: dict = {}
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                metadata = json.load(fh)
            for entry in metadata.get("topics", []):
                table = pq.read_table(os.path.join(path, entry["path"]))
                frames[entry["topic"]] = TopicFrame(entry["topic"], table)
        else:
            for fn in sorted(os.listdir(path)):
                if fn.endswith(".parquet"):
                    topic = "/" + fn[: -len(".parquet")]
                    frames[topic] = TopicFrame(topic, pq.read_table(os.path.join(path, fn)))
        return cls(frames, metadata)

    @classmethod
    def from_mcap(cls, mcap_path: str, out_dir: str | None = None) -> "RoboticsDataFrame":
        """Convert an MCAP log and load it. ``out_dir`` defaults to a temp directory."""
        from . import convert_mcap

        out_dir = out_dir or tempfile.mkdtemp(prefix="prf_mcap_")
        convert_mcap(mcap_path, out_dir)
        return cls.from_converted(out_dir)

    @classmethod
    def from_ros2_bag(cls, bag_path: str, out_dir: str | None = None) -> "RoboticsDataFrame":
        """Convert a ROS 2 bag (`.db3`) and load it. ``out_dir`` defaults to a temp directory."""
        from . import convert_ros2_bag

        out_dir = out_dir or tempfile.mkdtemp(prefix="prf_bag_")
        convert_ros2_bag(bag_path, out_dir)
        return cls.from_converted(out_dir)

    # ---- access -------------------------------------------------------------

    @property
    def topics(self) -> list[str]:
        return list(self._frames)

    def __getitem__(self, topic: str) -> TopicFrame:
        return self._frames[topic]

    def __contains__(self, topic: str) -> bool:
        return topic in self._frames

    def __iter__(self) -> Iterator[str]:
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def time_range(self) -> tuple[int, int] | None:
        """``(min, max)`` ``log_time`` across all non-empty topics, or ``None`` if empty."""
        starts, ends = [], []
        for f in self._frames.values():
            if len(f):
                t = f.log_time
                starts.append(int(t.min()))
                ends.append(int(t.max()))
        return (min(starts), max(ends)) if starts else None

    # ---- operations ---------------------------------------------------------

    def slice(self, start: int, end: int) -> "RoboticsDataFrame":
        """A new frame with every topic restricted to ``start <= log_time < end`` (nanoseconds)."""
        out: dict[str, TopicFrame] = {}
        for name, frame in self._frames.items():
            t = frame.log_time
            mask = (t >= start) & (t < end)
            out[name] = TopicFrame(name, frame.table.filter(pa.array(mask)))
        return RoboticsDataFrame(out, self.metadata)

    def align(self, reference: str, tolerance: int | None = None) -> AlignedFrame:
        """Time-synchronize every topic onto ``reference``'s timestamps with a backward as-of join:
        each reference row gets the most recent (``log_time <=`` it) row of every other topic.

        ``tolerance`` (nanoseconds), if given, drops matches older than it (→ ``NaN``/``None``).
        Other topics' columns are prefixed by topic name (e.g. ``imu.accel.x``).
        """
        if reference not in self._frames:
            raise KeyError(f"reference topic {reference!r} not in {self.topics}")

        ref = self._frames[reference]
        ref_t = ref.log_time
        order = np.argsort(ref_t, kind="stable")
        ref_t_sorted = ref_t[order]

        data: dict[str, np.ndarray] = {"log_time": ref_t_sorted}
        for c in ref.columns:
            data[c] = ref.column(c)[order]

        for name, frame in self._frames.items():
            if name == reference or len(frame) == 0:
                continue
            t = frame.log_time
            fo = np.argsort(t, kind="stable")
            t_sorted = t[fo]
            # Backward as-of: index of the last sample at or before each reference timestamp.
            idx = np.searchsorted(t_sorted, ref_t_sorted, side="right") - 1
            valid = idx >= 0
            if tolerance is not None:
                safe = np.clip(idx, 0, len(t_sorted) - 1)
                dt = np.where(valid, ref_t_sorted - t_sorted[safe], np.iinfo(np.int64).max)
                valid &= dt <= tolerance

            prefix = _short(name)
            for c in frame.columns:
                col = frame.column(c)[fo]
                data[f"{prefix}.{c}"] = _gather(col, idx, valid)

        return AlignedFrame(data)

    def __repr__(self) -> str:
        return f"RoboticsDataFrame(topics={self.topics})"


def _gather(col: np.ndarray, idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Gather ``col[idx]`, marking invalid positions as ``NaN`` (numeric) or ``None`` (other)."""
    safe = np.clip(idx, 0, len(col) - 1)
    out = col[safe]
    if not valid.all():
        if np.issubdtype(out.dtype, np.floating) or np.issubdtype(out.dtype, np.integer):
            out = out.astype(np.float64)
            out[~valid] = np.nan
        else:
            out = out.astype(object)
            out[~valid] = None
    return out
