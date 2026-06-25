"""Tests for the v0.1.x quick-win additions: dataset statistics (meta/stats.json),
deterministic train/val split, and loader checkpoint/resume (position + seek).
"""

import json

import numpy as np

import pyroboframes as prf

from test_loader import make_dataset


def _write_stats(root: str):
    stats = {
        "observation.state": {
            "mean": [1.0, 2.0, 3.0],
            "std": [0.5, 0.5, 0.5],
            "min": [0.0, 0.0, 0.0],
            "max": [9.0, 9.0, 9.0],
            "count": 100,
        },
        "action": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    }
    with open(f"{root}/meta/stats.json", "w") as fh:
        json.dump(stats, fh)


def test_stats_absent_is_none(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    assert ds.stats() is None


def test_stats_parsed(tmp_path):
    make_dataset(str(tmp_path))
    _write_stats(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    stats = ds.stats()
    assert stats is not None
    s = stats["observation.state"]
    assert s["mean"] == [1.0, 2.0, 3.0]
    assert s["max"] == [9.0, 9.0, 9.0]
    assert s["count"] == 100
    assert stats["action"]["std"] == [1.0, 1.0, 1.0]
    assert stats["action"]["count"] is None


def test_train_val_split_partitions_episodes(tmp_path):
    make_dataset(str(tmp_path), episodes=10, length=5)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    train, val = ds.train_val_split(val_fraction=0.2, seed=0)
    assert len(val) == 2
    assert len(train) == 8
    assert sorted(train + val) == list(range(10))
    # Deterministic for a fixed seed.
    assert (train, val) == ds.train_val_split(val_fraction=0.2, seed=0)


def test_episodes_metadata(tmp_path):
    make_dataset(str(tmp_path), episodes=3, length=20)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    eps = ds.episodes()
    assert len(eps) == 3
    assert eps[0] == {"episode_index": 0, "length": 20, "from_index": 0, "to_index": 20}
    assert eps[2]["from_index"] == 40 and eps[2]["to_index"] == 60


def test_loader_normalize(tmp_path):
    make_dataset(str(tmp_path))
    # state[i] = [i,0,0]; with mean [10,0,0] std [2,1,1] -> ((i-10)/2, 0, 0).
    with open(f"{tmp_path}/meta/stats.json", "w") as fh:
        json.dump(
            {
                "observation.state": {"mean": [10.0, 0.0, 0.0], "std": [2.0, 1.0, 1.0]},
                "action": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
            },
            fh,
        )
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    loader = ds.loader(batch_size=10, shuffle=False, normalize=["observation.state"])
    b0 = next(iter(loader))
    np.testing.assert_allclose(b0["observation.state"][5], [(5 - 10) / 2, 0.0, 0.0])
    np.testing.assert_array_equal(b0["action"][5], [5.0, 5.0, 5.0])  # untouched


def test_prefetch_matches_sync(tmp_path):
    total = make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    def frames(**kw):
        seen = []
        for b in ds.loader(batch_size=8, shuffle=False, **kw):
            seen.extend(int(round(x)) for x in b["observation.state"][:, 0])
        return seen

    sync = frames()
    pre = frames(num_workers=3, prefetch=2)  # off-GIL prefetch pipeline
    assert sync == pre == list(range(total))  # same order, every frame once
    assert len(ds.loader(batch_size=8, num_workers=2)) == len(ds.loader(batch_size=8))


def test_loader_filters_to_split_episodes(tmp_path):
    # 4 episodes x 25 frames; episode e owns global frames [e*25, (e+1)*25).
    make_dataset(str(tmp_path), episodes=4, length=25)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    train, val = ds.train_val_split(val_fraction=0.5, seed=1)
    assert len(train) == 2 and len(val) == 2

    def frames(loader):
        seen = []
        for b in loader:
            seen.extend(int(round(x)) for x in b["observation.state"][:, 0])
        return sorted(seen)

    def expected(eps):
        return sorted(f for e in eps for f in range(e * 25, (e + 1) * 25))

    train_frames = frames(ds.loader(batch_size=8, shuffle=True, seed=3, episodes=train))
    val_frames = frames(ds.loader(batch_size=8, shuffle=False, episodes=val))

    assert train_frames == expected(train)
    assert val_frames == expected(val)
    # No leakage: train and val frames are disjoint and together cover everything.
    assert set(train_frames).isdisjoint(val_frames)
    assert sorted(train_frames + val_frames) == list(range(100))


def test_loader_position_and_seek_resume(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    # Consume two batches, remember where we are.
    loader = ds.loader(batch_size=10, shuffle=False)
    assert loader.position == 0
    it = iter(loader)
    next(it)
    next(it)
    assert loader.position == 20

    # Resume a fresh (identical) loader at the saved position; remaining frames must match.
    resumed = ds.loader(batch_size=10, shuffle=False)
    resumed.seek(20)
    assert resumed.position == 20
    remaining = []
    for b in resumed:
        remaining.extend(int(round(x)) for x in b["observation.state"][:, 0])
    assert remaining == list(range(20, 100))  # frames 0..19 were skipped

    # Seek past the end is clamped (no error, empty iteration).
    done = ds.loader(batch_size=10, shuffle=False)
    done.seek(10_000)
    assert list(done) == []
