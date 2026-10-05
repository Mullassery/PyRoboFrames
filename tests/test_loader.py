"""End-to-end test: generate a synthetic LeRobotDataset v3.0 with pyarrow, open it through the
Rust engine, and iterate the dataloader — verifying real state/action batches come out as NumPy.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pyroboframes as prf

CAM = "observation.images.top"
VID_W, VID_H = 32, 24


def make_dataset(root: str, episodes=2, length=50, with_video=False):
    total = episodes * length
    os.makedirs(f"{root}/meta/episodes/chunk-000")
    os.makedirs(f"{root}/data/chunk-000")

    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": episodes,
        "total_frames": total,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            CAM: {"dtype": "video", "shape": [480, 640, 3]},
            "observation.state": {"dtype": "float32", "shape": [3]},
            "action": {"dtype": "float32", "shape": [3]},
        },
    }
    with open(f"{root}/meta/info.json", "w") as fh:
        json.dump(info, fh)

    ep = pa.table(
        {
            "episode_index": pa.array(list(range(episodes)), pa.int64()),
            "length": pa.array([length] * episodes, pa.int64()),
            "dataset_from_index": pa.array(
                [i * length for i in range(episodes)], pa.int64()
            ),
            "dataset_to_index": pa.array(
                [(i + 1) * length for i in range(episodes)], pa.int64()
            ),
            "data/chunk_index": pa.array([0] * episodes, pa.int64()),
            "data/file_index": pa.array([0] * episodes, pa.int64()),
            f"videos/{CAM}/chunk_index": pa.array([0] * episodes, pa.int64()),
            f"videos/{CAM}/file_index": pa.array([0] * episodes, pa.int64()),
            f"videos/{CAM}/from_timestamp": pa.array(
                [i * length / 30 for i in range(episodes)], pa.float64()
            ),
            f"videos/{CAM}/to_timestamp": pa.array(
                [(i + 1) * length / 30 for i in range(episodes)], pa.float64()
            ),
        }
    )
    pq.write_table(ep, f"{root}/meta/episodes/chunk-000/file-000.parquet")

    state = [[float(i), 0.0, 0.0] for i in range(total)]
    action = [[float(i)] * 3 for i in range(total)]
    data = pa.table(
        {
            "observation.state": pa.array(state, pa.list_(pa.float32())),
            "action": pa.array(action, pa.list_(pa.float32())),
        }
    )
    pq.write_table(data, f"{root}/data/chunk-000/file-000.parquet")

    if with_video:
        vdir = f"{root}/videos/{CAM}/chunk-000"
        os.makedirs(vdir)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={VID_W}x{VID_H}:rate=30",
                "-frames:v",
                str(total),
                "-pix_fmt",
                "yuv420p",
                f"{vdir}/file-000.mp4",
            ],
            check=True,
        )
    return total


def test_dataset_metadata(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    assert ds.num_frames == 100
    assert ds.num_episodes == 2
    assert ds.fps == 30.0
    assert ds.cameras == [CAM]


def test_sequential_loader_yields_correct_batches(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    loader = ds.loader(batch_size=10, shuffle=False)

    assert len(loader) == 10
    batches = list(loader)
    assert len(batches) == 10

    b0 = batches[0]
    assert set(b0.keys()) >= {"observation.state", "action", "episode_index"}
    assert b0["observation.state"].shape == (10, 3)
    assert b0["observation.state"].dtype == np.float32

    # Sequential order: batch 0 row 5 is global frame 5 -> state [5, 0, 0].
    np.testing.assert_array_equal(b0["observation.state"][5], [5.0, 0.0, 0.0])
    # Last batch covers frames 90..99.
    np.testing.assert_array_equal(batches[-1]["action"][9], [99.0, 99.0, 99.0])
    # Episode boundary at frame 50.
    assert batches[4]["episode_index"][9] == 0  # frame 49
    assert batches[5]["episode_index"][0] == 1  # frame 50


def test_shuffle_is_a_permutation(tmp_path):
    total = make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    loader = ds.loader(batch_size=8, shuffle=True, shuffle_buffer=16, seed=123)

    seen = []
    for batch in loader:
        # state[:,0] encodes the global frame index by construction.
        seen.extend(int(round(x)) for x in batch["observation.state"][:, 0])

    assert sorted(seen) == list(range(total))  # every frame exactly once


def test_drop_last(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    loader = ds.loader(batch_size=30, shuffle=False, drop_last=True)
    assert len(loader) == 3  # 100 // 30
    assert all(b["observation.state"].shape[0] == 30 for b in loader)


def test_windowed_loader_returns_3d(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    loader = ds.loader(
        batch_size=10,
        shuffle=False,
        delta_timestamps={"observation.state": [-0.1, 0.0]},  # -3 frames, current
        tolerance_s=1e-3,
    )
    b0 = next(iter(loader))
    # windowed feature -> [batch, num_deltas, dim]; un-windowed feature -> [batch, 1, dim]
    assert b0["observation.state"].shape == (10, 2, 3)
    assert b0["action"].shape == (10, 1, 3)
    # row 5 == global frame 5 (episode 0): history at frame 2, current at frame 5
    np.testing.assert_array_equal(
        b0["observation.state"][5], [[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    )
    np.testing.assert_array_equal(b0["action"][5], [[5.0, 5.0, 5.0]])


def test_validate_clean_dataset(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    report = ds.validate()
    assert report.ok
    assert report.errors == []
    report.raise_if_errors()  # no-op when clean


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_frame_loader_returns_image_batches(tmp_path):
    make_dataset(str(tmp_path), with_video=True)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    loader = ds.loader(batch_size=4, shuffle=False, cameras=[CAM])

    b0 = next(iter(loader))
    frames = b0[CAM]
    assert frames.shape == (4, VID_H, VID_W, 3)  # [batch, H, W, 3]
    assert frames.dtype == np.uint8
    # tabular features still come along
    assert b0["observation.state"].shape == (4, 3)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_frame_loader_batch_frames_match_individually_decoded_frames(tmp_path):
    """Each slot in a batched `[batch, H, W, 3]` array must hold *that* frame's own pixels,
    not another frame's (or a stale/zeroed slot) — regression test for the batch-array
    assembly path, which writes each decoded frame directly into its slot in one shared
    buffer instead of building it from independent per-frame copies."""
    make_dataset(str(tmp_path), with_video=True)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    batch_size = 4
    batched = next(iter(ds.loader(batch_size=batch_size, shuffle=False, cameras=[CAM])))[CAM]
    assert batched.shape == (batch_size, VID_H, VID_W, 3)

    # ffmpeg's `testsrc` pattern scrolls, so consecutive frames differ — decoding one frame
    # at a time (batch_size=1) must reproduce each slot of the batched array exactly.
    singles = []
    for i in range(batch_size):
        single_loader = ds.loader(batch_size=1, shuffle=False, cameras=[CAM])
        it = iter(single_loader)
        for _ in range(i + 1):
            frame = next(it)[CAM][0]
        singles.append(frame)

    for i in range(batch_size):
        np.testing.assert_array_equal(
            batched[i], singles[i], err_msg=f"batch slot {i} doesn't match its own frame"
        )
    # Frames actually differ from each other (catches an offset bug that copies one frame
    # into every slot, which would otherwise pass an all-slots-equal-to-single-frame check).
    assert not np.array_equal(batched[0], batched[1])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_windowed_frame_loader_batch_slots_match_individual_decode(tmp_path):
    """Same regression as the non-windowed check above, but for the `[batch, steps, H, W, 3]`
    windowed-camera assembly path, which has its own `(idx * steps + step)` offset math."""
    make_dataset(str(tmp_path), with_video=True)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    batch_size = 3
    loader = ds.loader(
        batch_size=batch_size,
        shuffle=False,
        cameras=[CAM],
        delta_timestamps={CAM: [-1 / 30, 0.0]},
        tolerance_s=1e-3,
    )
    batched = next(iter(loader))[CAM]
    assert batched.shape == (batch_size, 2, VID_H, VID_W, 3)

    # Un-windowed loader gives the "current" (delta 0.0) frame directly, at step index -1.
    single_loader = ds.loader(batch_size=batch_size, shuffle=False, cameras=[CAM])
    current_frames = next(iter(single_loader))[CAM]
    for i in range(batch_size):
        np.testing.assert_array_equal(
            batched[i, -1],
            current_frames[i],
            err_msg=f"batch item {i}'s current-step slot doesn't match its own frame",
        )
    # Steps within an item aren't all identical (history step differs from current, except
    # right at episode start where the delta clamps) — item 1 isn't at an episode boundary.
    assert not np.array_equal(batched[1, 0], batched[1, 1])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_output_numpy_zerocopy_matches_packed_array_and_owns_its_memory(tmp_path):
    """`output='numpy_zerocopy'` must return the same pixels as the default packed-array
    path, just reshaped into a list of individually-owned arrays instead of one combined
    [batch, H, W, 3] array. Each array's `.base` is a distinct `PySliceContainer` wrapping
    its own Rust-allocated buffer -- rust-numpy's zero-copy transfer-of-ownership mechanism
    (`PyArray::from_owned_array_bound`), not a NumPy-allocated copy. `.base is None` /
    `flags['OWNDATA']` is NOT the right check here: a zero-copy array wrapping *foreign*
    (non-NumPy-allocated) memory correctly reports `OWNDATA=False` with `.base` set to its
    owning object -- that's what a genuine zero-copy foreign-buffer array looks like, not a
    sign of copying. What actually distinguishes "N independent per-frame buffers" from "N
    views into one shared packed buffer" is whether `.base` is the *same* object across
    frames (shared) or a *different* object per frame (independent) -- checked below."""
    make_dataset(str(tmp_path), with_video=True)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    batch_size = 4
    packed = next(iter(ds.loader(batch_size=batch_size, shuffle=False, cameras=[CAM])))[CAM]

    zerocopy_loader = ds.loader(
        batch_size=batch_size, shuffle=False, cameras=[CAM], output="numpy_zerocopy"
    )
    frames = next(iter(zerocopy_loader))[CAM]

    assert isinstance(frames, list)
    assert len(frames) == batch_size
    base_ids = {id(f.base) for f in frames}
    assert len(base_ids) == batch_size, (
        "frames share a base object -- they're views into one packed buffer, "
        "not independent zero-copy allocations"
    )
    for i, frame in enumerate(frames):
        assert frame.shape == (VID_H, VID_W, 3)
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(
            frame, packed[i], err_msg=f"zerocopy frame {i} differs from the packed array's slot"
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_output_numpy_zerocopy_windowed_matches_packed_array(tmp_path):
    """Same correctness check as above, for the windowed (`delta_timestamps`) nested
    list-of-lists structure."""
    make_dataset(str(tmp_path), with_video=True)
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))

    batch_size = 3
    delta_timestamps = {CAM: [-1 / 30, 0.0]}

    packed = next(
        iter(
            ds.loader(
                batch_size=batch_size,
                shuffle=False,
                cameras=[CAM],
                delta_timestamps=delta_timestamps,
                tolerance_s=1e-3,
            )
        )
    )[CAM]

    zerocopy_loader = ds.loader(
        batch_size=batch_size,
        shuffle=False,
        cameras=[CAM],
        delta_timestamps=delta_timestamps,
        tolerance_s=1e-3,
        output="numpy_zerocopy",
    )
    items = next(iter(zerocopy_loader))[CAM]

    assert isinstance(items, list)
    assert len(items) == batch_size
    all_frames = [frame for steps in items for frame in steps]
    base_ids = {id(f.base) for f in all_frames}
    assert len(base_ids) == len(all_frames), (
        "frames share a base object -- they're views into one packed buffer, "
        "not independent zero-copy allocations"
    )
    for i, steps in enumerate(items):
        assert isinstance(steps, list)
        assert len(steps) == 2
        for s, frame in enumerate(steps):
            assert frame.shape == (VID_H, VID_W, 3)
            np.testing.assert_array_equal(
                frame,
                packed[i, s],
                err_msg=f"zerocopy item {i} step {s} differs from the packed array's slot",
            )


def test_output_numpy_zerocopy_rejected_with_prefetch(tmp_path):
    """The off-GIL prefetch pipeline (`num_workers>0`) assembles batches via a separate Rust
    code path (`pyroboframes_core::pipeline`) that doesn't implement the per-frame zero-copy
    mode -- must fail clearly at construction time, not with a confusing downstream error
    from the (unrelated) framework-conversion helper."""
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    with pytest.raises(Exception):
        ds.loader(output="numpy_zerocopy", num_workers=1)


def test_output_torch(tmp_path):
    torch = pytest.importorskip("torch")
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    b0 = next(iter(ds.loader(batch_size=4, shuffle=False, output="torch")))
    assert isinstance(b0["observation.state"], torch.Tensor)
    assert tuple(b0["observation.state"].shape) == (4, 3)


def test_output_mlx(tmp_path):
    mx = pytest.importorskip("mlx.core")
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    b0 = next(iter(ds.loader(batch_size=4, shuffle=False, output="mlx")))
    assert isinstance(b0["observation.state"], mx.array)
    assert tuple(b0["observation.state"].shape) == (4, 3)


def test_output_invalid_raises(tmp_path):
    make_dataset(str(tmp_path))
    ds = prf.RoboFrameDataset.from_path(str(tmp_path))
    with pytest.raises(Exception):
        ds.loader(output="tensorflow")


def test_bad_path_raises(tmp_path):
    with pytest.raises(Exception):
        prf.RoboFrameDataset.from_path(str(tmp_path / "nope"))
