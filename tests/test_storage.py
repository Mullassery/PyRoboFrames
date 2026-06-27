"""P3 — native storage + LeRobot interop: save/round-trip and LeRobot write-back."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import pyroboframes as prf


def _make_df(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(pa.table({"log_time": [10, 20], "x": [1.0, 2.0], "ok": [True, False]}),
                   str(src / "state.parquet"))
    (src / "metadata.json").write_text(
        '{"format":"pyroboframes-columnar","version":1,'
        '"topics":[{"topic":"/state","path":"state.parquet","columns":{"x":"float64","ok":"bool"}}]}'
    )
    return prf.RoboticsDataFrame.from_converted(str(src))


def test_save_round_trips(tmp_path):
    df = _make_df(tmp_path)
    out = tmp_path / "saved"
    df.save(str(out))

    # metadata.json + stats.json written, and reload matches the original.
    assert (out / "metadata.json").exists()
    assert (out / "stats.json").exists()

    reloaded = prf.RoboticsDataFrame.from_converted(str(out))
    assert reloaded.topics == ["/state"]
    np.testing.assert_array_equal(reloaded["/state"].log_time, [10, 20])
    np.testing.assert_array_equal(reloaded["/state"].column("x"), [1.0, 2.0])

    import json

    stats = json.loads((out / "stats.json").read_text())
    assert stats["/state"]["x"]["mean"] == 1.5
    assert "ok" not in stats["/state"]  # bool column has no numeric stats


def test_lerobot_write_back_round_trips(tmp_path):
    # Two episodes of 3 frames; state[i] = [i, i+0.5], action[i] = [-i].
    n = 6
    state = np.stack([np.arange(n), np.arange(n) + 0.5], axis=1).astype(np.float32)
    action = (-np.arange(n)).reshape(n, 1).astype(np.float32)
    out = tmp_path / "lerobot_ds"

    prf.write_lerobot_dataset(
        str(out),
        features={"observation.state": state, "action": action},
        episode_lengths=[3, 3],
        fps=30.0,
    )

    # Read it back with the Rust v3.0 reader; values + episode structure must match.
    ds = prf.RoboFrameDataset.from_path(str(out))
    assert ds.num_frames == 6
    assert ds.num_episodes == 2

    batch = next(iter(ds.loader(batch_size=6, shuffle=False)))
    np.testing.assert_allclose(batch["observation.state"], state)
    np.testing.assert_allclose(batch["action"], action)
    # Episode index: first 3 frames episode 0, next 3 episode 1.
    np.testing.assert_array_equal(batch["episode_index"], [0, 0, 0, 1, 1, 1])

    # stats.json round-trips into the loader's normalization path.
    norm = ds.loader(batch_size=6, shuffle=False, normalize=["observation.state"])
    nb = next(iter(norm))
    # column 0 mean is 2.5 (arange(6)); normalized values must be centered.
    assert abs(float(nb["observation.state"][:, 0].mean())) < 1e-5


def test_lerobot_write_back_validates_inputs(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        prf.write_lerobot_dataset(
            str(tmp_path / "bad"),
            features={"observation.state": np.zeros((5, 2), np.float32)},
            episode_lengths=[3, 3],  # sums to 6, not 5
        )


def test_hub_importer_is_exported():
    # Network-dependent; just assert the entrypoint exists and errors clearly without the dep.
    assert callable(prf.download_lerobot_dataset)
