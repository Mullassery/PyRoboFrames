"""Sparse/masked data support (masking.py): coverage, interpolation, zero-fill.

These exercise the module against the real topic-based ``RoboticsDataFrame``
(the API ``masking.py`` previously referenced — ``num_episodes``,
``slice(episode_index=...)``, ``to_pyarrow()`` — never existed, so every
function was effectively dead code / a no-op).
"""

import numpy as np
import pyarrow as pa
from pyroboframes.dataframe import RoboticsDataFrame, TopicFrame
from pyroboframes.masking import (
    MaskedDataFrame,
    SensorHealthMonitor,
    fill_with_zeros,
    interpolate_missing,
)


def _df() -> RoboticsDataFrame:
    state = pa.table(
        {
            "log_time": [1, 2, 3, 4, 5],
            "x": [1.0, np.nan, np.nan, 4.0, 5.0],
            "action": [0.1, np.nan, 0.3, np.nan, 0.5],
        }
    )
    imu = pa.table({"log_time": [1, 2, 3], "a": [np.nan, 2.0, 3.0]})
    return RoboticsDataFrame(
        {"/state": TopicFrame("/state", state), "/imu": TopicFrame("/imu", imu)}
    )


def test_coverage_by_column_reports_real_fractions():
    cov = MaskedDataFrame(_df()).coverage_by_column()
    assert cov["state.x"] == 3 / 5
    assert cov["state.action"] == 3 / 5
    assert cov["imu.a"] == 2 / 3


def test_coverage_single_topic_and_valid_indices():
    mdf = MaskedDataFrame(_df())
    assert mdf.coverage_by_column("/imu") == {"a": 2 / 3}
    np.testing.assert_array_equal(
        mdf.get_valid_indices("/state", columns=["x"]), [True, False, False, True, True]
    )


def test_coverage_report_mentions_topics():
    report = MaskedDataFrame(_df()).coverage_report()
    assert "Topic /state" in report and "  x " in report and "%" in report


def test_forward_fill_leaves_leading_gap():
    filled = interpolate_missing(_df(), method="forward_fill")
    np.testing.assert_array_equal(filled["/state"].column("x"), [1.0, 1.0, 1.0, 4.0, 5.0])
    # imu.a's first sample has no earlier valid value -> stays NaN.
    imu = filled["/imu"].column("a")
    assert np.isnan(imu[0])
    np.testing.assert_array_equal(imu[1:], [2.0, 3.0])


def test_backward_fill_and_linear_and_nearest():
    df = _df()
    bf = interpolate_missing(df, method="backward_fill")
    np.testing.assert_array_equal(bf["/state"].column("x"), [1.0, 4.0, 4.0, 4.0, 5.0])

    lin = interpolate_missing(df, method="linear")
    np.testing.assert_allclose(lin["/state"].column("x"), [1.0, 2.0, 3.0, 4.0, 5.0])

    near = interpolate_missing(df, method="nearest")
    np.testing.assert_array_equal(near["/imu"].column("a"), [2.0, 2.0, 3.0])


def test_interpolate_does_not_mutate_by_default_but_does_inplace():
    df = _df()
    out = interpolate_missing(df, method="forward_fill")
    assert np.isnan(df["/state"].column("x")[1])  # original untouched
    assert out is not df

    same = interpolate_missing(df, method="forward_fill", inplace=True)
    assert same is df
    np.testing.assert_array_equal(df["/state"].column("x"), [1.0, 1.0, 1.0, 4.0, 5.0])


def test_unknown_method_raises():
    import pytest

    with pytest.raises(ValueError):
        interpolate_missing(_df(), method="spline")  # type: ignore[arg-type]


def test_fill_with_zeros_defaults_to_action_columns():
    filled = fill_with_zeros(_df())
    np.testing.assert_array_equal(
        filled["/state"].column("action"), [0.1, 0.0, 0.3, 0.0, 0.5]
    )
    # x is not an action column, so it is left as-is.
    assert np.isnan(filled["/state"].column("x")[1])


def test_fill_with_zeros_specific_columns():
    df = _df()
    filled = fill_with_zeros(df, columns=["state.x"])
    np.testing.assert_array_equal(filled["/state"].column("x"), [1.0, 0.0, 0.0, 4.0, 5.0])
    assert np.isnan(filled["/state"].column("action")[1])


def test_sensor_health_monitor():
    monitor = SensorHealthMonitor(_df())
    rates = monitor.failure_rate_by_column()
    assert abs(rates["state.x"] - 2 / 5) < 1e-12
    assert abs(rates["imu.a"] - 1 / 3) < 1e-12
    # /state is 40%/40% missing on its two columns -> mean 0.4 > 0.1.
    assert set(monitor.failure_topics(threshold=0.1)) == {"/state", "/imu"}
    assert monitor.failure_topics(threshold=0.9) == []
