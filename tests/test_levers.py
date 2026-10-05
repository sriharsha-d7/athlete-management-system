import numpy as np
import pandas as pd
import pytest

from ml.levers import apply_lever, parse_shifts


def frame():
    return pd.DataFrame({
        "sleep_hours_3d": [6.0, 7.0], "sleep_hours_7d": [6.5, 7.5], "sleep_hours_28d": [6.8, 7.2],
        "acute_load_7d": [500.0, 300.0], "chronic_load_28d": [250.0, 300.0],
        "acwr": [2.0, 1.0], "load_strain_7d": [4000.0, 2000.0],
    })


def test_parse_shifts():
    assert parse_shifts("a:1.0|b:0.5") == [("a", 1.0), ("b", 0.5)]


def test_add_lever_weights_and_does_not_mutate_input():
    df = frame()
    out = apply_lever(df, parse_shifts("sleep_hours_3d:1.0|sleep_hours_28d:0.5"), "add", 1.0)
    assert out.sleep_hours_3d.tolist() == [7.0, 8.0]
    assert out.sleep_hours_28d.tolist() == pytest.approx([7.3, 7.7])
    assert out.sleep_hours_7d.tolist() == df.sleep_hours_7d.tolist()   # untouched feature
    assert df.sleep_hours_3d.tolist() == [6.0, 7.0]                    # input unchanged


def test_per_row_steps():
    out = apply_lever(frame(), parse_shifts("sleep_hours_3d:1.0"), "add", np.array([0.5, 0.0]))
    assert out.sleep_hours_3d.tolist() == [6.5, 7.0]


def test_scale_lever_recomputes_derived_features():
    out = apply_lever(frame(), parse_shifts("acute_load_7d:1.0"), "scale", -0.2)
    assert out.acute_load_7d.tolist() == pytest.approx([400.0, 240.0])
    assert out.acwr.tolist() == pytest.approx([1.6, 0.8])               # acute / chronic recomputed
    assert out.load_strain_7d.tolist() == pytest.approx([3200.0, 1600.0])


def test_unknown_step_type_raises():
    with pytest.raises(ValueError):
        apply_lever(frame(), [("sleep_hours_3d", 1.0)], "multiply", 1.0)
