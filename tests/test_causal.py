"""The ground-truth model must encode the relationships the README claims. If someone edits causal.py these fail loudly."""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "data_gen"))
import causal as C  # noqa: E402


def base(**over):
    f = dict(yoyo_ir1_m=2100.0, sprint_30m_s=4.20, cmj_cm=38.0, squat_1rm_rel=1.7, nordic_n=340.0,
             sleep_hours_3d=8.0, sleep_hours_7d=8.0, carbs_gpkg_3d=6.0, protein_gpkg_7d=1.7, hydration_l_3d=3.0,
             acwr=1.0, hrv_z=0.0, days_since_return=np.nan, prior_injuries=0.0, age_years=26.0)
    f.update(over)
    return {k: np.array([v], dtype=float) for k, v in f.items()}


def test_more_sleep_never_hurts_performance_or_safety():
    lo, hi = base(sleep_hours_3d=6.0, sleep_hours_7d=6.0), base(sleep_hours_3d=7.5, sleep_hours_7d=7.5)
    assert C.rating_signal(hi) > C.rating_signal(lo)
    assert C.injury_log_rate(hi) < C.injury_log_rate(lo)


def test_acwr_spike_raises_injury_rate_and_lowers_rating():
    ok, spike = base(acwr=1.0), base(acwr=1.8)
    assert C.injury_log_rate(spike) > C.injury_log_rate(ok)
    assert C.rating_signal(spike) < C.rating_signal(ok)


def test_acwr_sweet_spot_is_flat():
    a, b = base(acwr=0.9), base(acwr=1.25)
    assert C.injury_log_rate(a) == C.injury_log_rate(b)


def test_weak_hamstrings_increase_injury_but_not_rating():
    weak, strong = base(nordic_n=230.0), base(nordic_n=400.0)
    assert C.injury_log_rate(weak) > C.injury_log_rate(strong)
    assert C.rating_signal(weak) == C.rating_signal(strong)


def test_faster_sprint_raises_rating():
    assert C.rating_signal(base(sprint_30m_s=4.05)) > C.rating_signal(base(sprint_30m_s=4.35))


def test_exposure_weights_rank_match_above_training():
    assert C.EXPOSURE_WEIGHT["match"] > C.EXPOSURE_WEIGHT["training"] > C.EXPOSURE_WEIGHT["recovery"]
