"""Shared ML configuration. Features are NOT listed here: every non-metadata numeric column in the
dbt ML marts is a feature, so dbt (macros/ml_feature_columns.sql) stays the single source of truth."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models"

# Time-based split: never train on the future of anything you evaluate on.
TRAIN_END = "2025-12-31"
VAL_END = "2026-03-31"          # validation: Jan-Mar 2026 (early stopping + calibration)
                                # test:       Apr-Sep 2026 (never touched until final evaluation)

INJURY_META = ["athlete_id", "date_day", "position_group", "injury_next_7d", "label_is_complete"]
PERF_META = ["athlete_id", "date_day", "position_group", "performance_rating", "minutes_played", "opponent_strength"]
POSITION_GROUPS = ["Goalkeeper", "Defender", "Wide", "Midfielder", "Forward"]

# Domain priors enforced as monotonic constraints. They make counterfactual "what if he slept an hour
# more" answers physiologically sensible instead of artefacts of a noisy tree split. +1 = prediction
# must not decrease as the feature increases, -1 = must not increase. Features not listed are free.
MONOTONE_INJURY = {
    "sleep_hours_3d": -1, "sleep_hours_7d": -1, "sleep_hours_28d": -1,
    "protein_gpkg_7d": -1, "nordic_n": -1, "hrv_z": -1,
}
MONOTONE_PERF = {
    "sleep_hours_3d": 1, "sleep_hours_7d": 1, "protein_gpkg_7d": 1,
    "carbs_gpkg_3d": 1, "carbs_gpkg_7d": 1, "hydration_l_3d": 1, "hydration_l_7d": 1,
    "yoyo_ir1_m": 1, "cmj_cm": 1, "squat_1rm_rel": 1, "sprint_30m_s": -1,
    "hrv_z": 1, "rating_avg_5m": 1,
}

# Calibrated 7-day injury probability -> traffic-light band
RISK_BANDS = [(0.20, "Critical"), (0.10, "High"), (0.05, "Moderate"), (0.0, "Low")]

STATE_ROWS = 14        # recommendation engine averages counterfactuals over the athlete's last N available days
STATE_MAX_AGE_DAYS = 45
SEED = 7


def risk_band(p: float) -> str:
    for cut, name in RISK_BANDS:
        if p >= cut:
            return name
    return "Low"
