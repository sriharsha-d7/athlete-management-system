"""
Ground-truth data-generating process for the synthetic league.

Everything here is a pure function of *feature arrays whose names match the dbt
feature mart* (`fct_athlete_day`). The simulator calls these functions to decide
who gets injured and how well they play; `ml/validate.py` calls the very same
functions to ask: "did the recommendation engine recover the true effect of each
lever?". That is only possible because the data is synthetic and the causal
structure is known, which is the main reason to simulate rather than scrape.
"""
from __future__ import annotations

import numpy as np

# Population mean / sd used to standardise physical qualities.
POP = {
    "yoyo": (2100.0, 350.0),
    "sprint": (4.20, 0.14),
    "cmj": (38.0, 4.5),
    "squat": (1.70, 0.28),
    "nordic": (340.0, 55.0),
}

# Baseline log-rate of a time-loss injury per weighted exposure-hour.
# Tuned so the league averages roughly 2 time-loss injuries per player-season.
B0_INJURY = -7.30

# Exposure weights per session-hour (a match hour is ~5x riskier than training).
EXPOSURE_WEIGHT = {"match": 5.0, "training": 1.0, "gym": 0.4, "recovery": 0.2,
                   "individual": 0.4, "rehab": 0.15}


def _c(x, lo, hi):
    return np.clip(x, lo, hi)


def fitness_composite(yoyo, sprint, cmj, squat):
    zy = (yoyo - POP["yoyo"][0]) / POP["yoyo"][1]
    zs = -(sprint - POP["sprint"][0]) / POP["sprint"][1]  # faster = better
    zc = (cmj - POP["cmj"][0]) / POP["cmj"][1]
    zq = (squat - POP["squat"][0]) / POP["squat"][1]
    return 0.35 * zy + 0.25 * zs + 0.20 * zc + 0.20 * zq


def rating_components(f: dict) -> dict:
    """Additive contribution of each driver to a match performance rating."""
    dsr = np.asarray(f["days_since_return"], dtype=float)
    returning = np.where(np.isnan(dsr), 0.0, (dsr <= 21).astype(float))
    return {
        "fitness": 0.50 * fitness_composite(f["yoyo_ir1_m"], f["sprint_30m_s"],
                                            f["cmj_cm"], f["squat_1rm_rel"]),
        "sleep": 0.32 * _c(f["sleep_hours_3d"] - 8.0, -3.0, 0.5),
        "carbs": 0.12 * _c(f["carbs_gpkg_3d"] - 6.0, -4.0, 1.5),
        "protein": 0.35 * _c(f["protein_gpkg_7d"] - 1.6, -0.8, 0.3),
        "hydration": 0.10 * _c(f["hydration_l_3d"] - 3.0, -1.5, 0.5),
        "load_balance": -0.9 * _c(f["acwr"] - 1.3, 0, 1.0) - 0.6 * _c(0.8 - f["acwr"], 0, 0.5),
        "recovery_state": 0.18 * _c(f["hrv_z"], -2.5, 2.5),
        "return_from_injury": -0.30 * returning,
    }


def rating_signal(f: dict) -> np.ndarray:
    return sum(rating_components(f).values())


def injury_log_rate(f: dict, frailty_log=0.0) -> np.ndarray:
    """Log injury rate per weighted exposure-hour (excluding the exposure term)."""
    dsr = np.asarray(f["days_since_return"], dtype=float)
    returning = np.where(np.isnan(dsr), 0.0, ((dsr >= 1) & (dsr <= 14)).astype(float))
    return (
        B0_INJURY
        + 1.7 * _c(f["acwr"] - 1.3, 0, 1.2)                    # acute load spike
        + 0.9 * _c(0.75 - f["acwr"], 0, 0.5)                   # under-prepared
        + 0.30 * _c(8.0 - f["sleep_hours_7d"], 0, 3.0)         # sleep debt
        + 0.9 * _c(1.6 - f["protein_gpkg_7d"], 0, 1.0)         # protein shortfall
        + 0.60 * _c((340.0 - f["nordic_n"]) / 55.0, 0, 2.5)    # posterior-chain weakness
        + 0.22 * np.minimum(f["prior_injuries"], 5)            # injury history
        + 0.045 * (f["age_years"] - 26.0)
        + 0.35 * _c(-f["hrv_z"], 0, 3.0)                       # suppressed HRV
        + 0.60 * returning                                     # first 2 weeks back
        + frailty_log                                          # unobserved heterogeneity
    )
