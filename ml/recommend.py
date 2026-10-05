#!/usr/bin/env python
"""
Counterfactual lever engine: "what does this athlete need MORE of to perform better and stay available?"

For every athlete and every lever in the dbt seed catalog (sleep, protein, carbs, hydration, load,
aerobic capacity, speed, power, strength, hamstring strength) we:

  1. take the athlete's last N available days as their current state,
  2. cap the lever's step at the athlete's actual shortfall against target (dbt gap analysis) -
     no point telling someone already at 9 h of sleep to sleep more,
  3. move the features that lever controls (seed `feature_shifts`),
  4. re-score BOTH models and measure
        expected_rating_gain          = E[rating | after] - E[rating | now]
        expected_risk_reduction_pp    = P(injury 7d | now) - P(injury 7d | after)   (percentage points)
  5. rank levers per athlete by a transparent priority score.

Because the data is synthetic, the true causal effect of every lever is known (data_gen/causal.py), so the
same frames are also scored against ground truth to check the engine recovers the right effects.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_gen"))
from ml import config as cfg  # noqa: E402
from ml.data import load_injury_frame, make_X  # noqa: E402
from ml.levers import apply_lever, parse_shifts  # noqa: E402
from warehouse.conn import Warehouse  # noqa: E402

try:
    import causal as truth  # data_gen/causal.py: ground-truth data-generating process
except Exception:  # pragma: no cover
    truth = None


def current_state(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Last N available days per athlete, no older than STATE_MAX_AGE_DAYS."""
    d = df[(df.date_day <= as_of) & (df.date_day >= as_of - pd.Timedelta(days=cfg.STATE_MAX_AGE_DAYS))]
    d = d.sort_values(["athlete_id", "date_day"])
    return d.groupby("athlete_id", group_keys=False).tail(cfg.STATE_ROWS).reset_index(drop=True)


def score(bundle, frame):
    inj, perf = bundle["injury"], bundle["performance"]
    Xi = make_X(frame, inj["features"])[inj["columns"]]
    Xp = make_X(frame, perf["features"])[perf["columns"]]
    p_inj = inj["calibrator"].predict(inj["model"].predict_proba(Xi)[:, 1])
    return p_inj, perf["model"].predict(Xp)


def truth_effects(before: pd.DataFrame, after: pd.DataFrame, fill: pd.Series):
    """Ground-truth change in rating signal and injury log-rate for the same frames (None if unavailable)."""
    if truth is None:
        return None, None
    cols = ["yoyo_ir1_m", "sprint_30m_s", "cmj_cm", "squat_1rm_rel", "nordic_n", "sleep_hours_3d", "sleep_hours_7d",
            "carbs_gpkg_3d", "protein_gpkg_7d", "hydration_l_3d", "acwr", "hrv_z", "prior_injuries", "age_years"]
    f0 = {c: before[c].fillna(fill[c]).values for c in cols}
    f1 = {c: after[c].fillna(fill[c]).values for c in cols}
    f0["days_since_return"] = before["days_since_return"].values.astype(float)
    f1["days_since_return"] = after["days_since_return"].values.astype(float)
    d_rating = truth.rating_signal(f1) - truth.rating_signal(f0)
    d_lograte = truth.injury_log_rate(f1) - truth.injury_log_rate(f0)   # negative = safer
    return d_rating, d_lograte


def build_recommendations(bundle, wh, with_truth=True):
    df = load_injury_frame(wh)
    as_of = pd.to_datetime(wh.query("select max(date_day) d from marts.fct_athlete_day")["d"].iloc[0])
    state = current_state(df, as_of)
    gaps = wh.query("select athlete_id, metric, current_value, target_value, shortfall, needed_direction "
                    "from marts.mart_athlete_gap_analysis")
    levers = wh.query("select * from reference.seed_lever_catalog")
    fill = state.select_dtypes("number").median()

    p0, r0 = score(bundle, state)
    state = state.assign(_p0=p0, _r0=r0)
    rows = []
    for lv in levers.itertuples():
        shifts = parse_shifts(lv.feature_shifts)
        g = gaps[gaps.metric == lv.metric].set_index("athlete_id")
        m = state.athlete_id.map(g.shortfall)
        direction = state.athlete_id.map(g.needed_direction)
        target = state.athlete_id.map(g.target_value)
        cur = state.athlete_id.map(g.current_value)
        if lv.step_type == "add":
            sign = np.sign(lv.step_size)
            mag = np.minimum(abs(lv.step_size), m.fillna(0).values)           # cap at the actual shortfall
            step = sign * mag
        else:                                                                  # 'scale': direction from the band
            dirn = direction.map({"decrease": -1.0, "increase": 1.0}).fillna(0.0).values
            step = dirn * abs(lv.step_size)
        step = np.where(np.isnan(step), 0.0, step)
        if not (np.abs(step) > 0).any():
            continue
        after = apply_lever(state, shifts, lv.step_type, step)
        p1, r1 = score(bundle, after)
        t_rating, t_lograte = truth_effects(state, after, fill) if with_truth else (None, None)
        res = pd.DataFrame({"athlete_id": state.athlete_id, "step": step, "_p0": p0, "_p1": p1,
                            "d_rating": r1 - r0})
        if t_rating is not None:
            res["t_d_rating"], res["t_d_lograte"] = t_rating, t_lograte
        agg = res.groupby("athlete_id").agg(
            recommended_step=("step", "mean"), p0=("_p0", "mean"), p1=("_p1", "mean"),
            expected_rating_gain=("d_rating", "mean"),
            **({"true_rating_gain": ("t_d_rating", "mean"), "true_lograte_change": ("t_d_lograte", "mean")}
               if t_rating is not None else {}),
            n_state_rows=("step", "size")).reset_index()
        agg = agg[agg.recommended_step.abs() > 0]
        agg["lever_id"] = lv.lever_id
        agg["current_value"] = agg.athlete_id.map(cur.groupby(state.athlete_id).first())
        agg["target_value"] = agg.athlete_id.map(target.groupby(state.athlete_id).first())
        agg["expected_risk_reduction_pp"] = (agg.p0 - agg.p1) * 100.0
        agg["expected_relative_risk_reduction"] = (agg.p0 - agg.p1) / agg.p0.clip(lower=1e-6)
        if t_rating is not None:
            agg["true_relative_risk_reduction"] = 1.0 - np.exp(agg.true_lograte_change)
        rows.append(agg)
    rec = pd.concat(rows, ignore_index=True)

    # transparent priority score: equal weight on normalised performance gain and risk reduction
    gmax = max(rec.expected_rating_gain.clip(lower=0).max(), 1e-9)
    rmax = max(rec.expected_risk_reduction_pp.clip(lower=0).max(), 1e-9)
    rec["priority_score"] = 0.5 * rec.expected_rating_gain.clip(lower=0) / gmax \
        + 0.5 * rec.expected_risk_reduction_pp.clip(lower=0) / rmax
    rec["priority_rank"] = rec.groupby("athlete_id").priority_score.rank(ascending=False, method="first").astype(int)
    rec["as_of_date"] = as_of.date()
    return rec, as_of


def validation_tables(rec: pd.DataFrame):
    """Compare model-estimated lever effects with the simulator's true effects."""
    if "true_rating_gain" not in rec:
        return None, {}
    per = []
    for lever, g in rec.groupby("lever_id"):
        sp_r = spearmanr(g.expected_rating_gain, g.true_rating_gain).statistic if g.true_rating_gain.nunique() > 1 else np.nan
        sp_k = spearmanr(g.expected_relative_risk_reduction, g.true_relative_risk_reduction).statistic \
            if g.true_relative_risk_reduction.nunique() > 1 else np.nan
        per.append(dict(lever_id=lever, n_athletes=len(g),
                        mean_model_rating_gain=g.expected_rating_gain.mean(), mean_true_rating_gain=g.true_rating_gain.mean(),
                        spearman_rating=sp_r,
                        mean_model_rel_risk_reduction=g.expected_relative_risk_reduction.mean(),
                        mean_true_rel_risk_reduction=g.true_relative_risk_reduction.mean(), spearman_risk=sp_k))
    per = pd.DataFrame(per)

    # does the model pick the right lever for each athlete?
    piv_m = rec.pivot_table(index="athlete_id", columns="lever_id", values="expected_rating_gain")
    piv_t = rec.pivot_table(index="athlete_id", columns="lever_id", values="true_rating_gain")
    common = piv_m.columns.intersection(piv_t.columns)
    top_m, top_t = piv_m[common].idxmax(axis=1), piv_t[common].idxmax(axis=1)
    top3_hit = np.mean([top_t[a] in piv_m.loc[a, common].nlargest(3).index for a in piv_m.index])
    pooled = spearmanr(rec.expected_rating_gain, rec.true_rating_gain).statistic
    pooled_risk = spearmanr(rec.expected_relative_risk_reduction, rec.true_relative_risk_reduction).statistic
    # sign agreement where the true effect is meaningfully non-zero
    mask = rec.true_rating_gain.abs() > 0.01
    sign_agree = (np.sign(rec.expected_rating_gain[mask]) == np.sign(rec.true_rating_gain[mask])).mean()
    # a lazy baseline: tell everybody the most common true top lever. The engine must beat this to matter.
    lazy = float((top_t == top_t.value_counts().idxmax()).mean())
    summary = {
        "rec_rating_gain_spearman_pooled": pooled,
        "rec_risk_reduction_spearman_pooled": pooled_risk,
        "rec_top1_lever_match_rate": float((top_m == top_t).mean()),
        "baseline_always_same_lever_match_rate": lazy,
        # the stricter test: within ONE lever, does the model know which athletes benefit most?
        "rec_median_within_lever_spearman_rating": float(per.spearman_rating.median()),
        "rec_true_top_lever_in_model_top3": float(top3_hit),
        "rec_rating_gain_sign_agreement": float(sign_agree),
    }
    return per, summary


def main():
    bundle = joblib.load(cfg.ARTIFACT_DIR / "bundle.joblib")
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)
    with Warehouse() as wh:
        rec, as_of = build_recommendations(bundle, wh)
        per_lever, summary = validation_tables(rec)
        out = rec[["athlete_id", "as_of_date", "lever_id", "current_value", "target_value", "recommended_step",
                   "expected_rating_gain", "expected_risk_reduction_pp", "expected_relative_risk_reduction",
                   "priority_score", "priority_rank", "n_state_rows"]].copy()
        out["model_version"], out["scored_at"] = bundle["version"], now
        wh.write(out, "ml", "lever_recommendations")
        if per_lever is not None:
            per_lever["model_version"] = bundle["version"]
            wh.write(per_lever, "ml", "recommendation_validation")
            met = pd.DataFrame([("recommendation_engine", bundle["version"], k, float(v), "ground_truth", now)
                                for k, v in summary.items()],
                               columns=["model_name", "model_version", "metric", "value", "data_split", "trained_at"])
            existing = wh.query("select * from ml.model_metrics where model_name <> 'recommendation_engine'")
            wh.write(pd.concat([existing, met], ignore_index=True), "ml", "model_metrics")

    print(f"as-of {as_of.date()}: {len(out):,} athlete-lever recommendations for {out.athlete_id.nunique()} athletes")
    if per_lever is not None:
        print("\n=== RECOMMENDATION ENGINE vs GROUND TRUTH ===")
        print(pd.Series(summary).round(3).to_string())
        print("\nper lever (model vs true mean effect, rank correlation):")
        print(per_lever.drop(columns="model_version").round(3).to_string(index=False))
    top = out[out.priority_rank == 1].lever_id.value_counts()
    print("\n#1 lever across the squad:\n", top.to_string())


if __name__ == "__main__":
    main()
