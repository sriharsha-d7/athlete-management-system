#!/usr/bin/env python
"""
Train + evaluate the two models, score every athlete-day, and write everything back to the warehouse.

  1. Injury model  : P(time-loss injury in next 7 days)   HistGradientBoostingClassifier + Platt calibration
  2. Performance   : expected match rating                  HistGradientBoostingRegressor

Both use a strict time split (train <= 2025-12, val Jan-Mar 2026, test Apr-Sep 2026), monotonic
constraints from domain priors, and are compared against simple baselines so "the model works" is
a measured claim rather than an assumption.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ml import config as cfg  # noqa: E402
from ml.calibration import PlattCalibrator  # noqa: E402
from ml.data import (INJURY_META, PERF_META, feature_columns, load_injury_frame,  # noqa: E402
                     load_perf_frame, make_X)
from warehouse.conn import Warehouse  # noqa: E402


# ----------------------------------------------------------------------------- helpers
def topk(y, p, frac):
    n = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:n]
    prec = y[idx].mean()
    return dict(precision=prec, lift=prec / y.mean(), recall=y[idx].sum() / max(y.sum(), 1))


def mono_dict(cols, spec):
    return {c: spec.get(c, 0) for c in cols}


def bin_calibration(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    g = pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin").agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()
    return g


# ----------------------------------------------------------------------------- injury model
def train_injury(wh):
    df = load_injury_frame(wh)
    feats = feature_columns(df, INJURY_META)
    lab = df[df.label_is_complete == 1]
    tr, va, te = (lab[lab.data_split == s] for s in ("train", "val", "test"))
    Xtr, Xva, Xte = (make_X(d, feats) for d in (tr, va, te))
    ytr, yva, yte = (d.injury_next_7d.values.astype(int) for d in (tr, va, te))
    print(f"[injury] rows train/val/test = {len(tr):,}/{len(va):,}/{len(te):,}  "
          f"base rate {ytr.mean():.3%}/{yva.mean():.3%}/{yte.mean():.3%}")

    mono = mono_dict(Xtr.columns, cfg.MONOTONE_INJURY)
    base_kw = dict(learning_rate=0.05, max_depth=5, max_leaf_nodes=24, min_samples_leaf=150,
                   l2_regularization=2.0, monotonic_cst=mono, random_state=cfg.SEED, early_stopping=False)
    probe = HistGradientBoostingClassifier(max_iter=600, **base_kw).fit(Xtr, ytr)
    val_ll = [log_loss(yva, p[:, 1]) for p in probe.staged_predict_proba(Xva)]
    best_iter = int(np.argmin(val_ll)) + 1
    model = HistGradientBoostingClassifier(max_iter=best_iter, **base_kw).fit(Xtr, ytr)
    print(f"[injury] best iteration on validation log-loss: {best_iter}")

    iso = PlattCalibrator().fit(model.predict_proba(Xva)[:, 1], yva)   # smooth, strictly monotone
    cal = lambda X: iso.predict(model.predict_proba(X)[:, 1])  # noqa: E731

    p_te = cal(Xte)
    metrics = {
        "roc_auc": roc_auc_score(yte, p_te), "pr_auc": average_precision_score(yte, p_te),
        "brier": brier_score_loss(yte, p_te), "base_rate": yte.mean(),
    }
    for f in (0.05, 0.10):
        for k, v in topk(yte, p_te, f).items():
            metrics[f"{k}_top{int(f*100)}pct"] = v

    # baselines on the same test rows
    metrics["baseline_acwr_only_roc_auc"] = roc_auc_score(yte, te["acwr"].fillna(1.0).values)
    base_cols = ["acwr", "prior_injuries", "age_years", "sleep_hours_7d", "protein_gpkg_7d", "nordic_n", "hrv_z"]
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))
    lr.fit(tr[base_cols], ytr)
    metrics["baseline_logistic_roc_auc"] = roc_auc_score(yte, lr.predict_proba(te[base_cols])[:, 1])
    metrics["baseline_logistic_pr_auc"] = average_precision_score(yte, lr.predict_proba(te[base_cols])[:, 1])

    # permutation importance on a test sample (AUC drop when a feature is shuffled)
    samp = np.random.default_rng(cfg.SEED).choice(len(Xte), size=min(20000, len(Xte)), replace=False)
    pi = permutation_importance(model, Xte.iloc[samp], yte[samp], scoring="roc_auc", n_repeats=3,
                                random_state=cfg.SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": Xte.columns, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)

    calib = bin_calibration(yte, p_te)
    # score all available athlete-days (including the last 6 days that have no complete label yet)
    Xall = make_X(df, feats)
    p_all = cal(Xall)
    scores = pd.DataFrame({
        "athlete_id": df.athlete_id.values, "date_day": df.date_day.dt.date.values,
        "injury_risk_7d": p_all, "risk_band": [cfg.risk_band(x) for x in p_all],
        "data_split": np.where(df.label_is_complete.values == 1, df.data_split.values, "scoring_only"),
    })
    bundle = dict(model=model, calibrator=iso, features=feats, columns=list(Xtr.columns), best_iter=best_iter)
    return bundle, metrics, imp, calib, scores


# ----------------------------------------------------------------------------- performance model
def train_perf(wh, injury_frame_cols):
    df = load_perf_frame(wh)
    feats = feature_columns(df, PERF_META)
    tr, va, te = (df[df.data_split == s] for s in ("train", "val", "test"))
    Xtr, Xva, Xte = (make_X(d, feats) for d in (tr, va, te))
    ytr, yva, yte = (d.performance_rating.values for d in (tr, va, te))
    print(f"[perf] matches train/val/test = {len(tr):,}/{len(va):,}/{len(te):,}")

    mono = mono_dict(Xtr.columns, cfg.MONOTONE_PERF)
    base_kw = dict(learning_rate=0.04, max_depth=3, max_leaf_nodes=8, min_samples_leaf=40,
                   l2_regularization=3.0, monotonic_cst=mono, random_state=cfg.SEED, early_stopping=False)
    probe = HistGradientBoostingRegressor(max_iter=600, **base_kw).fit(Xtr, ytr)
    val_mse = [np.mean((yva - p) ** 2) for p in probe.staged_predict(Xva)]
    best_iter = int(np.argmin(val_mse)) + 1
    model = HistGradientBoostingRegressor(max_iter=best_iter, **base_kw).fit(Xtr, ytr)
    print(f"[perf] best iteration on validation MSE: {best_iter}")

    p_te = model.predict(Xte)
    form_fill = tr.rating_avg_5m.mean()
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=50.0)).fit(Xtr, ytr)
    metrics = {
        "mae": mean_absolute_error(yte, p_te), "rmse": float(np.sqrt(np.mean((yte - p_te) ** 2))),
        "r2": r2_score(yte, p_te),
        "baseline_mean_mae": mean_absolute_error(yte, np.full_like(yte, ytr.mean())),
        "baseline_form_mae": mean_absolute_error(yte, te.rating_avg_5m.fillna(form_fill).values),
        "baseline_form_r2": r2_score(yte, te.rating_avg_5m.fillna(form_fill).values),
        "baseline_ridge_mae": mean_absolute_error(yte, ridge.predict(Xte)),
        "baseline_ridge_r2": r2_score(yte, ridge.predict(Xte)),
        "rating_sd": float(yte.std()),
    }
    pi = permutation_importance(model, Xte, yte, scoring="neg_mean_absolute_error", n_repeats=5,
                                random_state=cfg.SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": Xte.columns, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    bundle = dict(model=model, features=feats, columns=list(Xtr.columns), best_iter=best_iter)
    return bundle, metrics, imp


# ----------------------------------------------------------------------------- main
def main():
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)
    with Warehouse() as wh:
        inj_bundle, inj_m, inj_imp, calib, scores = train_injury(wh)
        perf_bundle, perf_m, perf_imp = train_perf(wh, inj_bundle["features"])

        # performance prediction for every athlete-day ("expected rating if he played today")
        df_all = load_injury_frame(wh)
        assert set(perf_bundle["features"]) == set(inj_bundle["features"]), "feature sets must match"
        Xall = make_X(df_all, perf_bundle["features"])[perf_bundle["columns"]]
        pred = pd.DataFrame({
            "athlete_id": df_all.athlete_id.values, "date_day": df_all.date_day.dt.date.values,
            "predicted_rating": perf_bundle["model"].predict(Xall),
            "data_split": df_all.data_split.values})

        version = "v" + now.strftime("%Y%m%d") + "-" + hashlib.md5(
            f"{len(df_all)}{inj_bundle['best_iter']}{perf_bundle['best_iter']}".encode()).hexdigest()[:6]
        for t in (scores, pred):
            t["model_version"], t["scored_at"] = version, now

        met_rows = []
        for model_name, m in (("injury_risk_7d", inj_m), ("performance_rating", perf_m)):
            for k, v in m.items():
                met_rows.append((model_name, version, k, float(v), "test", now))
        metrics_df = pd.DataFrame(met_rows, columns=["model_name", "model_version", "metric", "value", "data_split", "trained_at"])
        imp_df = pd.concat([inj_imp.assign(model_name="injury_risk_7d"), perf_imp.assign(model_name="performance_rating")])
        imp_df["model_version"] = version
        calib["model_name"], calib["model_version"] = "injury_risk_7d", version

        wh.write(scores, "ml", "injury_risk_scores")
        wh.write(pred, "ml", "performance_predictions")
        wh.write(metrics_df, "ml", "model_metrics")
        wh.write(imp_df, "ml", "feature_importance")
        wh.write(calib, "ml", "calibration_curve")

    cfg.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"injury": inj_bundle, "performance": perf_bundle, "version": version},
                cfg.ARTIFACT_DIR / "bundle.joblib")
    print("\n=== INJURY MODEL (test: Apr-Sep 2026) ===")
    print(json.dumps({k: round(float(v), 4) for k, v in inj_m.items()}, indent=2))
    print("\n=== PERFORMANCE MODEL (test) ===")
    print(json.dumps({k: round(float(v), 4) for k, v in perf_m.items()}, indent=2))
    print("\ntop injury drivers:\n", inj_imp.head(8).to_string(index=False))
    print("\ntop performance drivers:\n", perf_imp.head(8).to_string(index=False))
    print(f"\nmodel version: {version}")


if __name__ == "__main__":
    main()
