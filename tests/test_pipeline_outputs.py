"""Integration checks against the built DuckDB warehouse. Skipped when the pipeline has not been run."""
import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "athlete.duckdb"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="run `make all` first")


@pytest.fixture(scope="module")
def wh():
    sys.path.insert(0, str(ROOT))
    from warehouse.conn import Warehouse
    w = Warehouse(read_only=True)
    yield w
    w.close()


def test_time_split_has_no_overlap(wh):
    from ml.data import add_split
    df = add_split(wh.query("select athlete_id, date_day from marts.mart_ml_injury_features"))
    rng = df.groupby("data_split").date_day.agg(["min", "max"])
    assert rng.loc["train", "max"] < rng.loc["val", "min"]
    assert rng.loc["val", "max"] < rng.loc["test", "min"]


def test_no_feature_peeks_at_labels(wh):
    from ml.data import feature_columns
    from ml.config import INJURY_META
    cols = feature_columns(wh.query("select * from marts.mart_ml_injury_features limit 5"), INJURY_META)
    banned = ("injury_next", "label", "future", "target")
    assert not [c for c in cols if any(b in c for b in banned)]


def test_models_beat_baselines(wh):
    m = wh.query("select * from ml.model_metrics")
    inj = m[m.model_name == "injury_risk_7d"].set_index("metric").value
    perf = m[m.model_name == "performance_rating"].set_index("metric").value
    assert inj.roc_auc > inj.baseline_acwr_only_roc_auc + 0.05
    assert inj.roc_auc > inj.baseline_logistic_roc_auc
    assert inj.lift_top5pct > 3
    assert perf.mae < perf.baseline_form_mae < perf.baseline_mean_mae


def test_risk_scores_are_probabilities_and_cover_features(wh):
    s = wh.query("select injury_risk_7d from ml.injury_risk_scores")
    n_feat = wh.query("select count(*) n from marts.mart_ml_injury_features").n.iloc[0]
    assert s.injury_risk_7d.between(0, 1).all()
    assert len(s) == n_feat


def test_recommendations_respect_shortfalls(wh):
    d = wh.query("""select r.athlete_id, r.lever_id, r.recommended_step, g.shortfall, l.step_type
                    from ml.lever_recommendations r
                    join reference.seed_lever_catalog l using (lever_id)
                    join marts.mart_athlete_gap_analysis g on g.athlete_id = r.athlete_id and g.metric = l.metric""")
    add = d[d.step_type == "add"]
    assert (add.recommended_step.abs() <= add.shortfall + 1e-6).all()   # never recommend more than the gap


def test_engine_recovers_ground_truth_better_than_chance(wh):
    m = wh.query("select * from ml.model_metrics where model_name = 'recommendation_engine'").set_index("metric").value
    assert m.rec_rating_gain_spearman_pooled > 0.5
    assert m.rec_median_within_lever_spearman_rating > 0.2
