"""
Athlete Performance Intelligence: coach-facing dashboard over the dbt marts.

    streamlit run app/streamlit_app.py                       # local DuckDB
    WAREHOUSE=snowflake streamlit run app/streamlit_app.py   # Snowflake (read-only role recommended)

The app does no modelling: every number comes from a dbt mart or an ml.* table, so what a coach sees is
exactly what the tests covered.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from warehouse.conn import Warehouse  # noqa: E402

# Palette: validated categorical slots 1-2 + reserved status colours (always paired with a text label)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND_COLORS = {"Low": "#0ca30c", "Moderate": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
BANDS = list(BAND_COLORS)

st.set_page_config(page_title="Athlete Performance Intelligence", layout="wide", page_icon=":material/sports_soccer:")


@st.cache_resource
def _wh():
    return Warehouse(read_only=True)


@st.cache_data(ttl=600, show_spinner=False)
def q(sql: str) -> pd.DataFrame:
    return _wh().query(sql)


def style(fig: go.Figure, height=320, legend=True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=13),
        title=dict(font=dict(size=15, color=INK), x=0), hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(color=INK2)))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED), zeroline=False)
    return fig


# ------------------------------------------------------------------------------------------------ data
try:
    plan = q("select * from marts.mart_athlete_action_plan")
    squad = q("select * from marts.mart_squad_overview order by team_id")
    state = q("select * from marts.mart_athlete_current_state")
except Exception as e:  # pragma: no cover
    st.error("Marts not found. Run `make all` first (generate data, load, dbt build, train, dbt build).")
    st.exception(e)
    st.stop()

risk_now = q("""select r.athlete_id, r.team_id, r.position_group, r.injury_risk_7d, r.risk_band, r.predicted_rating,
                       r.readiness_score, r.is_unavailable, r.acwr, r.date_day
                from marts.mart_athlete_risk_daily r
                where r.date_day = (select max(date_day) from marts.mart_athlete_risk_daily)""")
as_of = pd.to_datetime(risk_now.date_day.iloc[0]).date()

st.title("Athlete Performance Intelligence")
st.caption(f"Synthetic league data, as of {as_of}. Every figure is read from dbt marts; models are scored in `ml.*`.")

tab_squad, tab_athlete, tab_levers, tab_models = st.tabs(["Squad", "Athlete", "What the squad needs", "Model quality"])

# ------------------------------------------------------------------------------------------------ squad
with tab_squad:
    n = len(risk_now)
    hi = int(risk_now.risk_band.isin(["High", "Critical"]).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Athletes", n)
    c2.metric("Injured now", int(risk_now.is_unavailable.sum()))
    c3.metric("High / critical injury risk", hi, help="Calibrated probability of a time-loss injury in the next 7 days >= 10%")
    c4.metric("Average readiness", f"{risk_now.readiness_score.mean():.0f} / 100",
              help="Heuristic from HRV, sleep, soreness and fatigue. Not a clinical measure.")

    bands = (risk_now[risk_now.is_unavailable == 0].groupby(["team_id", "risk_band"]).size()
             .unstack(fill_value=0).reindex(columns=BANDS, fill_value=0))
    fig = go.Figure()
    for b in BANDS:
        fig.add_bar(x=bands.index, y=bands[b], name=b, marker=dict(color=BAND_COLORS[b], line=dict(color="#fcfcfb", width=2)),
                    hovertemplate="%{x}: %{y} athletes<extra>" + b + "</extra>")
    fig.update_layout(barmode="stack", title="Available athletes by injury-risk band (next 7 days)")
    st.plotly_chart(style(fig, 340), width="stretch")

    st.subheader("Athletes to watch")
    watch = (risk_now[risk_now.risk_band.isin(["High", "Critical"]) & (risk_now.is_unavailable == 0)]
             .merge(state[["athlete_id", "risk_flags"]], on="athlete_id")
             .sort_values("injury_risk_7d", ascending=False))
    st.dataframe(
        watch.assign(injury_risk_7d=(watch.injury_risk_7d * 100).round(1))[
            ["athlete_id", "team_id", "position_group", "risk_band", "injury_risk_7d", "acwr", "risk_flags"]]
        .rename(columns={"injury_risk_7d": "7-day risk %", "risk_flags": "why (rule-based flags)"}),
        hide_index=True, width="stretch")
    st.caption("Risk comes from the calibrated gradient-boosting model; the flags are simple evidence-threshold rules shown so a "
               "coach can see the likely reasons without trusting the model blindly.")

# ------------------------------------------------------------------------------------------------ athlete
with tab_athlete:
    order = risk_now.sort_values("injury_risk_7d", ascending=False)
    pick = st.selectbox("Athlete (sorted by current injury risk)", order.athlete_id,
                        format_func=lambda a: f"{a}  |  {order.set_index('athlete_id').loc[a, 'position_group']}  |  "
                                              f"{order.set_index('athlete_id').loc[a, 'risk_band']} risk")
    me = risk_now[risk_now.athlete_id == pick].iloc[0]
    meta = state[state.athlete_id == pick].iloc[0]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("7-day injury risk", f"{me.injury_risk_7d * 100:.1f}%", me.risk_band, delta_color="off")
    a2.metric("Expected match rating", f"{me.predicted_rating:.2f}")
    a3.metric("Readiness", f"{me.readiness_score:.0f} / 100")
    a4.metric("ACWR", f"{me.acwr:.2f}" if pd.notna(me.acwr) else "n/a", meta.acwr_zone)
    if meta.risk_flags:
        st.warning(f"**Flags:** {meta.risk_flags}")
    if me.is_unavailable:
        st.info("Currently unavailable (injured). The plan below applies to the last days he was available.")

    hist = q(f"""select date_day, injury_risk_7d, acwr, sleep_hours_7d, predicted_rating, is_unavailable
                 from marts.mart_athlete_risk_daily where athlete_id = '{pick}' order by date_day""")
    hist["date_day"] = pd.to_datetime(hist.date_day)
    hist = hist[hist.date_day >= hist.date_day.max() - pd.Timedelta(days=240)]

    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure(go.Scatter(x=hist.date_day, y=hist.injury_risk_7d * 100, mode="lines", line=dict(color=BLUE, width=2),
                                 name="7-day injury risk", hovertemplate="%{y:.1f}%<extra>7-day injury risk</extra>"))
        for cut, name in ((10, "High"), (20, "Critical")):
            f.add_hline(y=cut, line=dict(color=BAND_COLORS[name], width=1, dash="dot"),
                        annotation_text=f"{name} >= {cut}%", annotation_font_color=INK2, annotation_position="top left")
        f.update_yaxes(title="%")
        f.update_layout(title="Injury risk, next 7 days")
        st.plotly_chart(style(f, legend=False), width="stretch")
    with c2:
        f = go.Figure()
        f.add_hrect(y0=0.8, y1=1.3, fillcolor="#0ca30c", opacity=0.10, line_width=0)
        f.add_trace(go.Scatter(x=hist.date_day, y=hist.acwr, mode="lines", line=dict(color=ORANGE, width=2), name="ACWR",
                               hovertemplate="%{y:.2f}<extra>ACWR</extra>"))
        f.add_annotation(xref="paper", x=0.01, yref="y", y=1.05, text="sweet spot 0.8-1.3", showarrow=False,
                         font=dict(color=INK2, size=12), xanchor="left")
        f.update_layout(title="Acute:chronic workload ratio")
        st.plotly_chart(style(f, legend=False), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        f = go.Figure(go.Scatter(x=hist.date_day, y=hist.sleep_hours_7d, mode="lines", line=dict(color=BLUE, width=2),
                                 name="Sleep, 7-day average", hovertemplate="%{y:.1f} h<extra>7-day sleep</extra>"))
        f.add_hline(y=8.5, line=dict(color=AXIS, width=1, dash="dot"), annotation_text="target 8.5 h",
                    annotation_font_color=INK2, annotation_position="top left")
        f.update_layout(title="Sleep (hours per night)")
        st.plotly_chart(style(f, legend=False), width="stretch")
    with c4:
        gap = q(f"select * from marts.mart_athlete_gap_analysis where athlete_id = '{pick}' order by shortfall_pct desc")
        gap = gap.assign(label=gap.metric.str.replace("_", " "))
        f = go.Figure(go.Bar(
            y=gap.label, x=(gap.shortfall_pct * 100).round(1), orientation="h", marker=dict(color=BLUE),
            customdata=gap[["current_value", "target_value", "status"]].round(2).values,
            hovertemplate="%{y}: %{x}% short of target<br>now %{customdata[0]} vs target %{customdata[1]} "
                          "(%{customdata[2]})<extra></extra>"))
        f.update_layout(title="Shortfall vs target (% of target)", hovermode="closest", yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(f, legend=False), width="stretch")

    st.subheader("Action plan: what he needs more of")
    mine = plan[plan.athlete_id == pick].sort_values("priority_rank")
    if mine.empty:
        st.info("No recommendations (no recent available days to score).")
    else:
        st.dataframe(
            mine.assign(rating_gain=mine.expected_rating_gain.round(3), risk_pp=mine.expected_risk_reduction_pp.round(2),
                        now=mine.current_value.round(2), target=mine.target_value.round(2), step=mine.recommended_step.round(2))[
                ["priority_rank", "lever", "now", "target", "step", "unit", "rating_gain", "risk_pp", "horizon", "coaching_action"]]
            .rename(columns={"priority_rank": "#", "rating_gain": "+ match rating", "risk_pp": "- injury risk (pp)"}),
            hide_index=True, width="stretch")
        st.caption("Effects are what the models predict if only that lever moves, capped at the athlete's real shortfall. "
                   "They are estimates from observational data, not a trial.")

# ------------------------------------------------------------------------------------------------ squad needs
with tab_levers:
    team = st.selectbox("Team", ["All"] + sorted(plan.team_id.unique()))
    p = plan if team == "All" else plan[plan.team_id == team]
    top = (p[p.priority_rank <= 3].groupby("lever").athletes.sum() if "athletes" in p else
           p[p.priority_rank <= 3].groupby("lever").athlete_id.nunique()).sort_values()
    f = go.Figure(go.Bar(y=top.index, x=top.values, orientation="h", marker=dict(color=BLUE),
                         hovertemplate="%{y}: %{x} athletes<extra></extra>"))
    f.update_layout(title="Athletes for whom this is a top-3 lever", hovermode="closest")
    st.plotly_chart(style(f, 380, legend=False), width="stretch")

    agg = (p.groupby("lever").agg(athletes=("athlete_id", "nunique"), avg_rating_gain=("expected_rating_gain", "mean"),
                                  avg_risk_reduction_pp=("expected_risk_reduction_pp", "mean"),
                                  horizon=("horizon", "first")).round(3).sort_values("avg_rating_gain", ascending=False))
    st.dataframe(agg, width="stretch")
    st.caption("Averages are over athletes who actually have a shortfall on that lever.")

# ------------------------------------------------------------------------------------------------ models
with tab_models:
    m = q("select * from ml.model_metrics")
    st.subheader("Held-out performance (Apr-Sep 2026, never used for training or tuning)")
    c1, c2 = st.columns(2)
    inj = m[m.model_name == "injury_risk_7d"].set_index("metric").value
    perf = m[m.model_name == "performance_rating"].set_index("metric").value
    with c1:
        st.markdown("**Injury risk (next 7 days)**")
        st.dataframe(pd.DataFrame({
            "Model": [inj.roc_auc, inj.pr_auc, inj.lift_top5pct],
            "Baseline: logistic (7 features)": [inj.baseline_logistic_roc_auc, inj.baseline_logistic_pr_auc, None],
            "Baseline: ACWR rule only": [inj.baseline_acwr_only_roc_auc, None, None],
        }, index=["ROC-AUC", "PR-AUC", "Lift in top 5% of risk"]).round(3), width="stretch")
        st.caption(f"Weekly base rate on the test window: {inj.base_rate * 100:.1f}%.")
    with c2:
        st.markdown("**Match rating**")
        st.dataframe(pd.DataFrame({
            "Model": [perf.mae, perf.r2],
            "Baseline: ridge regression": [perf.baseline_ridge_mae, perf.baseline_ridge_r2],
            "Baseline: recent form": [perf.baseline_form_mae, perf.baseline_form_r2],
            "Baseline: league mean": [perf.baseline_mean_mae, None],
        }, index=["MAE (rating points)", "R-squared"]).round(3), width="stretch")

    cal = q("select * from ml.calibration_curve order by bin")
    imp = q("select * from ml.feature_importance")
    c1, c2 = st.columns(2)
    with c1:
        f = go.Figure()
        top_v = max(cal.mean_predicted.max(), cal.observed_rate.max()) * 100 * 1.05
        f.add_trace(go.Scatter(x=[0, top_v], y=[0, top_v], mode="lines", line=dict(color=AXIS, width=1, dash="dot"),
                               name="Perfect calibration", hoverinfo="skip"))
        f.add_trace(go.Scatter(x=cal.mean_predicted * 100, y=cal.observed_rate * 100, mode="lines+markers", name="Model",
                               line=dict(color=BLUE, width=2), marker=dict(size=8, color=BLUE, line=dict(color="#fcfcfb", width=2)),
                               hovertemplate="predicted %{x:.1f}% -> observed %{y:.1f}%<extra></extra>"))
        f.update_layout(title="Calibration by risk decile (test)", hovermode="closest")
        f.update_xaxes(title="Predicted 7-day risk, %")
        f.update_yaxes(title="Observed injury rate, %")
        st.plotly_chart(style(f, 340), width="stretch")
        st.caption("Close on average, but compressed at the extremes: the safest deciles are over-predicted and the riskiest "
                   "decile under-predicted. The test window also has a higher base rate than the window used to calibrate.")
    with c2:
        which = st.radio("Drivers of", ["injury_risk_7d", "performance_rating"], horizontal=True,
                         format_func=lambda s: "injury risk" if s.startswith("inj") else "match rating")
        d = imp[imp.model_name == which].nlargest(12, "importance_mean").sort_values("importance_mean")
        f = go.Figure(go.Bar(y=d.feature, x=d.importance_mean, orientation="h", marker=dict(color=BLUE),
                             hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        f.update_layout(title="Permutation importance (test)", hovermode="closest")
        st.plotly_chart(style(f, 340, legend=False), width="stretch")

    st.subheader("Does the recommendation engine find the true effects?")
    st.caption("Because the data is simulated, the real effect of every lever is known. These compare the engine's estimates "
               "with that ground truth.")
    rv = q("select * from ml.recommendation_validation order by lever_id")
    rec_m = m[m.model_name == "recommendation_engine"].set_index("metric").value
    st.dataframe(rv.drop(columns=["model_version"]).round(3), hide_index=True, width="stretch")
    if len(rec_m):
        st.dataframe(rec_m.round(3).rename("value"), width="stretch")
