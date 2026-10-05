{#
  THE deliverable: for each athlete, what they need more of, ranked, with the expected payoff.
  Joins model output (ml.lever_recommendations) to the lever catalog (coaching wording, horizon)
  and the descriptive gap analysis (status vs target) so a coach reads one row per action.
#}
with latest_risk as (
    select athlete_id, injury_risk_7d, risk_band, predicted_rating, date_day as risk_date
    from {{ ref('mart_athlete_risk_daily') }}
    qualify row_number() over (partition by athlete_id order by date_day desc) = 1
)

select
    r.athlete_id,
    d.team_id,
    d.position_code,
    d.position_group,
    r.as_of_date,
    r.priority_rank,
    r.lever_id,
    l.label                                   as lever,
    l.category,
    l.unit,
    r.current_value,
    r.target_value,
    r.recommended_step,
    r.expected_rating_gain,
    r.expected_risk_reduction_pp,
    r.priority_score,
    g.status                                  as metric_status,
    g.percentile_in_position,
    l.horizon,
    l.coaching_action,
    lr.injury_risk_7d                         as current_injury_risk_7d,
    lr.risk_band                              as current_risk_band,
    lr.predicted_rating                       as current_expected_rating,
    c.risk_flags,
    r.model_version
from {{ source('ml', 'lever_recommendations') }} r
join {{ ref('seed_lever_catalog') }} l          on l.lever_id = r.lever_id
join {{ ref('dim_athlete') }} d                 on d.athlete_id = r.athlete_id
left join {{ ref('mart_athlete_gap_analysis') }} g
  on g.athlete_id = r.athlete_id and g.metric = l.metric
left join latest_risk lr                         on lr.athlete_id = r.athlete_id
left join {{ ref('mart_athlete_current_state') }} c on c.athlete_id = r.athlete_id
