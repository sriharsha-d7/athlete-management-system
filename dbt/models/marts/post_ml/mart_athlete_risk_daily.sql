{# Daily risk + expected-performance timeline per athlete: the fact the dashboard plots. #}
select
    f.athlete_id,
    f.date_day,
    f.team_id,
    f.position_group,
    f.acwr,
    f.acwr_zone,
    f.acute_load_7d,
    f.chronic_load_28d,
    f.sleep_hours_7d,
    f.hrv_z,
    f.readiness_score,
    f.is_unavailable,
    r.injury_risk_7d,
    r.risk_band,
    r.data_split,
    p.predicted_rating,
    f.injury_next_7d                    as actual_injury_next_7d,
    f.label_is_complete
from {{ ref('fct_athlete_day') }} f
join {{ source('ml', 'injury_risk_scores') }} r
  on r.athlete_id = f.athlete_id and r.date_day = f.date_day
left join {{ source('ml', 'performance_predictions') }} p
  on p.athlete_id = f.athlete_id and p.date_day = f.date_day
