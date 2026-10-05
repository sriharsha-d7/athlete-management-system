{#
  Training table for the 7-day injury-risk model.
  Grain  : athlete x day.
  Rows   : only days the athlete was available to be injured, with a warm rolling window.
  Label  : injury_next_7d (NULL-safe: use label_is_complete = 1 for training/evaluation;
           the most recent 6 days are kept so they can still be SCORED).
#}
select
    f.athlete_id,
    f.date_day,
    f.position_group,
    f.injury_next_7d,
    f.label_is_complete,
    {{ ml_feature_columns('f') }}
from {{ ref('fct_athlete_day') }} f
where f.has_full_history = 1
  and f.is_unavailable = 0
