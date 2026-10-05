{#
  Training table for the match-performance model.
  Grain  : athlete x match (>= 30 minutes so the rating is meaningful).
  Target : performance_rating. Features are the pre-match state (same definitions as the
           injury mart) so a counterfactual "what if he slept an hour more" can be scored
           by feeding modified feature rows to the same model.
#}
select
    f.athlete_id,
    f.date_day,
    f.position_group,
    m.performance_rating,
    m.minutes_played,
    m.opponent_strength,
    {{ ml_feature_columns('f') }}
from {{ ref('fct_athlete_day') }} f
join {{ ref('fct_match_performance') }} m
  on m.athlete_id = f.athlete_id and m.date_day = f.date_day
where f.has_full_history = 1
  and m.minutes_played >= 30
