-- Rows flagged label_is_complete must have a label, and the final 6 days of data must NOT be flagged
-- (their 7-day forward window is truncated, so a 0 there would be a false negative).
select athlete_id, date_day, label_is_complete, injury_next_7d
from {{ ref('mart_ml_injury_features') }}
where (label_is_complete = 1 and injury_next_7d is null)
   or (label_is_complete = 1 and date_day > {{ dbt.dateadd('day', -6, "cast('" ~ var('calendar_end') ~ "' as date)") }})
