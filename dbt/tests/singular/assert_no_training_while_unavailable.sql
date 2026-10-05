-- Data-integrity rule: an injured athlete may only have rehab sessions.
-- Any other session on an unavailable day means the injury feed and the GPS feed disagree.
select athlete_id, date_day, sessions, rehab_sessions
from {{ ref('int_athlete_day_base') }}
where is_unavailable = 1
  and sessions - rehab_sessions > 0
