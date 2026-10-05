{#
  Dense spine: one row for EVERY calendar day an athlete is on the roster, whether or not
  anything was recorded. Rolling windows are only correct on a dense grain - on a sparse
  table "the last 7 rows" is not "the last 7 days".
#}
select
    a.athlete_id,
    c.date_day,
    row_number() over (partition by a.athlete_id order by c.date_day) as day_index
from {{ ref('stg_athletes') }} a
join {{ ref('int_calendar') }} c
  on c.date_day >= a.signed_date
