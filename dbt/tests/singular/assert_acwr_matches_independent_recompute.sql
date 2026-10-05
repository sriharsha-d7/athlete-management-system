{#
  Independent re-derivation of ACWR from raw daily load using explicit LAG() terms instead of
  window frames. If someone changes a window frame in int_rolling_features and accidentally
  lets today's (or tomorrow's) load leak in, the two computations diverge and this test fails.
  Checked only where a full 28 prior days exist, so both methods are defined.
#}
with lagged as (
    select
        athlete_id,
        date_day,
        day_index,
        {% for k in range(1, 8) %}coalesce(lag(daily_load, {{ k }}) over (partition by athlete_id order by date_day), 0){{ " +" if not loop.last }} {% endfor %} as sum7,
        {% for k in range(1, 29) %}coalesce(lag(daily_load, {{ k }}) over (partition by athlete_id order by date_day), 0){{ " +" if not loop.last }} {% endfor %} as sum28
    from {{ ref('int_athlete_day_base') }}
),

recomputed as (
    select athlete_id, date_day, (sum7 / 7.0) / nullif(sum28 / 28.0, 0) as acwr_check
    from lagged
    where day_index > 28
)

select r.athlete_id, r.date_day, r.acwr_check, f.acwr
from recomputed r
join {{ ref('int_rolling_features') }} f using (athlete_id, date_day)
where abs(r.acwr_check - f.acwr) > 1e-6
