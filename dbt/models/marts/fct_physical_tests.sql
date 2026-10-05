select
    t.test_id,
    t.athlete_id,
    d.team_id,
    d.position_group,
    t.date_day,
    t.cmj_cm,
    t.sprint_30m_s,
    t.yoyo_ir1_m,
    t.squat_1rm_rel,
    t.nordic_n,
    t.body_fat_pct,
    t.cmj_cm        - lag(t.cmj_cm)        over (partition by t.athlete_id order by t.date_day) as cmj_change,
    t.sprint_30m_s  - lag(t.sprint_30m_s)  over (partition by t.athlete_id order by t.date_day) as sprint_change,
    t.yoyo_ir1_m    - lag(t.yoyo_ir1_m)    over (partition by t.athlete_id order by t.date_day) as yoyo_change,
    t.squat_1rm_rel - lag(t.squat_1rm_rel) over (partition by t.athlete_id order by t.date_day) as squat_change,
    t.nordic_n      - lag(t.nordic_n)      over (partition by t.athlete_id order by t.date_day) as nordic_change
from {{ ref('stg_physical_tests') }} t
join {{ ref('dim_athlete') }} d using (athlete_id)
