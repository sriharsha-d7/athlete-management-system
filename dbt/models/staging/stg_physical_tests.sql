select
    test_id,
    athlete_id,
    test_date            as date_day,
    cmj_cm,
    sprint_30m_s,
    yoyo_ir1_m,
    squat_1rm_rel,
    nordic_n,
    body_fat_pct
from {{ source('raw', 'physical_tests') }}
qualify row_number() over (partition by test_id order by _loaded_at desc) = 1
