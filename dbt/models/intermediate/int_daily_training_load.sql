select
    athlete_id,
    session_date                                                              as date_day,
    sum(srpe_load)                                                            as daily_load,
    sum(duration_min)                                                         as daily_minutes,
    count(*)                                                                  as sessions,
    sum(case when session_type = 'match' then duration_min else 0 end)        as match_minutes,
    sum(case when session_type = 'gym' then 1 else 0 end)                     as gym_sessions,
    sum(case when session_type = 'rehab' then 1 else 0 end)                   as rehab_sessions,
    sum(total_distance_m)                                                     as total_distance_m,
    sum(hsr_distance_m)                                                       as hsr_distance_m,
    sum(sprint_count)                                                         as sprint_count,
    max(max_speed_kmh)                                                        as max_speed_kmh
from {{ ref('stg_training_sessions') }}
group by 1, 2
