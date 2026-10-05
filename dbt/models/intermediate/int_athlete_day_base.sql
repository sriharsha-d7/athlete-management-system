{#
  One row per athlete per rostered day with every raw signal attached. No window maths here -
  that lives in int_rolling_features so the joins stay easy to audit.
#}
with spine as (
    select * from {{ ref('int_athlete_day_spine') }}
),

athletes as (
    select * from {{ ref('stg_athletes') }}
),

unavailable as (
    -- one row per day an athlete is out injured
    select s.athlete_id, s.date_day, 1 as is_unavailable
    from spine s
    join {{ ref('stg_injuries') }} i
      on i.athlete_id = s.athlete_id
     and s.date_day between i.unavailable_from and i.unavailable_to
    group by 1, 2
),

onsets as (
    select athlete_id, injury_date as date_day, 1 as injury_onset
    from {{ ref('stg_injuries') }}
    group by 1, 2
)

select
    s.athlete_id,
    s.date_day,
    s.day_index,
    a.team_id,
    a.position_code,
    a.birth_date,
    a.height_cm,
    a.weight_kg,

    -- load (zero-filled: no session means no load, not unknown load)
    coalesce(l.daily_load, 0)           as daily_load,
    coalesce(l.daily_minutes, 0)        as daily_minutes,
    coalesce(l.sessions, 0)             as sessions,
    coalesce(l.match_minutes, 0)        as match_minutes,
    coalesce(l.gym_sessions, 0)         as gym_sessions,
    coalesce(l.rehab_sessions, 0)       as rehab_sessions,
    coalesce(l.hsr_distance_m, 0)       as hsr_distance_m,
    coalesce(l.sprint_count, 0)         as sprint_count,
    l.max_speed_kmh,
    coalesce(l.total_distance_m, 0)     as total_distance_m,

    -- recovery (nullable: missing means not recorded)
    sl.sleep_hours,
    sl.sleep_efficiency_pct,
    w.hrv_rmssd_ms,
    w.resting_hr_bpm,
    w.soreness,
    w.fatigue,
    w.stress,
    w.mood,

    -- nutrition (nullable)
    n.protein_gpkg,
    n.carbs_gpkg,
    n.kcal_per_kg,
    n.hydration_l,

    -- physical tests (only populated on test days; carried forward downstream)
    t.cmj_cm,
    t.sprint_30m_s,
    t.yoyo_ir1_m,
    t.squat_1rm_rel,
    t.nordic_n,
    t.body_fat_pct,

    -- availability / medical
    coalesce(o.injury_onset, 0)         as injury_onset,
    coalesce(u.is_unavailable, 0)       as is_unavailable,

    -- schedule
    sch.is_match_day,
    sch.matches_next_7d
from spine s
join athletes a                                   on a.athlete_id = s.athlete_id
left join {{ ref('int_daily_training_load') }} l  on l.athlete_id = s.athlete_id and l.date_day = s.date_day
left join {{ ref('stg_sleep_daily') }} sl         on sl.athlete_id = s.athlete_id and sl.date_day = s.date_day
left join {{ ref('stg_wellness_daily') }} w       on w.athlete_id = s.athlete_id and w.date_day = s.date_day
left join {{ ref('stg_nutrition_daily') }} n      on n.athlete_id = s.athlete_id and n.date_day = s.date_day
left join {{ ref('stg_physical_tests') }} t       on t.athlete_id = s.athlete_id and t.date_day = s.date_day
left join onsets o                                on o.athlete_id = s.athlete_id and o.date_day = s.date_day
left join unavailable u                           on u.athlete_id = s.athlete_id and u.date_day = s.date_day
left join {{ ref('int_team_schedule') }} sch      on sch.team_id = a.team_id and sch.date_day = s.date_day
