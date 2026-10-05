{#
  Rolling-window feature engineering on the dense athlete-day grain.

  TIMING CONVENTION (the whole point of this model - see macros/windows.sql):
    Every feature is what a coach could know on the MORNING of date_day:
      * load + nutrition  : through yesterday          -> w_prior(n)
      * sleep + HRV + wellness : through this morning  -> w_incl(n)
      * physical tests    : latest test on or before today
      * labels            : look FORWARD from today    -> w_future(n)   (labels only!)
#}
with base as (
    select * from {{ ref('int_athlete_day_base') }}
),

with_form as (
    select
        b.*,
        f.form_rating_5m as form_incl
    from base b
    left join {{ ref('int_match_form') }} f
      on f.athlete_id = b.athlete_id and f.date_day = b.date_day
),

windows as (
    select
        *,

        -- ------------------------------------------------------------- load
        avg(daily_load) over ({{ w_prior(7) }})                                  as acute_load_7d,
        avg(daily_load) over ({{ w_prior(28) }})                                 as chronic_load_28d,
        stddev_samp(daily_load) over ({{ w_prior(7) }})                          as load_sd_7d,
        sum(daily_load) over ({{ w_prior(7) }})                                  as weekly_load_7d,
        sum(hsr_distance_m) over ({{ w_prior(7) }})                              as hsr_distance_7d,
        sum(sprint_count) over ({{ w_prior(7) }})                                as sprint_count_7d,
        max(max_speed_kmh) over ({{ w_prior(28) }})                              as max_speed_28d,
        sum(sessions) over ({{ w_prior(7) }})                                    as sessions_7d,
        sum(match_minutes) over ({{ w_prior(28) }})                              as match_minutes_28d,
        sum(gym_sessions) over ({{ w_prior(28) }})                               as gym_sessions_28d,

        -- ---------------------------------------------------------- recovery
        avg(sleep_hours) over ({{ w_incl(3) }})                                  as sleep_hours_3d,
        avg(sleep_hours) over ({{ w_incl(7) }})                                  as sleep_hours_7d,
        avg(sleep_hours) over ({{ w_incl(28) }})                                 as sleep_hours_28d,
        stddev_samp(sleep_hours) over ({{ w_incl(7) }})                          as sleep_variability_7d,
        avg(sleep_efficiency_pct) over ({{ w_incl(7) }})                         as sleep_efficiency_7d,
        avg(hrv_rmssd_ms) over ({{ w_prior(28) }})                               as hrv_baseline_28d,
        stddev_samp(hrv_rmssd_ms) over ({{ w_prior(28) }})                       as hrv_sd_28d,
        avg(resting_hr_bpm) over ({{ w_prior(28) }})                             as rhr_baseline_28d,
        stddev_samp(resting_hr_bpm) over ({{ w_prior(28) }})                     as rhr_sd_28d,
        avg(soreness) over ({{ w_incl(3) }})                                     as soreness_3d,
        avg(fatigue) over ({{ w_incl(3) }})                                      as fatigue_3d,
        avg(stress) over ({{ w_incl(7) }})                                       as stress_7d,
        avg(mood) over ({{ w_incl(7) }})                                         as mood_7d,
        count(hrv_rmssd_ms) over ({{ w_incl(7) }}) / 7.0                         as wellness_log_rate_7d,

        -- --------------------------------------------------------- nutrition
        avg(protein_gpkg) over ({{ w_prior(7) }})                                as protein_gpkg_7d,
        avg(carbs_gpkg) over ({{ w_prior(3) }})                                  as carbs_gpkg_3d,
        avg(carbs_gpkg) over ({{ w_prior(7) }})                                  as carbs_gpkg_7d,
        avg(kcal_per_kg) over ({{ w_prior(7) }})                                 as kcal_per_kg_7d,
        avg(hydration_l) over ({{ w_prior(3) }})                                 as hydration_l_3d,
        avg(hydration_l) over ({{ w_prior(7) }})                                 as hydration_l_7d,
        count(protein_gpkg) over ({{ w_prior(7) }}) / 7.0                        as nutrition_log_rate_7d,

        -- ---------------------------------- physical tests (carry last value)
        last_value(cmj_cm ignore nulls) over ({{ w_all_incl() }})                as cmj_latest,
        last_value(sprint_30m_s ignore nulls) over ({{ w_all_incl() }})          as sprint_30m_latest,
        last_value(yoyo_ir1_m ignore nulls) over ({{ w_all_incl() }})            as yoyo_latest,
        last_value(squat_1rm_rel ignore nulls) over ({{ w_all_incl() }})         as squat_latest,
        last_value(nordic_n ignore nulls) over ({{ w_all_incl() }})              as nordic_latest,
        last_value(body_fat_pct ignore nulls) over ({{ w_all_incl() }})          as body_fat_latest,
        max(case when cmj_cm is not null then date_day end) over ({{ w_all_incl() }}) as last_test_date,

        -- ----------------------------------------------- medical / history
        coalesce(sum(injury_onset) over ({{ w_all_prior() }}), 0)                as prior_injuries,
        max(case when injury_onset = 1 then date_day end) over ({{ w_all_prior() }})    as last_injury_date,
        max(case when is_unavailable = 1 then date_day end) over ({{ w_all_prior() }})  as last_unavailable_date,
        last_value(form_incl ignore nulls) over ({{ w_all_prior() }})            as rating_avg_5m,

        -- ------------------------------------------------ labels (FUTURE!)
        max(injury_onset) over ({{ w_future(7) }})                               as injury_next_7d,
        count(*) over ({{ w_future(7) }})                                        as future_days_available
    from with_form
)

select
    athlete_id,
    date_day,
    day_index,
    team_id,
    position_code,
    height_cm,
    weight_kg,
    {{ dbt.datediff('birth_date', 'date_day', 'day') }} / 365.25                 as age_years,

    -- raw daily values worth keeping for dashboards
    daily_load, daily_minutes, sessions, match_minutes, hrv_rmssd_ms, resting_hr_bpm,
    sleep_hours, soreness, fatigue, stress, mood, is_match_day, matches_next_7d,
    is_unavailable, injury_onset,

    -- load
    acute_load_7d,
    chronic_load_28d,
    acute_load_7d / nullif(chronic_load_28d, 0)                                  as acwr,
    acute_load_7d / nullif(load_sd_7d, 0)                                        as load_monotony_7d,
    weekly_load_7d * (acute_load_7d / nullif(load_sd_7d, 0))                     as load_strain_7d,
    hsr_distance_7d, sprint_count_7d, max_speed_28d, sessions_7d, match_minutes_28d, gym_sessions_28d,

    -- recovery
    sleep_hours_3d, sleep_hours_7d, sleep_hours_28d, sleep_variability_7d, sleep_efficiency_7d,
    (hrv_rmssd_ms - hrv_baseline_28d) / nullif(hrv_sd_28d, 0)                    as hrv_z,
    (resting_hr_bpm - rhr_baseline_28d) / nullif(rhr_sd_28d, 0)                  as rhr_z,
    soreness_3d, fatigue_3d, stress_7d, mood_7d, wellness_log_rate_7d,

    -- nutrition
    protein_gpkg_7d, carbs_gpkg_3d, carbs_gpkg_7d, kcal_per_kg_7d, hydration_l_3d, hydration_l_7d,
    nutrition_log_rate_7d,

    -- physical
    cmj_latest                                                                   as cmj_cm,
    sprint_30m_latest                                                            as sprint_30m_s,
    yoyo_latest                                                                  as yoyo_ir1_m,
    squat_latest                                                                 as squat_1rm_rel,
    nordic_latest                                                                as nordic_n,
    body_fat_latest                                                              as body_fat_pct,
    {{ dbt.datediff('last_test_date', 'date_day', 'day') }}                      as days_since_test,

    -- history
    prior_injuries,
    {{ dbt.datediff('last_injury_date', 'date_day', 'day') }}                    as days_since_last_injury,
    {{ dbt.datediff('last_unavailable_date', 'date_day', 'day') }}               as days_since_return,
    rating_avg_5m,

    -- trust flags + labels
    case when day_index > {{ var('min_history_days') }} then 1 else 0 end        as has_full_history,
    case when future_days_available = {{ var('label_horizon_days') }} then 1 else 0 end as label_is_complete,
    injury_next_7d
from windows
