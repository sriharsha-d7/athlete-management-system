{#
  One row per athlete describing today's state (the latest date in the data), including
  transparent rule-based flags. Flags use the evidence thresholds in seed_evidence_targets
  so a coach can see WHY an athlete is amber before any model is involved.
#}
with as_of as (
    select max(date_day) as as_of_date from {{ ref('fct_athlete_day') }}
),

th as (
    select
        max(case when metric = 'sleep_hours_7d'  then target_min end) as sleep_min,
        max(case when metric = 'protein_gpkg_7d' then target_min end) as protein_min,
        max(case when metric = 'carbs_gpkg_7d'   then target_min end) as carbs_min,
        max(case when metric = 'hydration_l_7d'  then target_min end) as hydration_min,
        max(case when metric = 'acwr'            then target_max end) as acwr_max,
        max(case when metric = 'acwr'            then target_min end) as acwr_min,
        max(case when metric = 'nordic_n'        then target_min end) as nordic_min
    from {{ ref('seed_evidence_targets') }}
)

select
    f.athlete_id,
    f.team_id,
    f.position_group,
    f.position_code,
    f.date_day                                   as as_of_date,
    f.is_unavailable,
    f.readiness_score,
    f.acwr, f.acwr_zone, f.acute_load_7d, f.chronic_load_28d,
    f.sleep_hours_7d, f.protein_gpkg_7d, f.carbs_gpkg_7d, f.hydration_l_7d,
    f.hrv_z, f.cmj_cm, f.sprint_30m_s, f.yoyo_ir1_m, f.squat_1rm_rel, f.nordic_n,
    f.prior_injuries, f.days_since_return, f.rating_avg_5m,
    concat_ws(', ',
        case when f.acwr > 1.5                                  then 'ACWR spike (>1.5)' end,
        case when f.acwr > th.acwr_max and f.acwr <= 1.5        then 'ACWR elevated' end,
        case when f.acwr < th.acwr_min                           then 'Undertrained (ACWR low)' end,
        case when f.sleep_hours_7d < th.sleep_min               then 'Sleep debt' end,
        case when f.protein_gpkg_7d < th.protein_min            then 'Low protein' end,
        case when f.carbs_gpkg_7d < th.carbs_min                then 'Low carbohydrate' end,
        case when f.hydration_l_7d < th.hydration_min           then 'Low hydration' end,
        case when f.hrv_z < -1.0                                then 'HRV suppressed' end,
        case when f.nordic_n < th.nordic_min                    then 'Weak hamstrings' end,
        case when f.days_since_return between 1 and 14          then 'Recently returned from injury' end
    )                                            as risk_flags
from {{ ref('fct_athlete_day') }} f
join as_of a on f.date_day = a.as_of_date
cross join th
