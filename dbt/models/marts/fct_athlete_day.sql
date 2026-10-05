{#
  The wide daily fact: one row per athlete per rostered day with load, recovery, nutrition,
  physical and availability signals plus two heuristic business metrics.
  readiness_score is a transparent 0-100 heuristic for dashboards, NOT a clinical measure.
#}
select
    f.*,
    d.position_group,
    least(100, greatest(0,
        60
        + 10  * coalesce(least(greatest(f.hrv_z, -3), 3), 0)
        + 6   * (coalesce(f.sleep_hours_3d, 7.5) - 7.5)
        - 2.5 * (coalesce(f.soreness_3d, 3) - 3)
        - 2.0 * (coalesce(f.fatigue_3d, 3) - 3)
    ))                                                                       as readiness_score,
    case when f.acwr is null            then null
         when f.acwr < 0.8              then 'undertrained'
         when f.acwr <= 1.3             then 'sweet_spot'
         when f.acwr <= 1.5             then 'caution'
         else 'danger' end                                                   as acwr_zone
from {{ ref('int_rolling_features') }} f
join {{ ref('dim_athlete') }} d using (athlete_id)
