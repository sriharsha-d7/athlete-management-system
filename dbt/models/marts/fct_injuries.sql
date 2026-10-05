{# Each injury enriched with the athlete's state on the morning it happened, for root-cause analysis. #}
select
    i.injury_id,
    i.athlete_id,
    d.team_id,
    d.position_group,
    i.injury_date,
    i.body_part,
    i.injury_type,
    i.mechanism,
    i.occurred_during,
    i.days_out,
    i.severity,
    f.acwr                  as acwr_at_onset,
    f.acwr_zone             as acwr_zone_at_onset,
    f.sleep_hours_7d        as sleep_hours_7d_at_onset,
    f.protein_gpkg_7d       as protein_gpkg_7d_at_onset,
    f.nordic_n              as nordic_n_at_onset,
    f.hrv_z                 as hrv_z_at_onset,
    f.prior_injuries        as prior_injuries_at_onset
from {{ ref('stg_injuries') }} i
join {{ ref('dim_athlete') }} d using (athlete_id)
left join {{ ref('fct_athlete_day') }} f
  on f.athlete_id = i.athlete_id and f.date_day = i.injury_date
