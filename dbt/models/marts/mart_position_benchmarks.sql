{#
  "What does a high performer in this position look like?"
  Elite = player-matches whose rating is in the top quartile of their position group.
  For each controllable / testable metric we publish the median and IQR of the athlete's
  *pre-match state* in those matches, next to the position-wide median for contrast.
  Long format: one row per position_group x metric.
#}
{% set metrics = [
    'sleep_hours_7d', 'protein_gpkg_7d', 'carbs_gpkg_7d', 'hydration_l_7d', 'acwr', 'chronic_load_28d',
    'cmj_cm', 'sprint_30m_s', 'yoyo_ir1_m', 'squat_1rm_rel', 'nordic_n'
] %}

with matches as (
    select
        p.position_group,
        p.performance_rating,
        f.sleep_hours_7d, f.protein_gpkg_7d, f.carbs_gpkg_7d, f.hydration_l_7d, f.acwr, f.chronic_load_28d,
        f.cmj_cm, f.sprint_30m_s, f.yoyo_ir1_m, f.squat_1rm_rel, f.nordic_n
    from {{ ref('mart_ml_performance_features') }} p
    join {{ ref('fct_athlete_day') }} f
      on f.athlete_id = p.athlete_id and f.date_day = p.date_day
),

thresholds as (
    select
        position_group,
        percentile_cont(0.75) within group (order by performance_rating) as elite_rating_cutoff
    from matches
    group by 1
),

elite as (
    select m.*
    from matches m
    join thresholds t using (position_group)
    where m.performance_rating >= t.elite_rating_cutoff
)

{% for metric in metrics %}
select
    e.position_group,
    '{{ metric }}'                                                          as metric,
    percentile_cont(0.5)  within group (order by e.{{ metric }})            as elite_median,
    percentile_cont(0.25) within group (order by e.{{ metric }})            as elite_p25,
    percentile_cont(0.75) within group (order by e.{{ metric }})            as elite_p75,
    (select percentile_cont(0.5) within group (order by a.{{ metric }})
       from matches a where a.position_group = e.position_group)            as position_median,
    count(e.{{ metric }})                                                   as n_elite_matches
from elite e
group by 1
{{ "union all" if not loop.last }}
{% endfor %}
