{#
  "What is this athlete missing?" - descriptive gap analysis at the latest date.

  One row per athlete x metric. The target is the evidence-based optimum when one exists
  (sleep, protein, carbs, hydration, ACWR band, hamstring strength) and otherwise the median of
  top-quartile performers in the athlete's position group (physical tests).
  shortfall is always >= 0 and expressed in the metric's own unit, in the direction that helps.
#}
{% set metrics = [
    'sleep_hours_7d', 'protein_gpkg_7d', 'carbs_gpkg_7d', 'hydration_l_7d', 'acwr',
    'cmj_cm', 'sprint_30m_s', 'yoyo_ir1_m', 'squat_1rm_rel', 'nordic_n'
] %}

with state as (
    select * from {{ ref('mart_athlete_current_state') }}
),

long as (
    {% for metric in metrics %}
    select athlete_id, team_id, position_group, as_of_date, '{{ metric }}' as metric, {{ metric }} as current_value
    from state
    {{ "union all" if not loop.last }}
    {% endfor %}
),

joined as (
    select
        l.*,
        e.direction,
        e.target_min,
        e.target_max,
        e.unit,
        coalesce(e.target_optimal, b.elite_median)  as target_value,
        case when e.target_optimal is not null then 'evidence' else 'position_benchmark' end as target_source,
        b.position_median
    from long l
    join {{ ref('seed_evidence_targets') }} e on e.metric = l.metric
    left join {{ ref('mart_position_benchmarks') }} b
      on b.position_group = l.position_group and b.metric = l.metric
),

scored as (
    select
        *,
        case direction
            when 'higher_better' then greatest(target_value - current_value, 0)
            when 'lower_better'  then greatest(current_value - target_value, 0)
            when 'band'          then greatest(current_value - target_max, target_min - current_value, 0)
        end                                                                       as shortfall,
        case direction
            when 'higher_better' then 'increase'
            when 'lower_better'  then 'decrease'
            when 'band'          then case when current_value > target_max then 'decrease'
                                           when current_value < target_min then 'increase'
                                           else 'maintain' end
        end                                                                       as needed_direction,
        percent_rank() over (partition by position_group, metric order by current_value) as raw_pct_rank
    from joined
    where current_value is not null
)

select
    athlete_id,
    team_id,
    position_group,
    as_of_date,
    metric,
    unit,
    direction,
    current_value,
    target_value,
    target_source,
    target_min,
    target_max,
    position_median,
    shortfall,
    needed_direction,
    shortfall / nullif(abs(target_value), 0)                                     as shortfall_pct,
    case when shortfall = 0 then 'on_target'
         when direction = 'higher_better' and target_min is not null and current_value < target_min then 'below_minimum'
         when direction = 'band' then 'out_of_band'
         else 'below_target' end                                                  as status,
    case when direction = 'lower_better' then 1 - raw_pct_rank else raw_pct_rank end as percentile_in_position
from scored
