{#
  Severity bands follow the football consensus statement on injury definitions:
  minimal 1-3 days, mild 4-7, moderate 8-28, severe > 28 days absent.

  An athlete is UNAVAILABLE from injury_date + 1 through injury_date + days_out.
  The injury_date itself is the onset day (the athlete was exposed and partly trained).
#}
select
    injury_id,
    athlete_id,
    injury_date,
    body_part,
    injury_type,
    mechanism,
    occurred_during,
    days_out,
    case when days_out <= 3  then 'minimal'
         when days_out <= 7  then 'mild'
         when days_out <= 28 then 'moderate'
         else 'severe' end                                         as severity,
    {{ dbt.dateadd('day', 1, 'injury_date') }}                     as unavailable_from,
    {{ dbt.dateadd('day', 'days_out', 'injury_date') }}            as unavailable_to
from {{ source('raw', 'injuries') }}
qualify row_number() over (partition by injury_id order by _loaded_at desc) = 1
