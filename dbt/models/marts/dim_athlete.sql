select
    a.athlete_id,
    a.team_id,
    a.position_code,
    p.position_name,
    p.position_group,
    a.birth_date,
    {{ dbt.datediff('a.birth_date', "cast('" ~ var('calendar_end') ~ "' as date)", 'day') }} / 365.25 as age_years_at_as_of,
    a.height_cm,
    a.weight_kg,
    round(a.weight_kg / power(a.height_cm / 100.0, 2), 1)                     as bmi,
    a.dominant_foot,
    a.signed_date
from {{ ref('stg_athletes') }} a
left join {{ ref('seed_position_groups') }} p
  on p.position_code = a.position_code
