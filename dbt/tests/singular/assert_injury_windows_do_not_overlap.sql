-- An athlete cannot start a new injury while still unavailable from a previous one.
select a.athlete_id, a.injury_id as injury_a, b.injury_id as injury_b
from {{ ref('stg_injuries') }} a
join {{ ref('stg_injuries') }} b
  on a.athlete_id = b.athlete_id
 and a.injury_id < b.injury_id
 and b.injury_date between a.injury_date and a.unavailable_to
