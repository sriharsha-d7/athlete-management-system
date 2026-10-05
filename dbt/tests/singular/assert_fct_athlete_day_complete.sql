-- The wide fact must contain exactly one row per spine day: no fan-out, no dropped athlete-days.
with spine as (select count(*) as n from {{ ref('int_athlete_day_spine') }}),
     fact  as (select count(*) as n from {{ ref('fct_athlete_day') }})
select spine.n as spine_rows, fact.n as fact_rows
from spine cross join fact
where spine.n <> fact.n
