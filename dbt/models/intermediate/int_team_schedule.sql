{# Planned fixtures are public knowledge ahead of time, so counting them forward is NOT leakage (unlike an athlete's own future sessions). #}
with teams as (
    select distinct team_id from {{ ref('stg_athletes') }}
),

grid as (
    select t.team_id, c.date_day
    from teams t cross join {{ ref('int_calendar') }} c
),

flagged as (
    select
        g.team_id,
        g.date_day,
        case when f.fixture_id is not null then 1 else 0 end as is_match_day
    from grid g
    left join (select distinct team_id, date_day, fixture_id from {{ ref('stg_fixtures') }}) f
      on f.team_id = g.team_id and f.date_day = g.date_day
)

select
    team_id,
    date_day,
    is_match_day,
    sum(is_match_day) over (
        partition by team_id order by date_day rows between current row and 6 following
    ) as matches_next_7d
from flagged
