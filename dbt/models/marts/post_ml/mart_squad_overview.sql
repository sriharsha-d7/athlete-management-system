{# Team-level readiness snapshot at the latest date, for the head coach / performance director view. #}
with latest as (
    select * from {{ ref('mart_athlete_risk_daily') }}
    where date_day = (select max(date_day) from {{ ref('mart_athlete_risk_daily') }})
),

top_lever as (
    select team_id, lever_id, count(*) as athletes
    from {{ ref('mart_athlete_action_plan') }}
    where priority_rank = 1
    group by 1, 2
),

top_lever_ranked as (
    select team_id, lever_id as most_common_top_lever, athletes as athletes_with_that_lever
    from top_lever
    qualify row_number() over (partition by team_id order by athletes desc, lever_id) = 1
)

select
    l.team_id,
    count(*)                                                              as athletes,
    sum(l.is_unavailable)                                                 as athletes_injured,
    sum(case when l.risk_band in ('High', 'Critical') then 1 else 0 end)  as athletes_high_risk,
    avg(l.injury_risk_7d)                                                 as avg_injury_risk_7d,
    avg(l.readiness_score)                                                as avg_readiness,
    avg(l.predicted_rating)                                               as avg_expected_rating,
    sum(case when l.acwr_zone = 'danger' then 1 else 0 end)               as athletes_acwr_danger,
    t.most_common_top_lever,
    t.athletes_with_that_lever
from latest l
left join top_lever_ranked t on t.team_id = l.team_id
group by l.team_id, t.most_common_top_lever, t.athletes_with_that_lever
