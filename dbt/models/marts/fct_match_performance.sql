select
    m.match_stat_id,
    m.fixture_id,
    m.athlete_id,
    d.team_id,
    d.position_group,
    m.date_day,
    m.minutes_played,
    m.started,
    m.performance_rating,
    m.goals,
    m.assists,
    m.xg,
    m.xa,
    m.pass_accuracy_pct,
    m.duels_won_pct,
    m.competition,
    m.home_away,
    m.opponent_strength
from {{ ref('stg_matches') }} m
join {{ ref('dim_athlete') }} d using (athlete_id)
