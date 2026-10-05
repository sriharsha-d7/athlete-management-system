select
    m.match_stat_id,
    m.fixture_id,
    m.athlete_id,
    m.match_date                       as date_day,
    m.minutes_played,
    m.started,
    m.performance_rating,
    m.goals,
    m.assists,
    m.xg,
    m.xa,
    m.pass_accuracy_pct,
    m.duels_won_pct,
    f.competition,
    f.home_away,
    f.opponent_strength
from {{ source('raw', 'match_player_stats') }} m
left join {{ source('raw', 'fixtures') }} f using (fixture_id)
qualify row_number() over (partition by m.match_stat_id order by m._loaded_at desc) = 1
