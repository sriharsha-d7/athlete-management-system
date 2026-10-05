select
    fixture_id,
    team_id,
    fixture_date as date_day,
    competition,
    home_away,
    opponent_strength
from {{ source('raw', 'fixtures') }}
