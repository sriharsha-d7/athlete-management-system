select
    athlete_id,
    team_id,
    upper(trim(position))     as position_code,
    birth_date,
    height_cm,
    weight_kg,
    lower(dominant_foot)      as dominant_foot,
    signed_date
from {{ source('raw', 'athletes') }}
