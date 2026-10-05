{# Rolling average of the last 5 match ratings, INCLUDING the match itself (carried forward later with a strict lag). #}
select
    athlete_id,
    date_day,
    performance_rating,
    avg(performance_rating) over (
        partition by athlete_id order by date_day rows between 4 preceding and current row
    ) as form_rating_5m
from {{ ref('stg_matches') }}
where minutes_played >= 15
