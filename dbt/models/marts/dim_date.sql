select
    date_day,
    extract(year from date_day)                                  as year,
    extract(month from date_day)                                 as month,
    dayofweek(date_day)                                          as day_of_week,   -- 0 = Sunday
    case when extract(month from date_day) = 6
           or (extract(month from date_day) = 7 and extract(day from date_day) <= 5) then 'off-season'
         when extract(month from date_day) = 7
           or (extract(month from date_day) = 8 and extract(day from date_day) <= 9) then 'pre-season'
         else 'in-season' end                                    as season_phase,
    case when extract(month from date_day) >= 7
         then cast(extract(year from date_day) as varchar) || '/' || right(cast(extract(year from date_day) + 1 as varchar), 2)
         else cast(extract(year from date_day) - 1 as varchar) || '/' || right(cast(extract(year from date_day) as varchar), 2)
    end                                                          as season
from {{ ref('int_calendar') }}
