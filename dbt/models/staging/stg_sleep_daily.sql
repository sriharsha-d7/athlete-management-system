{# date_day is the wake-up date: the night that ends on the morning of date_day. #}
select
    athlete_id,
    sleep_date                                           as date_day,
    case when sleep_hours between 2 and 14 then sleep_hours end as sleep_hours,
    sleep_efficiency_pct,
    deep_sleep_pct
from {{ source('raw', 'sleep_daily') }}
qualify row_number() over (partition by athlete_id, sleep_date order by _loaded_at desc) = 1
