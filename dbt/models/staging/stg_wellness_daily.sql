{# Out-of-physiological-range sensor readings are nulled rather than dropped so the day still counts as a check-in. #}
select
    athlete_id,
    wellness_date                                              as date_day,
    case when hrv_rmssd_ms between 5 and 250 then hrv_rmssd_ms end        as hrv_rmssd_ms,
    case when resting_hr_bpm between 30 and 100 then resting_hr_bpm end   as resting_hr_bpm,
    soreness_1_10                                              as soreness,
    fatigue_1_10                                               as fatigue,
    stress_1_10                                                as stress,
    mood_1_10                                                  as mood
from {{ source('raw', 'wellness_daily') }}
qualify row_number() over (partition by athlete_id, wellness_date order by _loaded_at desc) = 1
