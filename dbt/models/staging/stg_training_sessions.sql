{#
  Grain: one row per session_id.
  Cleaning rules (each one maps to a defect injected by the simulator):
    1. Duplicate deliveries      -> keep the latest _loaded_at per session_id
    2. Two GPS vendors           -> normalise max speed to km/h (VENDOR_B reports m/s)
    3. Sensor glitches           -> speeds above 45 km/h (world-record territory) become NULL
    4. Device failures           -> GPS metrics stay NULL; the session load (sRPE) is kept
    5. Impossible distances      -> HSR distance cannot exceed total distance
#}
with deduped as (

    select *
    from {{ source('raw', 'training_sessions') }}
    qualify row_number() over (partition by session_id order by _loaded_at desc) = 1

),

normalised as (

    select
        session_id,
        athlete_id,
        session_date,
        lower(session_type)                                           as session_type,
        duration_min,
        rpe,
        total_distance_m,
        hsr_distance_m,
        sprint_count,
        case when speed_unit = 'ms' then max_speed_raw * 3.6
             else max_speed_raw end                                   as max_speed_kmh_raw,
        player_load,
        avg_hr_bpm,
        gps_vendor
    from deduped
    where duration_min > 0
      and rpe between 0 and 10

)

select
    session_id,
    athlete_id,
    session_date,
    session_type,
    duration_min,
    rpe,
    duration_min * rpe                                                as srpe_load,
    total_distance_m,
    case when hsr_distance_m <= total_distance_m then hsr_distance_m end as hsr_distance_m,
    sprint_count,
    case when max_speed_kmh_raw <= 45 then max_speed_kmh_raw end      as max_speed_kmh,
    player_load,
    avg_hr_bpm,
    gps_vendor,
    total_distance_m is not null                                      as has_gps
from normalised
