{#
  Single source of truth for the model feature set. Both ML feature marts select exactly
  these columns; the Python layer treats every non-metadata numeric column in the mart as a
  feature, so adding a feature = add it here (and in int_rolling_features). Nothing else.
#}
{% macro ml_feature_columns(alias='f') %}
    {%- set cols = [
        'acute_load_7d', 'chronic_load_28d', 'acwr', 'load_monotony_7d', 'load_strain_7d',
        'hsr_distance_7d', 'sprint_count_7d', 'max_speed_28d', 'sessions_7d', 'match_minutes_28d', 'gym_sessions_28d',
        'sleep_hours_3d', 'sleep_hours_7d', 'sleep_hours_28d', 'sleep_variability_7d', 'sleep_efficiency_7d',
        'hrv_z', 'rhr_z', 'soreness_3d', 'fatigue_3d', 'stress_7d', 'mood_7d',
        'wellness_log_rate_7d', 'nutrition_log_rate_7d',
        'protein_gpkg_7d', 'carbs_gpkg_3d', 'carbs_gpkg_7d', 'kcal_per_kg_7d', 'hydration_l_3d', 'hydration_l_7d',
        'cmj_cm', 'sprint_30m_s', 'yoyo_ir1_m', 'squat_1rm_rel', 'nordic_n', 'body_fat_pct', 'days_since_test',
        'age_years', 'height_cm', 'weight_kg', 'prior_injuries', 'days_since_last_injury', 'days_since_return',
        'rating_avg_5m', 'matches_next_7d'
    ] -%}
    {%- for c in cols %}{{ alias }}.{{ c }}{{ "," if not loop.last }}
    {% endfor -%}
{% endmacro %}
