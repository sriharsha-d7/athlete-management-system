select
    n.athlete_id,
    n.nutrition_date                                          as date_day,
    n.kcal,
    n.protein_g,
    n.carbs_g,
    n.fat_g,
    n.hydration_l,
    -- per-kg values use roster body mass (static); good enough for a mart, flagged in the docs
    n.protein_g / a.weight_kg                                 as protein_gpkg,
    n.carbs_g   / a.weight_kg                                 as carbs_gpkg,
    n.kcal      / a.weight_kg                                 as kcal_per_kg
from {{ source('raw', 'nutrition_daily') }} n
join {{ source('raw', 'athletes') }} a using (athlete_id)
qualify row_number() over (partition by n.athlete_id, n.nutrition_date order by n._loaded_at desc) = 1
