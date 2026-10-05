{% test value_between(model, column_name, min_value=none, max_value=none) %}
{#- Fails for every row whose (non-null) value falls outside [min_value, max_value]. -#}
select {{ column_name }}
from {{ model }}
where {{ column_name }} is not null
  and (
    {%- if min_value is not none %} {{ column_name }} < {{ min_value }} {%- else %} 1 = 0 {%- endif %}
    or
    {%- if max_value is not none %} {{ column_name }} > {{ max_value }} {%- else %} 1 = 0 {%- endif %}
  )
{% endtest %}
