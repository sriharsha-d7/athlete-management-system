{#
  Use the custom schema exactly as given (STAGING, MARTS, ...) instead of dbt's
  default "<target_schema>_<custom_schema>" so DuckDB and Snowflake produce the
  same layout and the Python/ML code can rely on stable schema names.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
