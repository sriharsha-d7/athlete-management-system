{#
  calendar_series(start_date, end_date) -> one row per day, column date_day.
  Adapter-dispatched because generating rows differs between engines.
#}
{% macro calendar_series(start_date, end_date) %}
    {{ return(adapter.dispatch('calendar_series', 'athlete_ams')(start_date, end_date)) }}
{% endmacro %}

{% macro default__calendar_series(start_date, end_date) %}
    select cast(unnest(generate_series(
        cast('{{ start_date }}' as date), cast('{{ end_date }}' as date), interval 1 day)) as date) as date_day
{% endmacro %}

{% macro duckdb__calendar_series(start_date, end_date) %}
    select cast(unnest(generate_series(
        cast('{{ start_date }}' as date), cast('{{ end_date }}' as date), interval 1 day)) as date) as date_day
{% endmacro %}

{% macro snowflake__calendar_series(start_date, end_date) %}
    select date_day
    from (
        select dateadd(day, row_number() over (order by seq4()) - 1, to_date('{{ start_date }}')) as date_day
        from table(generator(rowcount => 5000))
    )
    where date_day <= to_date('{{ end_date }}')
{% endmacro %}
