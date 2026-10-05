{#
  Window-frame helpers for the dense athlete-day spine. Centralising them makes the
  anti-leakage convention explicit and testable:

    prior(n)   rows [t-n, t-1]   -> data known BEFORE the morning of day t
                                    (training load, nutrition)
    incl(n)    rows [t-n+1, t]   -> includes day t because it is measured on the
                                    morning of day t (sleep last night, HRV)
    future(n)  rows [t, t+n-1]   -> used ONLY to build labels, never features
#}
{% macro w_prior(n) -%}
    partition by athlete_id order by date_day rows between {{ n }} preceding and 1 preceding
{%- endmacro %}

{% macro w_incl(n) -%}
    partition by athlete_id order by date_day rows between {{ n - 1 }} preceding and current row
{%- endmacro %}

{% macro w_future(n) -%}
    partition by athlete_id order by date_day rows between current row and {{ n - 1 }} following
{%- endmacro %}

{% macro w_all_prior() -%}
    partition by athlete_id order by date_day rows between unbounded preceding and 1 preceding
{%- endmacro %}

{% macro w_all_incl() -%}
    partition by athlete_id order by date_day rows between unbounded preceding and current row
{%- endmacro %}
