# Athlete Management System: reproducible end-to-end pipeline.
# Default target is local DuckDB. For Snowflake:  WAREHOUSE=snowflake DBT_TARGET=snowflake make all
SHELL := /bin/bash
export DBT_PROFILES_DIR := $(CURDIR)/dbt
export DBT_TARGET ?= dev

.PHONY: help setup data load dbt-pre ml dbt-post all test dbt-test app docs clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:           ## install python dependencies
	pip install -r requirements.txt

data:            ## 1. simulate the league -> data/raw/*.parquet
	python data_gen/generate.py --out data/raw

load:            ## 2. load raw parquet into the warehouse RAW schema
	python warehouse/load_raw.py

dbt-pre:         ## 3. seeds + staging + intermediate + marts + tests (everything not needing ML output)
	cd dbt && dbt build --selector pre_ml --target $(DBT_TARGET)

ml:              ## 4. train both models, score every athlete-day, generate + validate recommendations
	python ml/train.py
	python ml/recommend.py

dbt-post:        ## 5. marts that join model output back in (action plan, risk timeline, squad view)
	cd dbt && dbt build --selector post_ml --target $(DBT_TARGET)

all: data load dbt-pre ml dbt-post   ## run the whole pipeline

test:            ## python unit tests (fast, no warehouse needed for most)
	python -m pytest -q

dbt-test:        ## re-run all dbt tests
	cd dbt && dbt test --target $(DBT_TARGET)

app:             ## launch the dashboard
	streamlit run app/streamlit_app.py

docs:            ## generate dbt docs (lineage graph + catalog)
	cd dbt && dbt docs generate --target $(DBT_TARGET) && dbt docs serve

clean:
	rm -rf data/raw data/athlete.duckdb* models dbt/target dbt/logs
