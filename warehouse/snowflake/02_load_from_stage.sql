-- =============================================================================
-- 02_load_from_stage.sql  |  Bulk-load the parquet extracts with PUT + COPY INTO.
-- Run from SnowSQL / Snowflake CLI (PUT is a client-side command).
-- Alternative: `python warehouse/load_raw.py --target snowflake` does the same
-- with write_pandas and needs no SnowSQL.
-- =============================================================================
USE ROLE AMS_LOADER;
USE WAREHOUSE AMS_TRANSFORM_WH;
USE SCHEMA ATHLETE_DB.RAW;

CREATE FILE FORMAT IF NOT EXISTS FF_PARQUET TYPE = PARQUET;
CREATE STAGE IF NOT EXISTS STG_RAW FILE_FORMAT = FF_PARQUET;

-- Adjust the local path to where `make data` wrote the files.
PUT file://./data/raw/*.parquet @STG_RAW AUTO_COMPRESS = FALSE OVERWRITE = TRUE;

-- One COPY per table. MATCH_BY_COLUMN_NAME keeps the load robust to column order.
COPY INTO ATHLETES            FROM @STG_RAW/athletes.parquet            MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO FIXTURES            FROM @STG_RAW/fixtures.parquet            MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO TRAINING_SESSIONS   FROM @STG_RAW/training_sessions.parquet   MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO WELLNESS_DAILY      FROM @STG_RAW/wellness_daily.parquet      MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO SLEEP_DAILY         FROM @STG_RAW/sleep_daily.parquet         MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO NUTRITION_DAILY     FROM @STG_RAW/nutrition_daily.parquet     MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO INJURIES            FROM @STG_RAW/injuries.parquet            MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO MATCH_PLAYER_STATS  FROM @STG_RAW/match_player_stats.parquet  MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
COPY INTO PHYSICAL_TESTS      FROM @STG_RAW/physical_tests.parquet      MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- Sanity check
SELECT 'ATHLETES' t, COUNT(*) n FROM ATHLETES UNION ALL
SELECT 'TRAINING_SESSIONS', COUNT(*) FROM TRAINING_SESSIONS UNION ALL
SELECT 'WELLNESS_DAILY', COUNT(*) FROM WELLNESS_DAILY UNION ALL
SELECT 'SLEEP_DAILY', COUNT(*) FROM SLEEP_DAILY UNION ALL
SELECT 'NUTRITION_DAILY', COUNT(*) FROM NUTRITION_DAILY UNION ALL
SELECT 'INJURIES', COUNT(*) FROM INJURIES UNION ALL
SELECT 'MATCH_PLAYER_STATS', COUNT(*) FROM MATCH_PLAYER_STATS UNION ALL
SELECT 'PHYSICAL_TESTS', COUNT(*) FROM PHYSICAL_TESTS;
