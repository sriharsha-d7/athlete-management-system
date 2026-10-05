"""
Thin warehouse abstraction so the loader, the ML layer and the Streamlit app run
unchanged against Snowflake (production) or DuckDB (local, zero-cost dev/CI).

    WAREHOUSE=duckdb     (default)  -> ./data/athlete.duckdb
    WAREHOUSE=snowflake             -> SNOWFLAKE_* environment variables
"""
from __future__ import annotations

import os
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DUCKDB_PATH = os.environ.get("DUCKDB_PATH", str(ROOT / "data" / "athlete.duckdb"))


def target() -> str:
    return os.environ.get("WAREHOUSE", "duckdb").lower()


class Warehouse:
    def __init__(self, read_only: bool = False):
        self.kind = target()
        if self.kind == "duckdb":
            import duckdb
            self.con = duckdb.connect(DUCKDB_PATH, read_only=read_only)
        elif self.kind == "snowflake":
            import snowflake.connector
            self.con = snowflake.connector.connect(
                account=os.environ["SNOWFLAKE_ACCOUNT"],
                user=os.environ["SNOWFLAKE_USER"],
                password=os.environ.get("SNOWFLAKE_PASSWORD"),
                role=os.environ.get("SNOWFLAKE_ROLE", "AMS_ML"),
                warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "AMS_TRANSFORM_WH"),
                database=os.environ.get("SNOWFLAKE_DATABASE", "ATHLETE_DB"),
            )
        else:
            raise ValueError(f"Unknown WAREHOUSE={self.kind}")

    # ---------------------------------------------------------------- read
    def query(self, sql: str) -> pd.DataFrame:
        if self.kind == "duckdb":
            df = self.con.execute(sql).df()
        else:
            cur = self.con.cursor()
            cur.execute(sql)
            df = cur.fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        return df

    def execute(self, sql: str) -> None:
        if self.kind == "duckdb":
            self.con.execute(sql)
        else:
            self.con.cursor().execute(sql)

    # --------------------------------------------------------------- write
    def write(self, df: pd.DataFrame, schema: str, table: str) -> None:
        """Replace schema.table with df."""
        if self.kind == "duckdb":
            self.con.execute(f"create schema if not exists {schema}")
            self.con.register("_df", df)
            self.con.execute(f"create or replace table {schema}.{table} as select * from _df")
            self.con.unregister("_df")
        else:
            from snowflake.connector.pandas_tools import write_pandas
            self.execute(f"create schema if not exists {schema.upper()}")
            d = df.copy()
            d.columns = [c.upper() for c in d.columns]
            write_pandas(self.con, d, table.upper(), schema=schema.upper(),
                         auto_create_table=True, overwrite=True, quote_identifiers=False)

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
