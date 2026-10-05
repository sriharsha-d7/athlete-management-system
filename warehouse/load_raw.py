#!/usr/bin/env python
"""Load the generated parquet extracts into the RAW schema.

    python warehouse/load_raw.py                      # DuckDB (default)
    WAREHOUSE=snowflake python warehouse/load_raw.py  # Snowflake via write_pandas
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from warehouse.conn import Warehouse  # noqa: E402

TABLES = ["athletes", "fixtures", "training_sessions", "wellness_daily", "sleep_daily",
          "nutrition_daily", "injuries", "match_player_stats", "physical_tests"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw")
    a = ap.parse_args()
    src = pathlib.Path(a.src)
    with Warehouse() as wh:
        for t in TABLES:
            df = pd.read_parquet(src / f"{t}.parquet")
            wh.write(df, "raw", t)
            print(f"[{wh.kind}] raw.{t}: {len(df):,} rows")


if __name__ == "__main__":
    main()
