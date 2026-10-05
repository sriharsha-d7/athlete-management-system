from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import INJURY_META, PERF_META, POSITION_GROUPS, TRAIN_END, VAL_END


def feature_columns(df: pd.DataFrame, meta: list[str]) -> list[str]:
    """Every numeric column that is not metadata. dbt owns the definition of 'feature'."""
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date_day"])
    df = df.copy()
    df["date_day"] = d
    df["data_split"] = np.select(
        [d <= pd.Timestamp(TRAIN_END), d <= pd.Timestamp(VAL_END)], ["train", "val"], default="test")
    return df


def make_X(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Numeric features + one-hot position group (fixed columns so train/score always align)."""
    X = df[feats].astype(float).copy()
    for g in POSITION_GROUPS:
        X[f"pos_{g}"] = (df["position_group"] == g).astype(float).values
    return X


def load_injury_frame(wh) -> pd.DataFrame:
    df = wh.query("select * from marts.mart_ml_injury_features")
    return add_split(df)


def load_perf_frame(wh) -> pd.DataFrame:
    df = wh.query("select * from marts.mart_ml_performance_features")
    return add_split(df)


__all__ = ["feature_columns", "add_split", "make_X", "load_injury_frame", "load_perf_frame",
           "INJURY_META", "PERF_META"]
