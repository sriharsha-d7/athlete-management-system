"""Turn a lever from the dbt seed catalog into a counterfactual edit of the feature frame."""
from __future__ import annotations

import numpy as np
import pandas as pd


def parse_shifts(spec: str) -> list[tuple[str, float]]:
    """'sleep_hours_3d:1.0|sleep_hours_7d:1.0|sleep_hours_28d:0.5' -> [(feature, weight), ...]"""
    out = []
    for part in str(spec).split("|"):
        name, w = part.split(":")
        out.append((name.strip(), float(w)))
    return out


def apply_lever(df: pd.DataFrame, shifts: list[tuple[str, float]], step_type: str, step) -> pd.DataFrame:
    """
    Return a copy of df with the lever pulled.

    step may be a scalar or a per-row array (each athlete has their own capped step).
      add   : feature += weight * step
      scale : feature *= (1 + weight * step)
    Derived features are recomputed so the frame stays internally consistent
    (ACWR = acute / chronic, strain scales with weekly load).
    """
    out = df.copy()
    step = np.asarray(step, dtype=float) if not np.isscalar(step) else float(step)
    for feat, w in shifts:
        if step_type == "add":
            out[feat] = out[feat] + w * step
        elif step_type == "scale":
            out[feat] = out[feat] * (1.0 + w * step)
        else:
            raise ValueError(f"unknown step_type {step_type}")
    if any(f == "acute_load_7d" for f, _ in shifts):
        out["acwr"] = out["acute_load_7d"] / out["chronic_load_28d"].replace(0, np.nan)
        if "load_strain_7d" in out:
            out["load_strain_7d"] = out["load_strain_7d"] * (out["acute_load_7d"] / df["acute_load_7d"].replace(0, np.nan))
    return out
