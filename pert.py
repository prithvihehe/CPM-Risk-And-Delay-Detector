"""PERT expected-duration analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cpm import compute_cpm


@dataclass
class PertConfig:
    optimistic_col: str = "optimistic_duration"
    most_likely_col: str = "most_likely_duration"
    pessimistic_col: str = "pessimistic_duration"


def _with_pert_columns(df: pd.DataFrame, cfg: PertConfig) -> pd.DataFrame:
    out = df.copy()
    for col in (cfg.optimistic_col, cfg.most_likely_col, cfg.pessimistic_col):
        if col not in out.columns:
            out[col] = out["duration"]
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(out["duration"])

    out["pert_expected_duration"] = (
        out[cfg.optimistic_col]
        + 4 * out[cfg.most_likely_col]
        + out[cfg.pessimistic_col]
    ) / 6.0
    out["pert_variance"] = ((out[cfg.pessimistic_col] - out[cfg.optimistic_col]) / 6.0) ** 2
    return out


def compute_pert_schedule(df: pd.DataFrame, cfg: PertConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      1) DataFrame with per-task PERT expected duration and variance
      2) CPM table computed using expected durations
    """
    cfg = cfg or PertConfig()
    pert_df = _with_pert_columns(df, cfg)

    pert_schedule_df = pert_df.copy()
    pert_schedule_df["duration"] = pert_schedule_df["pert_expected_duration"]
    pert_cpm = compute_cpm(pert_schedule_df)
    return pert_df, pert_cpm
