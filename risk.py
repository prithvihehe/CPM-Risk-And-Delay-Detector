"""
Rule-based risk scoring for project tasks.
"""

from __future__ import annotations

import pandas as pd


def risk_score_row(
    duration: float,
    resource_count: int,
    is_critical: bool,
    slack: float,
) -> float:
    """
    Additive rules:
    - critical -> +0.5
    - duration > 4 -> +0.3
    - resource_count > 2 -> +0.2
    - slack == 0 -> +0.3
    """
    score = 0.0
    if is_critical:
        score += 0.5
    if duration > 4:
        score += 0.3
    if resource_count > 2:
        score += 0.2
    if slack == 0:
        score += 0.3
    return round(score, 4)


def add_risk_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Append a risk_score column to the feature dataframe."""
    out = features.copy()
    slack_zero = out["slack"].abs() < 1e-9
    out["risk_score"] = (
        out["is_critical"].astype(bool) * 0.5
        + (out["duration"].astype(float) > 4) * 0.3
        + (out["resource_count"].astype(int) > 2) * 0.2
        + slack_zero * 0.3
    ).round(4)
    return out


def high_risk_mask(scores: pd.Series, threshold: float = 1.0) -> pd.Series:
    """Boolean mask for tasks at or above the risk threshold."""
    return scores >= threshold
