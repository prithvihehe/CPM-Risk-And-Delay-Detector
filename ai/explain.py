"""
Rule-based delay explanations weighted by RandomForest feature importances.

Loads weights from ai/feature_importance.json (written by ai/train.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai.data_generator import FEATURE_COLUMNS

DEFAULT_IMPORTANCE_PATH = Path(__file__).resolve().parent / "feature_importance.json"

# Map human-readable reason -> feature column name (for importance lookup)
REASON_FEATURE_KEY = {
    "Critical Path": "is_critical",
    "Long Duration": "duration",
    "High Resource Demand": "resource_count",
    "No Slack": "slack",
}


def load_feature_importances(path: Path | str | None = None) -> dict[str, float]:
    """Load feature name -> importance from JSON. Raises FileNotFoundError if missing."""
    p = Path(path) if path else DEFAULT_IMPORTANCE_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Run training to generate it: python -m ai.train"
        )
    with open(p, encoding="utf-8") as f:
        raw: Any = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object in {p}, got {type(raw).__name__}")
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k in FEATURE_COLUMNS:
            out[str(k)] = float(v)
    for name in FEATURE_COLUMNS:
        if name not in out:
            out[name] = 0.0
    return out


def impact_tier(importance_score: float) -> str:
    """Optional labels for model importance weight."""
    if importance_score > 0.3:
        return "High Impact"
    if importance_score >= 0.1:
        return "Medium Impact"
    return "Low Impact"


def explain_task(
    duration: float,
    resource_count: int,
    is_critical: bool,
    slack: float,
    *,
    importances: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """
    Return contributing factors as (reason_label, importance_score), sorted by
    importance descending. Only includes rules that apply to this task.

    importance_score is the RandomForest feature_importances_ weight for the
    feature tied to that reason (from feature_importance.json).
    """
    weights = importances if importances is not None else load_feature_importances()

    fired: list[tuple[str, float]] = []

    if bool(is_critical):
        key = REASON_FEATURE_KEY["Critical Path"]
        fired.append(("Critical Path", float(weights.get(key, 0.0))))

    if float(duration) > 5:
        key = REASON_FEATURE_KEY["Long Duration"]
        fired.append(("Long Duration", float(weights.get(key, 0.0))))

    if int(resource_count) > 3:
        key = REASON_FEATURE_KEY["High Resource Demand"]
        fired.append(("High Resource Demand", float(weights.get(key, 0.0))))

    if abs(float(slack)) < 1e-9:
        key = REASON_FEATURE_KEY["No Slack"]
        fired.append(("No Slack", float(weights.get(key, 0.0))))

    fired.sort(key=lambda x: x[1], reverse=True)
    return fired


def explain_task_with_tiers(
    duration: float,
    resource_count: int,
    is_critical: bool,
    slack: float,
    *,
    importances: dict[str, float] | None = None,
) -> list[tuple[str, float, str]]:
    """Like explain_task but adds impact tier per reason."""
    reasons = explain_task(
        duration,
        resource_count,
        is_critical,
        slack,
        importances=importances,
    )
    return [(r, w, impact_tier(w)) for r, w in reasons]


def format_reasons_line(reasons: list[tuple[str, float]]) -> str:
    """Single-line display with optional impact tier labels."""
    if not reasons:
        return "—"
    parts = [f"{r} (weight {w:.3f}, {impact_tier(w)})" for r, w in reasons]
    return "; ".join(parts)


def summarize_delay_risk(
    delay_probability: float,
    reasons: list[tuple[str, float]],
) -> str:
    """Short natural-language line for UI."""
    pct = f"{delay_probability * 100:.0f}%"
    if not reasons:
        return f"Estimated delay risk {pct}; no rule-based drivers matched (or explanations unavailable)."
    parts = [r for r, _ in reasons[:5]]
    because = ", ".join(parts[:-1]) + (f", and {parts[-1]}" if len(parts) > 1 else parts[0])
    return f"Estimated delay risk {pct} partly because of: {because}."
