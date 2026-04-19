"""
Shared CPM + risk + delay pipeline for Streamlit and the web UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Hashable

import networkx as nx
import pandas as pd

from ai.explain import (
    explain_task,
    format_reasons_line,
    load_feature_importances,
    summarize_delay_risk,
)
from ai.predict import DelayPredictor
from cpm import (
    REQUIRED_COLUMNS,
    build_dag,
    build_feature_frame,
    compute_cpm,
    critical_path_task_ids,
    validate_dependency_references,
)
from risk import add_risk_scores, high_risk_mask
from visuals import plot_dependency_graph, plot_explanation, plot_gantt, plot_risk

AI_FEATURES = ["duration", "resource_count", "is_critical", "slack"]


def missing_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _fig_to_dict(fig: Any) -> dict[str, Any]:
    return json.loads(fig.to_json())


@dataclass
class AnalysisResult:
    ok: bool
    missing_columns: list[str] = field(default_factory=list)
    dependency_errors: list[str] = field(default_factory=list)
    message: str | None = None
    warnings: list[str] = field(default_factory=list)
    df: pd.DataFrame | None = None
    cpm_df: pd.DataFrame | None = None
    dag: nx.DiGraph | None = None
    project_completion: float | None = None
    critical_path: list[Hashable] = field(default_factory=list)
    scored: pd.DataFrame | None = None
    high_risk_mask: pd.Series | None = None
    task_insights: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    es: dict[Hashable, float] = field(default_factory=dict)
    ef: dict[Hashable, float] = field(default_factory=dict)
    critical: dict[Hashable, bool] = field(default_factory=dict)
    importance_error: str | None = None


def run_analysis(df: pd.DataFrame, *, risk_threshold: float = 1.0) -> AnalysisResult:
    miss = missing_columns(df)
    if miss:
        return AnalysisResult(ok=False, missing_columns=miss)

    dep_errors = validate_dependency_references(df)
    if dep_errors:
        return AnalysisResult(ok=False, dependency_errors=dep_errors)

    try:
        g = build_dag(df)
        cpm_df = compute_cpm(df, g)
    except ValueError as e:
        return AnalysisResult(ok=False, message=str(e))

    cp_path = critical_path_task_ids(df, cpm_df)
    features = build_feature_frame(df, cpm_df)
    scored = add_risk_scores(features)

    warnings: list[str] = []
    try:
        predictor = DelayPredictor()
        scored["delay_probability"] = predictor.predict_batch(scored[AI_FEATURES])
    except FileNotFoundError as e:
        warnings.append(str(e))
        scored["delay_probability"] = float("nan")

    importance_error: str | None = None
    try:
        feature_importances = load_feature_importances()
    except FileNotFoundError as e:
        feature_importances = None
        importance_error = str(e)

    task_insights: list[dict[str, Any]] = []
    explain_lines: list[str] = []
    summaries: list[str] = []

    for _, row in scored.iterrows():
        tid = row["task_id"]
        dp = row["delay_probability"]
        reasons: list[tuple[str, float]] = []
        if feature_importances is not None:
            reasons = explain_task(
                float(row["duration"]),
                int(row["resource_count"]),
                bool(row["is_critical"]),
                float(row["slack"]),
                importances=feature_importances,
            )
            explain_lines.append(format_reasons_line(reasons))
            if pd.isna(dp):
                summaries.append("Delay probability unavailable.")
            else:
                summaries.append(summarize_delay_risk(float(dp), reasons))
        else:
            explain_lines.append("—")
            summaries.append(
                importance_error or "Explanations unavailable (missing feature importances)."
            )

        task_insights.append(
            {
                "task_id": tid,
                "delay_probability": float(dp) if pd.notna(dp) else None,
                "reasons": [(r, float(w)) for r, w in reasons],
            }
        )

    scored["delay_explanation"] = explain_lines
    scored["delay_summary"] = summaries

    hr = high_risk_mask(scored["risk_score"], threshold=risk_threshold)

    tasks: list[dict[str, Any]] = []
    for (_, row), ins in zip(scored.iterrows(), task_insights):
        tasks.append(
            {
                "id": row["task_id"],
                "name": row["task_name"],
                "delay_probability": ins["delay_probability"],
                "reasons": ins["reasons"],
            }
        )

    ES = cpm_df.set_index("task_id")["ES"].to_dict()
    EF = cpm_df.set_index("task_id")["EF"].to_dict()
    critical = cpm_df.set_index("task_id")["is_critical"].to_dict()

    return AnalysisResult(
        ok=True,
        warnings=warnings,
        df=df,
        cpm_df=cpm_df,
        dag=g,
        project_completion=float(cpm_df["EF"].max()),
        critical_path=list(cp_path),
        scored=scored,
        high_risk_mask=hr,
        task_insights=task_insights,
        tasks=tasks,
        es=ES,
        ef=EF,
        critical=critical,
        importance_error=importance_error,
    )


def analysis_to_api_payload(result: AnalysisResult) -> dict[str, Any]:
    """Serialize a successful AnalysisResult for the JSON API (includes Plotly figure dicts)."""
    if not result.ok or result.df is None or result.cpm_df is None or result.scored is None:
        return {"ok": False, "error": result.message or "Analysis failed."}

    df = result.df
    cpm_df = result.cpm_df
    scored = result.scored
    hr = result.high_risk_mask
    assert result.dag is not None
    assert hr is not None

    cpm_display = cpm_df.merge(df[["task_id", "task_name"]], on="task_id", how="left")
    scored_records = _dataframe_to_json_safe_records(scored)
    high_ids = set(scored.loc[hr, "task_id"].tolist()) if hr is not None else set()
    high_risk_flags = [tid in high_ids for tid in scored["task_id"].tolist()]

    explain_charts: dict[str, Any] = {}
    for ins in result.task_insights:
        tid = ins["task_id"]
        key = str(tid)
        explain_charts[key] = _fig_to_dict(plot_explanation(ins["reasons"]))

    payload: dict[str, Any] = {
        "ok": True,
        "warnings": result.warnings,
        "importance_error": result.importance_error,
        "project_completion": result.project_completion,
        "critical_path": [str(x) for x in result.critical_path],
        "cpm_table": _dataframe_to_json_safe_records(cpm_display),
        "scored_table": scored_records,
        "high_risk_flags": high_risk_flags,
        "task_insights": [
            {
                "task_id": str(ins["task_id"]),
                "delay_probability": ins["delay_probability"],
                "reasons": [{"label": r, "weight": w} for r, w in ins["reasons"]],
            }
            for ins in result.task_insights
        ],
        "charts": {
            "dependency": _fig_to_dict(plot_dependency_graph(result.dag, df, cpm_df)),
            "gantt": _fig_to_dict(
                plot_gantt(result.tasks, result.es, result.ef, result.critical)
            ),
            "delay_risk": _fig_to_dict(plot_risk(result.tasks)),
            "explanation_by_task": explain_charts,
        },
    }
    return payload


def _dataframe_to_json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for k, v in rec.items():
            if pd.isna(v):
                row[str(k)] = None
            elif hasattr(v, "item"):
                row[str(k)] = v.item()
            else:
                row[str(k)] = v
        rows.append(row)
    return rows


def error_payload(result: AnalysisResult) -> dict[str, Any]:
    if result.missing_columns:
        return {
            "ok": False,
            "error": "missing_columns",
            "missing_columns": result.missing_columns,
        }
    if result.dependency_errors:
        return {
            "ok": False,
            "error": "dependency_errors",
            "messages": result.dependency_errors,
        }
    return {"ok": False, "error": "analysis_failed", "message": result.message or "Unknown error."}
