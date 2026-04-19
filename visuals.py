"""
Plotly visualizations for CPM timeline, delay risk, and explanation factors.

Keeps chart construction separate from CPM / ML logic.
"""

from __future__ import annotations

import math
from typing import Any, Hashable

import pandas as pd
import plotly.graph_objects as go


def _task_label(task: dict[str, Any]) -> str:
    if "name" in task and str(task["name"]).strip():
        return f"{task['id']} — {task['name']}"
    return str(task["id"])


def risk_tier(delay_probability: float | None) -> str:
    """Optional UI tiers for delay probability."""
    if delay_probability is None or (isinstance(delay_probability, float) and math.isnan(delay_probability)):
        return "N/A"
    p = float(delay_probability)
    if p > 0.7:
        return "HIGH"
    if p >= 0.4:
        return "MEDIUM"
    return "LOW"


def risk_tier_color(tier: str) -> str:
    if tier == "HIGH":
        return "#c0392b"
    if tier == "MEDIUM":
        return "#e67e22"
    if tier == "LOW":
        return "#27ae60"
    return "#7f8c8d"


def plot_gantt(
    tasks: list[dict[str, Any]],
    ES: dict[Hashable, float],
    EF: dict[Hashable, float],
    critical: dict[Hashable, bool],
) -> go.Figure:
    """
    Timeline (Gantt-style) bar chart: ES → EF per task, colored by critical flag.
    Y-axis is reversed so the first task in `tasks` appears at the top.
    """
    if not tasks:
        fig = go.Figure()
        fig.update_layout(title="No tasks to display", height=240)
        return fig

    y_labels = [_task_label(t) for t in tasks]
    span = [float(EF[t["id"]]) - float(ES[t["id"]]) for t in tasks]
    base = [float(ES[t["id"]]) for t in tasks]
    bar_colors = [
        "#c0392b" if bool(critical.get(t["id"], False)) else "#95a5a6" for t in tasks
    ]

    fig = go.Figure(
        go.Bar(
            orientation="h",
            y=y_labels,
            x=span,
            base=base,
            marker_color=bar_colors,
            showlegend=False,
        )
    )
    # Legend entries (invisible markers) for critical vs non-critical colors
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color="#c0392b"),
            name="Critical (Yes)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color="#95a5a6"),
            name="Critical (No)",
        )
    )

    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="Time")
    fig.update_layout(
        title="Project timeline",
        height=max(320, 44 * len(tasks)),
        margin=dict(l=8, r=8, t=48, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_risk(tasks: list[dict[str, Any]]) -> go.Figure:
    """
    Bar chart: task vs delay_probability, sorted highest risk first.
    Colors optional tiers: HIGH / MEDIUM / LOW.
    """
    if not tasks:
        fig = go.Figure()
        fig.update_layout(title="No tasks to display", height=240)
        return fig

    rows = []
    for t in tasks:
        dp = t.get("delay_probability")
        if dp is not None and not (isinstance(dp, float) and math.isnan(dp)):
            p = float(dp)
        else:
            p = float("nan")
        tier = risk_tier(p if not math.isnan(p) else None)
        rows.append(
            {
                "Task": _task_label(t),
                "delay_probability": p,
                "tier": tier,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("delay_probability", ascending=False, na_position="last")

    colors = [risk_tier_color(str(x)) for x in df["tier"]]
    fig = go.Figure(
        go.Bar(
            x=df["delay_probability"],
            y=df["Task"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" if not math.isnan(v) else "—" for v in df["delay_probability"]],
            textposition="outside",
        )
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="Delay probability", range=[0, 1.05])
    fig.update_layout(
        title="Task delay risk (AI)",
        height=max(320, 40 * len(df)),
        margin=dict(l=8, r=48, t=48, b=8),
        showlegend=False,
    )
    return fig


def plot_explanation(reasons: list[tuple[str, float]]) -> go.Figure:
    """Horizontal bar chart: explanation factor vs importance weight."""
    if not reasons:
        fig = go.Figure()
        fig.update_layout(
            title="No explanation factors for this task",
            height=260,
            annotations=[
                dict(
                    text="Rules did not fire or importances unavailable.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
        )
        return fig

    factors = [r[0] for r in reasons]
    impacts = [float(r[1]) for r in reasons]
    df = pd.DataFrame({"Factor": factors, "Impact": impacts})
    df = df.sort_values("Impact", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["Impact"],
            y=df["Factor"],
            orientation="h",
            marker=dict(
                color=df["Impact"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Weight"),
            ),
        )
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="Model feature importance weight")
    fig.update_layout(
        title="Why this task may be risky",
        height=max(280, 44 * len(df)),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    return fig
