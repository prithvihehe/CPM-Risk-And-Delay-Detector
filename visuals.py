"""
Plotly visualizations for CPM timeline, delay risk, and explanation factors.

Keeps chart construction separate from CPM / ML logic.
"""

from __future__ import annotations

import math
from typing import Any, Hashable

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cpm import parse_dependencies


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


def plot_aon(
    df: pd.DataFrame,
    graph: nx.DiGraph,
    cpm_df: pd.DataFrame,
) -> go.Figure:
    """
    Activity-on-Node (AON) dependency diagram.
    - Node = task
    - Arrow/edge = precedence relation (pred -> task)
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No tasks to display", height=240)
        return fig

    name_map = df.set_index("task_id")["task_name"].astype(str).to_dict()
    duration_map = df.set_index("task_id")["duration"].astype(float).to_dict()
    cpm_map = cpm_df.set_index("task_id").to_dict("index")

    order = list(nx.topological_sort(graph))
    level: dict[Hashable, int] = {}
    for node in order:
        preds = list(graph.predecessors(node))
        level[node] = 0 if not preds else max(level[p] + 1 for p in preds)

    by_level: dict[int, list[Hashable]] = {}
    for n, lv in level.items():
        by_level.setdefault(lv, []).append(n)

    pos: dict[Hashable, tuple[float, float]] = {}
    for lv, nodes in sorted(by_level.items(), key=lambda x: x[0]):
        nodes = sorted(nodes, key=lambda x: str(x))
        m = len(nodes)
        for i, n in enumerate(nodes):
            y = (m - 1) / 2.0 - i
            pos[n] = (float(lv), float(y))

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_label_x: list[float] = []
    edge_label_y: list[float] = []
    edge_labels: list[str] = []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dur_v = float(duration_map.get(v, 0.0))

        # If an edge skips over one or more levels and is nearly horizontal,
        # route it with a small bend so it doesn't visually disappear behind
        # intermediate nodes (e.g., dependency 2 -> 4 when node 3 is in between).
        skips_levels = (x1 - x0) > 1.01
        nearly_horizontal = abs(y1 - y0) < 1e-9
        if skips_levels and nearly_horizontal:
            bend = 0.55
            edge_x.extend([x0, x0 + 0.35, x1 - 0.35, x1, None])
            edge_y.extend([y0, y0 + bend, y1 + bend, y1, None])
            edge_label_x.append((x0 + x1) / 2.0)
            edge_label_y.append(y0 + bend + 0.06)
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_label_x.append((x0 + x1) / 2.0)
            edge_label_y.append((y0 + y1) / 2.0 + 0.06)
        edge_labels.append(f"d={dur_v:.2f}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#8395a7"),
        hoverinfo="skip",
        name="Dependencies",
    )

    node_x: list[float] = []
    node_y: list[float] = []
    labels: list[str] = []
    hovers: list[str] = []
    colors: list[str] = []

    for n in order:
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        nm = name_map.get(n, str(n))
        labels.append(f"{n}<br>{nm}")
        cpm_row = cpm_map.get(n, {})
        es = float(cpm_row.get("ES", 0.0))
        ef = float(cpm_row.get("EF", 0.0))
        sl = float(cpm_row.get("slack", 0.0))
        critical = bool(cpm_row.get("is_critical", False))
        dur = float(duration_map.get(n, 0.0))
        hovers.append(
            f"Task: {n} — {nm}<br>Duration: {dur:.2f}<br>ES: {es:.2f}<br>EF: {ef:.2f}<br>Slack: {sl:.2f}<br>Critical: {'Yes' if critical else 'No'}"
        )
        colors.append("#c0392b" if critical else "#3498db")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        hovertext=hovers,
        hoverinfo="text",
        marker=dict(size=28, color=colors, line=dict(color="#2c3e50", width=1)),
        showlegend=False,
    )

    edge_label_trace = go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        mode="text",
        text=edge_labels,
        textposition="top center",
        textfont=dict(size=10, color="#34495e"),
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace])
    fig.update_layout(
        title="AON (Activity-on-Node) dependency diagram",
        xaxis=dict(title="Dependency level", showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=max(360, 80 * max(1, len(by_level))),
        margin=dict(l=8, r=8, t=50, b=8),
    )
    return fig


def plot_resource_leveling_square_flow(
    df: pd.DataFrame,
    scheduled_df: pd.DataFrame,
    *,
    time_unit: str = "time",
    show_original: bool = False,
    mode_label: str = "Resource levelling",
    start_col: str = "leveled_start",
    finish_col: str = "leveled_finish",
) -> go.Figure:
    """
    Square-style resource-allocation network diagram.

    Top row (optional): original CPM timeline
    Bottom (or only) row: selected resource-constrained timeline
    Right-angle connectors visualize predecessor links.
    """
    if df.empty or scheduled_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No tasks to display", height=240)
        return fig

    sched = scheduled_df.merge(
        df[["task_id", "task_name", "dependencies"]],
        on=["task_id", "task_name"],
        how="left",
    ).copy()
    sched = sched.sort_values(["ES", "task_id"]).reset_index(drop=True)

    lanes = {tid: float(i) for i, tid in enumerate(sched["task_id"].tolist())}
    lane_labels = {tid: f"{tid} — {nm}" for tid, nm in zip(sched["task_id"], sched["task_name"])}

    if show_original:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("Original schedule (CPM)", f"{mode_label} schedule"),
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
            subplot_titles=(f"{mode_label} schedule",),
        )

    def _task_bar(row: pd.Series, *, start_col: str, end_col: str, row_num: int, color: str) -> None:
        tid = row["task_id"]
        y = lanes[tid]
        xs = [float(row[start_col]), float(row[end_col])]
        ys = [y, y]
        dur = float(row[end_col]) - float(row[start_col])
        rcount = int(row["resource_count"])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers+text",
                line=dict(width=4, color=color),
                marker=dict(size=8, color=color),
                text=[str(tid), f"d={dur:.2f}, r={rcount}"],
                textposition="top center",
                hovertemplate=(
                    f"Task {tid} — {row['task_name']}<br>"
                    + f"Resources: {rcount}<br>"
                    + f"Start: %{{x:.2f}} {time_unit}<br>"
                    + f"Finish: {float(row[end_col]):.2f} {time_unit}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row_num,
            col=1,
        )

    def _dependency_links(*, start_col: str, end_col: str, row_num: int) -> None:
        info = sched.set_index("task_id")
        for _, r in sched.iterrows():
            succ = r["task_id"]
            deps = parse_dependencies(r.get("dependencies"))
            for pred in deps:
                if pred not in info.index:
                    continue
                pred_row = info.loc[pred]
                x0 = float(pred_row[end_col])
                y0 = lanes[pred]
                x1 = float(r[start_col])
                y1 = lanes[succ]

                # right-angled (square) connector: horizontal then vertical
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, x1],
                        y=[y0, y0, y1],
                        mode="lines",
                        line=dict(
                            width=1.8,
                            color="#7f8c8d",
                            dash="dash" if x1 - x0 > 1e-9 else "solid",
                        ),
                        hovertemplate=f"Dependency: {pred} → {succ}<extra></extra>",
                        showlegend=False,
                    ),
                    row=row_num,
                    col=1,
                )

    if show_original:
        for _, r in sched.iterrows():
            _task_bar(r, start_col="ES", end_col="EF", row_num=1, color="#2980b9")
            _task_bar(r, start_col=start_col, end_col=finish_col, row_num=2, color="#27ae60")
        _dependency_links(start_col="ES", end_col="EF", row_num=1)
        _dependency_links(start_col=start_col, end_col=finish_col, row_num=2)
    else:
        for _, r in sched.iterrows():
            _task_bar(r, start_col=start_col, end_col=finish_col, row_num=1, color="#27ae60")
        _dependency_links(start_col=start_col, end_col=finish_col, row_num=1)

    tickvals = [lanes[tid] for tid in sched["task_id"]]
    ticktext = [lane_labels[tid] for tid in sched["task_id"]]

    fig.update_yaxes(
        tickvals=tickvals,
        ticktext=ticktext,
        title="Activities",
        autorange="reversed",
        row=1,
        col=1,
    )
    if show_original:
        fig.update_yaxes(
            tickvals=tickvals,
            ticktext=ticktext,
            title="Activities",
            autorange="reversed",
            row=2,
            col=1,
        )
        fig.update_xaxes(title=f"Time ({time_unit})", row=2, col=1)
    else:
        fig.update_xaxes(title=f"Time ({time_unit})", row=1, col=1)
    fig.update_layout(
        title=f"{mode_label} square network / allocation flow",
        height=max((620 if show_original else 380), 70 * len(sched) + (240 if show_original else 120)),
        margin=dict(l=8, r=8, t=70, b=8),
    )
    return fig
