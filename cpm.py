"""
Critical Path Method (CPM) using a directed acyclic graph (DAG) built with NetworkX.
"""

from __future__ import annotations

from typing import Any, Hashable

import networkx as nx
import pandas as pd


REQUIRED_COLUMNS = (
    "task_id",
    "task_name",
    "duration",
    "dependencies",
    "resource_count",
    "task_type",
)

OPTIONAL_TRACKING_COLUMNS = (
    "baseline_start",
    "baseline_finish",
    "actual_start",
    "actual_finish",
    "progress_pct",
)


def parse_dependencies(raw: Any) -> list[Hashable]:
    """Parse dependency cell into a list of predecessor task ids."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    parts = []
    for token in s.replace(";", ",").split(","):
        t = token.strip()
        if not t:
            continue
        # Preserve numeric ids as int when possible
        try:
            if "." not in t:
                parts.append(int(t))
            else:
                parts.append(float(t))
        except ValueError:
            parts.append(t)
    return parts


def validate_dependency_references(df: pd.DataFrame) -> list[str]:
    """Return error messages for dependency ids that are not listed as task_id rows."""
    known = set(df["task_id"].tolist())
    errors: list[str] = []
    for _, row in df.iterrows():
        tid = row["task_id"]
        for pred in parse_dependencies(row["dependencies"]):
            if pred not in known:
                errors.append(
                    f"Task {tid!r} depends on {pred!r}, which is not a row in task_id."
                )
    return errors


def build_dag(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a precedence DAG: edge (pred -> task) means pred must finish before task starts.
    """
    g = nx.DiGraph()
    ids = df["task_id"].tolist()
    for tid in ids:
        g.add_node(tid)

    for _, row in df.iterrows():
        task_id = row["task_id"]
        for pred in parse_dependencies(row["dependencies"]):
            g.add_edge(pred, task_id)

    return g


def _topo_sort_or_raise(g: nx.DiGraph) -> list[Hashable]:
    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        raise ValueError(
            "Dependencies contain a cycle; CPM requires a DAG. "
            f"Example cycle: {cycles[0] if cycles else 'unknown'}"
        )
    return list(nx.topological_sort(g))


def compute_cpm(df: pd.DataFrame, g: nx.DiGraph | None = None) -> pd.DataFrame:
    """
    Compute ES, EF, LS, LF, slack, and is_critical for each task in df.

    Only rows whose task_id appears in df are returned with full metrics
    (subgraph used for scheduling is the DAG restricted to df task ids).
    """
    g = g if g is not None else build_dag(df)
    duration_map = df.set_index("task_id")["duration"].astype(float).to_dict()
    task_ids = list(df["task_id"])

    # Work on subgraph induced by known tasks only for scheduling
    sub = g.subgraph(task_ids).copy()
    order = _topo_sort_or_raise(sub)

    es: dict[Hashable, float] = {}
    ef: dict[Hashable, float] = {}

    for n in order:
        preds = list(sub.predecessors(n))
        d = float(duration_map.get(n, 0.0))
        if not preds:
            es[n] = 0.0
        else:
            es[n] = max(ef[p] for p in preds)
        ef[n] = es[n] + d

    if not ef:
        project_end = 0.0
    else:
        project_end = max(ef.values())

    ls: dict[Hashable, float] = {}
    lf: dict[Hashable, float] = {}

    for n in reversed(order):
        d = float(duration_map.get(n, 0.0))
        succs = list(sub.successors(n))
        if not succs:
            lf[n] = project_end
        else:
            lf[n] = min(ls[s] for s in succs)
        ls[n] = lf[n] - d

    rows = []
    for tid in task_ids:
        efi = ef[tid]
        esi = es[tid]
        lfi = lf[tid]
        lsi = ls[tid]
        slack = round(lfi - efi, 6)
        # numeric tolerance for float noise
        is_critical = abs(slack) < 1e-6
        rows.append(
            {
                "task_id": tid,
                "ES": esi,
                "EF": efi,
                "LS": lsi,
                "LF": lfi,
                "slack": slack,
                "is_critical": is_critical,
            }
        )

    return pd.DataFrame(rows)


def critical_path_task_ids(df: pd.DataFrame, cpm_df: pd.DataFrame | None = None) -> list[Hashable]:
    """Return task ids on the critical path (slack == 0), in topological order."""
    if cpm_df is None:
        cpm_df = compute_cpm(df)
    crit = cpm_df[cpm_df["is_critical"]]["task_id"].tolist()
    g = build_dag(df)
    sub = g.subgraph(crit).copy()
    if not crit:
        return []
    order = list(nx.topological_sort(sub))
    return [t for t in order if t in set(crit)]


def build_feature_frame(df: pd.DataFrame, cpm_df: pd.DataFrame) -> pd.DataFrame:
    """Merge base columns with CPM features: duration, resource_count, is_critical, slack."""
    base = df[
        ["task_id", "task_name", "duration", "resource_count", "task_type"]
    ].copy()
    base["duration"] = base["duration"].astype(float)
    base["resource_count"] = base["resource_count"].astype(int)
    merged = base.merge(
        cpm_df[["task_id", "slack", "is_critical"]],
        on="task_id",
        how="left",
    )
    return merged


def add_tracking_metrics(df: pd.DataFrame, cpm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add baseline-vs-actual schedule tracking columns.

    Falls back to CPM ES/EF when baseline fields are absent, so existing CSVs still work.
    Time units are numeric (project day / week unit, not datetime).
    """
    out = cpm_df[["task_id", "ES", "EF"]].merge(
        df,
        on="task_id",
        how="left",
    )

    for col in OPTIONAL_TRACKING_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["baseline_start"] = pd.to_numeric(out["baseline_start"], errors="coerce")
    out["baseline_finish"] = pd.to_numeric(out["baseline_finish"], errors="coerce")
    out["actual_start"] = pd.to_numeric(out["actual_start"], errors="coerce")
    out["actual_finish"] = pd.to_numeric(out["actual_finish"], errors="coerce")
    out["progress_pct"] = pd.to_numeric(out["progress_pct"], errors="coerce")

    # baseline fallback = deterministic CPM plan
    out["baseline_start"] = out["baseline_start"].fillna(out["ES"])
    out["baseline_finish"] = out["baseline_finish"].fillna(out["EF"])

    # derive missing actuals conservatively
    out["actual_start"] = out["actual_start"].fillna(out["baseline_start"])

    inferred_actual_finish = out["actual_start"] + out["duration"].astype(float)
    out["actual_finish"] = out["actual_finish"].fillna(inferred_actual_finish)

    out["progress_pct"] = out["progress_pct"].clip(lower=0, upper=100)

    out["start_variance"] = (out["actual_start"] - out["baseline_start"]).round(3)
    out["finish_variance"] = (out["actual_finish"] - out["baseline_finish"]).round(3)
    out["is_delayed_vs_baseline"] = out["finish_variance"] > 0
    return out


def tracking_summary(tracked_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate baseline vs actual schedule KPIs."""
    if tracked_df.empty:
        return {
            "planned_completion": 0.0,
            "actual_completion": 0.0,
            "project_delay": 0.0,
            "tasks_delayed_pct": 0.0,
        }

    planned_completion = float(tracked_df["baseline_finish"].max())
    actual_completion = float(tracked_df["actual_finish"].max())
    project_delay = actual_completion - planned_completion
    delayed_pct = float(tracked_df["is_delayed_vs_baseline"].mean() * 100)

    return {
        "planned_completion": round(planned_completion, 3),
        "actual_completion": round(actual_completion, 3),
        "project_delay": round(project_delay, 3),
        "tasks_delayed_pct": round(delayed_pct, 2),
    }
