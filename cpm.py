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
