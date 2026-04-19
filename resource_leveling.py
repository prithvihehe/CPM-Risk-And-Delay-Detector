"""
Simple resource-leveling heuristic scheduler.
"""

from __future__ import annotations

import networkx as nx

import pandas as pd

from cpm import build_dag


def _resource_usage_at(schedule: pd.DataFrame, time_point: float, start_col: str, finish_col: str) -> float:
    active = (schedule[start_col] <= time_point + 1e-9) & (schedule[finish_col] > time_point + 1e-9)
    if not active.any():
        return 0.0
    return float(schedule.loc[active, "resource_count"].sum())


def build_resource_usage(
    schedule_df: pd.DataFrame,
    *,
    start_col: str,
    finish_col: str,
    capacity: int,
) -> pd.DataFrame:
    """Build a step-like resource usage table at schedule event points."""
    points = sorted(
        set(schedule_df[start_col].astype(float).tolist())
        | set(schedule_df[finish_col].astype(float).tolist())
    )
    if not points:
        return pd.DataFrame(columns=["time", "used_resources", "capacity"])

    rows: list[dict[str, float]] = []
    for t in points:
        rows.append(
            {
                "time": float(t),
                "used_resources": _resource_usage_at(schedule_df, float(t), start_col, finish_col),
                "capacity": float(capacity),
            }
        )
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def level_resources(
    df: pd.DataFrame,
    cpm_df: pd.DataFrame,
    *,
    resource_capacity: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Serial schedule generation with finite resource capacity.

    Priority rule:
      - critical first
      - lower slack first
      - longer duration first
    """
    if resource_capacity <= 0:
        raise ValueError("resource_capacity must be > 0")

    merged = df.merge(
        cpm_df[["task_id", "slack", "is_critical", "ES", "EF"]],
        on="task_id",
        how="left",
    ).copy()

    g = build_dag(df)
    info = merged.set_index("task_id").to_dict("index")

    unscheduled = set(merged["task_id"].tolist())
    running: list[dict] = []
    starts: dict = {}
    finishes: dict = {}

    t = 0.0
    timeline_usage: list[dict[str, float]] = []

    while unscheduled or running:
        # finish completed tasks
        still_running: list[dict] = []
        for r in running:
            if r["finish"] <= t + 1e-9:
                finishes[r["task_id"]] = r["finish"]
            else:
                still_running.append(r)
        running = still_running

        used = sum(int(r["resource_count"]) for r in running)

        # candidates that have all predecessors finished and not started
        ready = []
        for tid in list(unscheduled):
            preds = list(g.predecessors(tid))
            if all(p in finishes for p in preds):
                row = info[tid]
                ready.append(
                    {
                        "task_id": tid,
                        "duration": float(row["duration"]),
                        "resource_count": int(row["resource_count"]),
                        "is_critical": bool(row["is_critical"]),
                        "slack": float(row["slack"]),
                    }
                )

        # priority ordering
        ready.sort(
            key=lambda x: (
                not x["is_critical"],
                x["slack"],
                -x["duration"],
            )
        )

        started_any = False
        for cand in ready:
            req = cand["resource_count"]
            if used + req <= resource_capacity:
                start = t
                finish = t + cand["duration"]
                running.append(
                    {
                        "task_id": cand["task_id"],
                        "start": start,
                        "finish": finish,
                        "resource_count": req,
                    }
                )
                starts[cand["task_id"]] = start
                unscheduled.remove(cand["task_id"])
                used += req
                started_any = True

        timeline_usage.append(
            {
                "time": t,
                "used_resources": float(used),
                "capacity": float(resource_capacity),
            }
        )

        if unscheduled or running:
            if running:
                next_finish = min(r["finish"] for r in running)
                # avoid infinite loops on tiny float errors
                t = max(t + 1e-6, float(next_finish)) if not started_any else t
                if started_any:
                    # if we started tasks this tick, keep t to allow immediate packing
                    pass
            else:
                # no running tasks, jump minimally
                t += 1.0

            if started_any:
                # progress time only when no further tasks can be started this instant
                used_now = sum(int(r["resource_count"]) for r in running)
                ready_now = []
                for tid in list(unscheduled):
                    preds = list(g.predecessors(tid))
                    if all(p in finishes for p in preds):
                        ready_now.append(tid)
                if not ready_now or used_now >= resource_capacity:
                    t = min(float(r["finish"]) for r in running)

    schedule = merged.copy()
    schedule["leveled_start"] = schedule["task_id"].map(starts).astype(float)
    schedule["leveled_finish"] = schedule["task_id"].map(finishes).astype(float)
    schedule["start_shift_vs_cpm"] = (schedule["leveled_start"] - schedule["ES"]).round(3)
    schedule["finish_shift_vs_cpm"] = (schedule["leveled_finish"] - schedule["EF"]).round(3)

    usage = pd.DataFrame(timeline_usage).drop_duplicates(subset=["time"]).sort_values("time")
    return schedule, usage


def leveling_summary(schedule_df: pd.DataFrame) -> dict[str, float]:
    if schedule_df.empty:
        return {
            "cpm_completion": 0.0,
            "leveled_completion": 0.0,
            "resource_leveling_delay": 0.0,
            "avg_start_shift": 0.0,
        }

    cpm_completion = float(schedule_df["EF"].max())
    leveled_completion = float(schedule_df["leveled_finish"].max())
    return {
        "cpm_completion": round(cpm_completion, 3),
        "leveled_completion": round(leveled_completion, 3),
        "resource_leveling_delay": round(leveled_completion - cpm_completion, 3),
        "avg_start_shift": round(float(schedule_df["start_shift_vs_cpm"].mean()), 3),
    }


def smooth_resources(
    df: pd.DataFrame,
    cpm_df: pd.DataFrame,
    *,
    resource_capacity: int,
    time_step: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resource smoothing heuristic:
    - keeps tasks within CPM [ES, LS] where possible (uses float/slack)
    - tries to reduce peak overlap while preserving CPM completion target
    """
    if resource_capacity <= 0:
        raise ValueError("resource_capacity must be > 0")

    merged = df.merge(
        cpm_df[["task_id", "ES", "EF", "LS", "LF", "slack", "is_critical"]],
        on="task_id",
        how="left",
    ).copy()
    merged["duration"] = merged["duration"].astype(float)
    merged["resource_count"] = merged["resource_count"].astype(int)

    g = build_dag(df)
    order = list(nx.topological_sort(g))
    info = merged.set_index("task_id").to_dict("index")

    scheduled_rows: list[dict] = []
    starts: dict = {}
    finishes: dict = {}

    for tid in order:
        row = info[tid]
        preds = list(g.predecessors(tid))
        pred_finish = max((finishes[p] for p in preds), default=float(row["ES"]))

        earliest = max(float(row["ES"]), float(pred_finish))
        latest = float(row["LS"])
        duration = float(row["duration"])

        if latest < earliest:
            candidate_starts = [earliest]
        else:
            n_steps = int(round((latest - earliest) / max(time_step, 1e-6)))
            candidate_starts = [earliest + i * time_step for i in range(max(0, n_steps) + 1)]
            if candidate_starts and candidate_starts[-1] < latest - 1e-9:
                candidate_starts.append(latest)

        best_start = earliest
        best_score = float("inf")

        provisional = pd.DataFrame(scheduled_rows)
        for s in candidate_starts:
            f = s + duration
            if preds and any(finishes[p] > s + 1e-9 for p in preds):
                continue

            trial = provisional.copy()
            trial = pd.concat(
                [
                    trial,
                    pd.DataFrame(
                        [
                            {
                                "task_id": tid,
                                "resource_count": int(row["resource_count"]),
                                "smooth_start": float(s),
                                "smooth_finish": float(f),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            usage = build_resource_usage(
                trial,
                start_col="smooth_start",
                finish_col="smooth_finish",
                capacity=resource_capacity,
            )
            if usage.empty:
                score = 0.0
            else:
                peak = float(usage["used_resources"].max())
                over_cap = float((usage["used_resources"] - resource_capacity).clip(lower=0).sum())
                score = 1000.0 * over_cap + peak + 0.01 * float(s)

            if score < best_score:
                best_score = score
                best_start = float(s)

        best_finish = best_start + duration
        starts[tid] = best_start
        finishes[tid] = best_finish
        scheduled_rows.append(
            {
                "task_id": tid,
                "resource_count": int(row["resource_count"]),
                "smooth_start": best_start,
                "smooth_finish": best_finish,
            }
        )

    out = merged.copy()
    out["smooth_start"] = out["task_id"].map(starts).astype(float)
    out["smooth_finish"] = out["task_id"].map(finishes).astype(float)
    out["start_shift_vs_cpm"] = (out["smooth_start"] - out["ES"]).round(3)
    out["finish_shift_vs_cpm"] = (out["smooth_finish"] - out["EF"]).round(3)

    usage = build_resource_usage(
        out,
        start_col="smooth_start",
        finish_col="smooth_finish",
        capacity=resource_capacity,
    )
    return out, usage


def smoothing_summary(schedule_df: pd.DataFrame) -> dict[str, float]:
    if schedule_df.empty:
        return {
            "cpm_completion": 0.0,
            "smoothed_completion": 0.0,
            "completion_change": 0.0,
            "avg_start_shift": 0.0,
        }

    cpm_completion = float(schedule_df["EF"].max())
    smoothed_completion = float(schedule_df["smooth_finish"].max())
    return {
        "cpm_completion": round(cpm_completion, 3),
        "smoothed_completion": round(smoothed_completion, 3),
        "completion_change": round(smoothed_completion - cpm_completion, 3),
        "avg_start_shift": round(float(schedule_df["start_shift_vs_cpm"].mean()), 3),
    }
