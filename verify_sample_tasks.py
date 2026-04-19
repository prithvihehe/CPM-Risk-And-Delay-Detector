"""
Verify CPM outputs for sample_tasks.csv against hand-checked expectations.

Run from repo root:
  python verify_sample_tasks.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

from analysis import run_analysis
from cpm import build_dag, compute_cpm, critical_path_task_ids, validate_dependency_references

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "sample_tasks.csv"

# Expected schedule (floats match compute_cpm rounding for slack)
_EXPECTED: dict[int, dict[str, float | bool]] = {
    1: {"ES": 0.0, "EF": 2.0, "LS": 0.0, "LF": 2.0, "slack": 0.0, "is_critical": True},
    2: {"ES": 2.0, "EF": 6.0, "LS": 2.0, "LF": 6.0, "slack": 0.0, "is_critical": True},
    3: {"ES": 2.0, "EF": 5.0, "LS": 3.0, "LF": 6.0, "slack": 1.0, "is_critical": False},
    4: {"ES": 6.0, "EF": 11.0, "LS": 6.0, "LF": 11.0, "slack": 0.0, "is_critical": True},
    5: {"ES": 11.0, "EF": 12.0, "LS": 11.0, "LF": 12.0, "slack": 0.0, "is_critical": True},
    6: {"ES": 12.0, "EF": 15.0, "LS": 12.0, "LF": 15.0, "slack": 0.0, "is_critical": True},
    7: {"ES": 15.0, "EF": 16.0, "LS": 15.0, "LF": 16.0, "slack": 0.0, "is_critical": True},
    8: {"ES": 2.0, "EF": 4.5, "LS": 13.5, "LF": 16.0, "slack": 11.5, "is_critical": False},
}

_EXPECTED_CRITICAL_PATH = [1, 2, 4, 5, 6, 7]
_EXPECTED_PROJECT_END = 16.0


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6)


def verify_cpm_table(cpm: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    by_id = cpm.set_index("task_id")
    for tid, exp in _EXPECTED.items():
        if tid not in by_id.index:
            errors.append(f"Missing task_id {tid} in CPM results.")
            continue
        row = by_id.loc[tid]
        for key in ("ES", "EF", "LS", "LF", "slack"):
            got = float(row[key])
            want = float(exp[key])  # type: ignore[arg-type]
            if not _close(got, want):
                errors.append(
                    f"task {tid} {key}: expected {want}, got {got}"
                )
        if bool(row["is_critical"]) != bool(exp["is_critical"]):
            errors.append(
                f"task {tid} is_critical: expected {exp['is_critical']}, got {row['is_critical']}"
            )
    return errors


def main() -> int:
    if not SAMPLE.is_file():
        print(f"Missing {SAMPLE}", file=sys.stderr)
        return 1

    df = pd.read_csv(SAMPLE)
    miss = [c for c in ("task_id", "task_name", "duration", "dependencies", "resource_count", "task_type") if c not in df.columns]
    if miss:
        print(f"Missing columns: {miss}", file=sys.stderr)
        return 1

    dep_err = validate_dependency_references(df)
    if dep_err:
        print("Dependency validation failed:", dep_err[:5], file=sys.stderr)
        return 1

    g = build_dag(df)
    cpm_df = compute_cpm(df, g)
    cp = critical_path_task_ids(df, cpm_df)
    proj_end = float(cpm_df["EF"].max())

    errs = verify_cpm_table(cpm_df)
    if not _close(proj_end, _EXPECTED_PROJECT_END):
        errs.append(f"Project completion (max EF): expected {_EXPECTED_PROJECT_END}, got {proj_end}")
    if list(cp) != _EXPECTED_CRITICAL_PATH:
        errs.append(f"Critical path: expected {_EXPECTED_CRITICAL_PATH}, got {list(cp)}")

    ar = run_analysis(df, risk_threshold=1.0)
    if not ar.ok:
        errs.append(f"run_analysis failed: {ar.message or ar.missing_columns or ar.dependency_errors}")
    else:
        if ar.project_completion is not None and not _close(float(ar.project_completion), _EXPECTED_PROJECT_END):
            errs.append(
                f"run_analysis project_completion: expected {_EXPECTED_PROJECT_END}, got {ar.project_completion}"
            )
        if list(ar.critical_path) != _EXPECTED_CRITICAL_PATH:
            errs.append(
                f"run_analysis critical_path: expected {_EXPECTED_CRITICAL_PATH}, got {list(ar.critical_path)}"
            )

    if errs:
        print("Verification FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("Verification OK: sample_tasks.csv matches expected CPM schedule.")
    print(f"  Project completion (max EF): {proj_end}")
    print(f"  Critical path (ordered): {' -> '.join(str(x) for x in cp)}")
    print(f"  Tasks: {len(df)} rows (parallel branch, merge, float duration, linear tail).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
