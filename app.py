"""
Streamlit UI: CSV upload, CPM analysis, risk scoring, and high-risk highlights.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from cpm import (
    REQUIRED_COLUMNS,
    build_dag,
    build_feature_frame,
    compute_cpm,
    critical_path_task_ids,
    validate_dependency_references,
)
from pert import compute_pert_schedule
from resource_leveling import (
    level_resources,
    leveling_summary,
    smooth_resources,
    smoothing_summary,
)
from ai.explain import (
    explain_task,
    format_reasons_line,
    load_feature_importances,
    summarize_delay_risk,
)
from ai.predict import DelayPredictor
from risk import add_risk_scores, high_risk_mask
from visuals import (
    plot_aon,
    plot_explanation,
    plot_gantt,
    plot_resource_leveling_square_flow,
    plot_risk,
    risk_tier,
)

AI_FEATURES = ["duration", "resource_count", "is_critical", "slack"]


def _valid_delay_probability(p: object) -> bool:
    if p is None:
        return False
    if isinstance(p, float) and pd.isna(p):
        return False
    return True


def validate_df(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def style_high_risk(df: pd.DataFrame, mask: pd.Series) -> "pd.io.formats.style.Styler":
    m = mask.reindex(df.index, fill_value=False)

    def highlight_row(row: pd.Series) -> list[str]:
        if bool(m.loc[row.name]):
            return ["background-color: #ffcccc"] * len(row)
        return [""] * len(row)

    return df.style.apply(highlight_row, axis=1)


def main() -> None:
    st.set_page_config(page_title="CPM & Risk", layout="wide")
    st.title("Critical Path & Risk Analysis")
    st.caption(
        "Upload a CSV with columns: "
        + ", ".join(REQUIRED_COLUMNS)
        + " | Optional: "
        + ", ".join(["optimistic_duration", "most_likely_duration", "pessimistic_duration"])
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    threshold = st.slider("High-risk threshold (risk score)", 0.0, 2.0, 1.0, 0.1)
    time_unit = st.sidebar.selectbox("Time unit", ["days", "weeks", "months"], index=0)
    default_capacity = 1


    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    miss = validate_df(df)
    if miss:
        st.error(f"Missing required columns: {', '.join(miss)}")
        return

    dep_errors = validate_dependency_references(df)
    if dep_errors:
        for msg in dep_errors[:20]:
            st.error(msg)
        if len(dep_errors) > 20:
            st.error(f"... and {len(dep_errors) - 20} more dependency errors.")
        return

    st.subheader("Uploaded data")
    st.dataframe(df, use_container_width=True)

    try:
        g = build_dag(df)
        cpm_df = compute_cpm(df, g)
    except ValueError as e:
        st.error(str(e))
        return

    cp_path = critical_path_task_ids(df, cpm_df)
    features = build_feature_frame(df, cpm_df)
    scored = add_risk_scores(features)

    try:
        predictor = DelayPredictor()
        scored["delay_probability"] = predictor.predict_batch(scored[AI_FEATURES])
    except FileNotFoundError as e:
        st.warning(str(e))
        scored["delay_probability"] = float("nan")

    importance_error: str | None = None
    try:
        feature_importances = load_feature_importances()
    except FileNotFoundError as e:
        feature_importances = None
        importance_error = str(e)

    task_insights: list[dict] = []
    explain_lines: list[str] = []
    summaries: list[str] = []

    for _, row in scored.iterrows():
        tid = row["task_id"]
        dp = row["delay_probability"]
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
            reasons = []
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

    hr = high_risk_mask(scored["risk_score"], threshold=threshold)

    st.subheader("CPM results")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Project completion (max EF)", f"{cpm_df['EF'].max():.2f} {time_unit}")
    with c2:
        st.write("**Critical path (ordered):**")
        st.write(" → ".join(str(x) for x in cp_path) if cp_path else "(none)")

    st.dataframe(
        cpm_df.merge(df[["task_id", "task_name"]], on="task_id", how="left"),
        use_container_width=True,
    )

    st.subheader("AON dependency diagram")
    st.plotly_chart(plot_aon(df, g, cpm_df), use_container_width=True)

    st.subheader("PERT schedule")
    pert_input, pert_cpm = compute_pert_schedule(df)
    pert_completion = float(pert_cpm["EF"].max())
    pert_merged = pert_input.merge(pert_cpm[["task_id", "is_critical"]], on="task_id", how="left")
    project_variance = float(pert_merged.loc[pert_merged["is_critical"], "pert_variance"].sum())
    project_sigma = project_variance ** 0.5
    approx_p80 = pert_completion + 0.8416 * project_sigma

    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("PERT expected completion", f"{pert_completion:.2f} {time_unit}")
    with p2:
        st.metric("PERT σ (critical path)", f"{project_sigma:.2f} {time_unit}")
    with p3:
        st.metric("Approx P80 completion", f"{approx_p80:.2f} {time_unit}")

    st.caption(
        "PERT uses expected duration per task: (Optimistic + 4×MostLikely + Pessimistic) / 6. "
        "Approx P80 shown using normal approximation on summed critical-path variance."
    )
    st.dataframe(
        pert_input[
            [
                "task_id",
                "task_name",
                "duration",
                "optimistic_duration",
                "most_likely_duration",
                "pessimistic_duration",
                "pert_expected_duration",
                "pert_variance",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Resource planning")
    default_capacity = max(1, int(df["resource_count"].max()))
    max_capacity = max(default_capacity, int(df["resource_count"].sum()))
    capacity = st.slider("Available resource capacity per time unit", 1, max_capacity, default_capacity)
    planning_mode = st.radio(
        "Planning mode",
        options=["Resource levelling", "Resource smoothing"],
        horizontal=True,
    )

    if planning_mode == "Resource levelling":
        leveled_df, usage_df = level_resources(df, cpm_df, resource_capacity=capacity)
        lsum = leveling_summary(leveled_df)
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            st.metric("CPM completion", f"{lsum['cpm_completion']:.2f} {time_unit}")
        with l2:
            st.metric("Leveled completion", f"{lsum['leveled_completion']:.2f} {time_unit}")
        with l3:
            st.metric("Leveling delay", f"{lsum['resource_leveling_delay']:.2f} {time_unit}")
        with l4:
            st.metric("Avg start shift", f"{lsum['avg_start_shift']:.2f} {time_unit}")

        st.plotly_chart(
            px.line(
                usage_df,
                x="time",
                y=["used_resources", "capacity"],
                title="Resource usage vs capacity (levelling)",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_resource_leveling_square_flow(
                df,
                leveled_df,
                time_unit=time_unit,
                show_original=False,
                mode_label="Resource levelling",
                start_col="leveled_start",
                finish_col="leveled_finish",
            ),
            use_container_width=True,
        )
        st.dataframe(
            leveled_df[
                [
                    "task_id",
                    "task_name",
                    "resource_count",
                    "ES",
                    "EF",
                    "leveled_start",
                    "leveled_finish",
                    "start_shift_vs_cpm",
                    "finish_shift_vs_cpm",
                ]
            ],
            use_container_width=True,
        )
    else:
        smoothed_df, usage_df = smooth_resources(df, cpm_df, resource_capacity=capacity)
        ssum = smoothing_summary(smoothed_df)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("CPM completion", f"{ssum['cpm_completion']:.2f} {time_unit}")
        with s2:
            st.metric("Smoothed completion", f"{ssum['smoothed_completion']:.2f} {time_unit}")
        with s3:
            st.metric("Completion change", f"{ssum['completion_change']:.2f} {time_unit}")
        with s4:
            st.metric("Avg start shift", f"{ssum['avg_start_shift']:.2f} {time_unit}")

        st.caption("Resource smoothing mode shown separately (no comparison diagram).")
        st.plotly_chart(
            px.line(
                usage_df,
                x="time",
                y=["used_resources", "capacity"],
                title="Resource usage vs capacity (smoothing)",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_resource_leveling_square_flow(
                df,
                smoothed_df,
                time_unit=time_unit,
                show_original=False,
                mode_label="Resource smoothing",
                start_col="smooth_start",
                finish_col="smooth_finish",
            ),
            use_container_width=True,
        )
        st.dataframe(
            smoothed_df[
                [
                    "task_id",
                    "task_name",
                    "resource_count",
                    "ES",
                    "EF",
                    "smooth_start",
                    "smooth_finish",
                    "start_shift_vs_cpm",
                    "finish_shift_vs_cpm",
                ]
            ],
            use_container_width=True,
        )

    st.subheader("Features, risk scores & AI delay probability")
    st.caption(
        "delay_probability comes from a RandomForest trained offline (see ai/). "
        "The app does not retrain the model."
    )
    if importance_error:
        st.warning(importance_error)

    base_display = [c for c in scored.columns if c not in ("delay_probability", "delay_explanation", "delay_summary")]
    display_cols = base_display + ["delay_probability", "delay_explanation", "delay_summary"]
    st.dataframe(
        style_high_risk(scored[display_cols], hr),
        use_container_width=True,
        column_config={
            "delay_probability": st.column_config.NumberColumn(
                "delay_probability",
                format="%.2f",
            ),
        },
    )

    st.subheader("Why might tasks be delayed?")
    st.caption(
        "Rule-based drivers (critical path, duration, resources, slack) weighted by "
        "model feature importances from ai/feature_importance.json."
    )
    with st.expander("Structured explanation (JSON per task)", expanded=False):
        st.json(task_insights)

    tasks: list[dict] = []
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

    st.subheader("Project Timeline (Gantt Chart)")
    st.plotly_chart(
        plot_gantt(tasks, ES, EF, critical),
        use_container_width=True,
    )

    st.subheader("Task Risk Levels")
    st.plotly_chart(plot_risk(tasks), use_container_width=True)

    st.subheader("Top 5 risky tasks (by delay probability)")
    top5 = sorted(
        [t for t in tasks if _valid_delay_probability(t.get("delay_probability"))],
        key=lambda t: float(t["delay_probability"]),
        reverse=True,
    )[:5]
    if top5:
        top_df = pd.DataFrame(
            [
                {
                    "task_id": t["id"],
                    "task_name": t["name"],
                    "delay_probability": f"{float(t['delay_probability']):.2f}",
                    "risk_level": risk_tier(float(t["delay_probability"])),
                }
                for t in top5
            ]
        )
        st.dataframe(top_df, use_container_width=True)
    else:
        st.info("No delay probabilities available for ranking (train the AI model first).")

    st.subheader("Explain risk for a task")
    task_ids = [t["id"] for t in tasks]
    selected_id = st.selectbox("Select Task", task_ids, index=0)
    selected = next(t for t in tasks if t["id"] == selected_id)
    dp = selected.get("delay_probability")
    if _valid_delay_probability(dp):
        st.caption(
            f"delay_probability = {float(dp):.2f} ({risk_tier(float(dp))})"
        )
    else:
        st.caption("delay_probability unavailable for this run.")
    st.plotly_chart(
        plot_explanation(selected["reasons"]),
        use_container_width=True,
    )

    high = scored.loc[hr].copy()
    st.subheader("High-risk tasks")
    if high.empty:
        st.success(f"No tasks at or above risk score {threshold}.")
    else:
        st.warning(
            f"{len(high)} task(s) with risk score ≥ {threshold} (highlighted in the table above)."
        )
        st.dataframe(high, use_container_width=True)


if __name__ == "__main__":
    main()
