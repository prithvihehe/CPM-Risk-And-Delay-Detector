"""
Streamlit UI: CSV upload, CPM analysis, risk scoring, and high-risk highlights.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis import run_analysis
from cpm import REQUIRED_COLUMNS
from visuals import plot_explanation, plot_gantt, plot_risk, risk_tier


def _valid_delay_probability(p: object) -> bool:
    if p is None:
        return False
    if isinstance(p, float) and pd.isna(p):
        return False
    return True


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
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    threshold = st.slider("High-risk threshold (risk score)", 0.0, 2.0, 1.0, 0.1)

    if uploaded is None:
        st.info("Upload a CSV to begin.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    result = run_analysis(df, risk_threshold=threshold)
    if not result.ok:
        if result.missing_columns:
            st.error(f"Missing required columns: {', '.join(result.missing_columns)}")
            return
        if result.dependency_errors:
            for msg in result.dependency_errors[:20]:
                st.error(msg)
            if len(result.dependency_errors) > 20:
                st.error(f"... and {len(result.dependency_errors) - 20} more dependency errors.")
            return
        st.error(result.message or "Analysis failed.")
        return

    assert result.scored is not None
    assert result.cpm_df is not None
    assert result.high_risk_mask is not None

    scored = result.scored
    cpm_df = result.cpm_df
    hr = result.high_risk_mask
    task_insights = result.task_insights
    tasks = result.tasks

    for w in result.warnings:
        st.warning(w)

    st.subheader("Uploaded data")
    st.dataframe(df, use_container_width=True)

    st.subheader("CPM results")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Project completion (max EF)", f"{result.project_completion:.2f}")
    with c2:
        st.write("**Critical path (ordered):**")
        st.write(
            " → ".join(str(x) for x in result.critical_path)
            if result.critical_path
            else "(none)"
        )

    st.dataframe(
        cpm_df.merge(df[["task_id", "task_name"]], on="task_id", how="left"),
        use_container_width=True,
    )

    st.subheader("Features, risk scores & AI delay probability")
    st.caption(
        "delay_probability comes from a RandomForest trained offline (see ai/). "
        "The app does not retrain the model."
    )
    if result.importance_error:
        st.warning(result.importance_error)

    base_display = [
        c
        for c in scored.columns
        if c not in ("delay_probability", "delay_explanation", "delay_summary")
    ]
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

    ES = cpm_df.set_index("task_id")["ES"].to_dict()
    EF = cpm_df.set_index("task_id")["EF"].to_dict()
    critical = cpm_df.set_index("task_id")["is_critical"].to_dict()

    st.subheader("Project Timeline (Gantt Chart)")
    st.plotly_chart(
        plot_gantt(tasks, ES, EF, critical),
        use_container_width=True,
    )

    st.subheader("Task Risk Levels")
    st.plotly_chart(
        plot_risk(tasks),
        use_container_width=True,
    )

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
