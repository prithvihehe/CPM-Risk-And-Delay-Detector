# CPM-Risk-And-Delay-Detector

Streamlit dashboard for construction schedule analytics with:

- CPM (critical path) computation
- Rule-based + ML-assisted delay risk scoring
- **Baseline vs Actual** schedule tracking
- **PERT expected duration + Monte Carlo** completion risk simulation
- **Resource leveling** under limited resource capacity

## Input CSV schema

### Required columns

- `task_id`
- `task_name`
- `duration`
- `dependencies` (comma-separated predecessor ids)
- `resource_count`
- `task_type`

### Optional tracking columns (Feature #1)

- `baseline_start`
- `baseline_finish`
- `actual_start`
- `actual_finish`
- `progress_pct`

If omitted, baseline defaults to CPM ES/EF.

### Optional PERT columns (Feature #2)

- `optimistic_duration`
- `most_likely_duration`
- `pessimistic_duration`

If omitted, the app uses deterministic `duration` for all three.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## New professional planning outputs

1. **Baseline vs Actual KPIs**
   - Planned completion vs Actual completion
   - Project delay
   - % delayed tasks

2. **PERT + Monte Carlo**
   - P50 / P80 / P90 completion
   - Deadline hit probability
   - Completion-time distribution histogram

3. **Resource leveling (finite capacity)**
   - Constrained schedule (leveled start/finish)
   - Shift vs unconstrained CPM schedule
   - Resource utilization vs capacity chart
