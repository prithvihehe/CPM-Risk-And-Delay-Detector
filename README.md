# CPM Risk & Delay Detector

Analyze project schedules from a CSV: run **Critical Path Method (CPM)**, **rule-based risk scores**, and optional **machine-learned delay probabilities** (RandomForest trained offline). Two interfaces are included: a **Streamlit** app and a **FastAPI** web UI with Plotly charts.

## Features

- **CPM**: Early/late start and finish, slack, critical path (NetworkX DAG).
- **Risk scoring**: Heuristic score from duration, resources, criticality, and slack (threshold highlights “high risk” tasks).
- **Delay probability**: Batch predictions from `ai/model.pkl` when present; explanations combine rule-based drivers with feature importances from `ai/feature_importance.json`.
- **Charts**: Gantt-style timeline, delay-risk bars, dependency graph, per-task explanation chart (Plotly).
- **Verification**: `verify_sample_tasks.py` checks CPM outputs for `sample_tasks.csv`.

## Requirements

- Python 3.10+ recommended  
- Install dependencies:

```bash
pip install -r requirements.txt
```

## CSV format

Each row is one task. Required columns:

| Column | Description |
|--------|-------------|
| `task_id` | Unique task identifier (number or string consistent with dependencies). |
| `task_name` | Display name. |
| `duration` | Duration (numeric; float allowed). |
| `dependencies` | Comma- or semicolon-separated predecessor `task_id` values; empty if none. |
| `resource_count` | Integer used for risk features and explanations. |
| `task_type` | Category label (e.g. dev, qa). |

Dependencies must reference existing `task_id` rows. The graph must be acyclic.

## Run the Streamlit UI

```bash
streamlit run app.py
```

Upload a CSV, set the high-risk threshold, and review tables and charts.

## Run the web UI (FastAPI)

```bash
python -m uvicorn web_app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload a CSV and optionally adjust the risk threshold. The API endpoint `POST /api/analyze` accepts multipart form fields `file` (CSV) and `threshold` (float).

Static assets are served from `/assets` (see `static/`).

## AI model (optional)

The app does **not** train the model at runtime. To produce `ai/model.pkl` and `ai/feature_importance.json`:

```bash
python -m ai.data_generator
python -m ai.train
```

If `ai/model.pkl` is missing, delay probability is omitted and the UI shows a warning.

## Sample data and verification

- **`sample_tasks.csv`**: Example project with parallel branches, a merge, float duration, and a linear tail (eight tasks).
- **Check CPM against expected values**:

```bash
python verify_sample_tasks.py
```

Exit code `0` means checks passed.

## Project layout (short)

| Path | Role |
|------|------|
| `cpm.py` | DAG build, CPM computation, feature frame for ML. |
| `risk.py` | Rule-based risk score. |
| `analysis.py` | Shared pipeline for Streamlit and `web_app`. |
| `ai/` | Training data generator, training, predictor, explanations. |
| `visuals.py` | Plotly figures (Gantt, risk, dependency graph, explanations). |
| `app.py` | Streamlit entrypoint. |
| `web_app.py` | FastAPI entrypoint. |
| `static/` | Web UI HTML, CSS, JavaScript. |
