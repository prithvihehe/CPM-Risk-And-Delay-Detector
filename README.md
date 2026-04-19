# CPM Risk & Delay Detector

Professional Streamlit dashboard for construction schedule analysis using:

- **CPM (Critical Path Method)** for deterministic planning
- **PERT schedule** for probabilistic duration estimation (3-point estimates)
- **Resource planning modes**:
  - Resource **Levelling**
  - Resource **Smoothing**
- **AON (Activity-on-Node)** dependency visualization
- **AI-assisted delay risk scoring + explanation**

---

## Key capabilities

### 1) CPM engine

- Validates dependency references and DAG validity
- Computes `ES`, `EF`, `LS`, `LF`, `slack`, `is_critical`
- Shows project completion and ordered critical path

### 2) AON visualization

- Dependency network with directed task links
- Critical activities highlighted
- Duration-aware labels for quick presentation understanding

### 3) PERT schedule

Uses standard expected duration formula:

$$
E = \frac{O + 4M + P}{6}
$$

Where:
- $O$ = optimistic duration
- $M$ = most likely duration
- $P$ = pessimistic duration

Outputs include:
- PERT expected completion
- Critical-path standard deviation ($\sigma$)
- Approximate P80 completion (normal approximation)

### 4) Resource planning toggle

#### Resource Levelling
- Enforces capacity constraints strictly
- May shift tasks and increase completion time
- Good when resource limits are hard constraints

#### Resource Smoothing
- Uses available float/slack where possible
- Reduces resource peaks with minimal schedule disturbance
- Useful for balancing without aggressive delay

Each mode is displayed **separately and clearly** in UI.

### 5) AI delay risk insights

- Rule-based risk score per task
- Optional model-based delay probability (`ai/model.pkl`)
- Human-readable explanation factors using feature importance

---

## Project structure

```text
CPM-Risk-And-Delay-Detector/
├─ app.py
├─ cpm.py
├─ pert.py
├─ risk.py
├─ visuals.py
├─ resource_leveling.py
├─ sample_tasks.csv
├─ requirements.txt
└─ ai/
   ├─ data_generator.py
   ├─ train.py
   ├─ predict.py
   ├─ explain.py
   ├─ training_data.csv
   ├─ model.pkl
   └─ feature_importance.json
```

---

## Input CSV schema

### Required columns

- `task_id`
- `task_name`
- `duration`
- `dependencies` (comma-separated predecessor ids, e.g. `"2,3"`)
- `resource_count`
- `task_type`

### Optional PERT columns

- `optimistic_duration`
- `most_likely_duration`
- `pessimistic_duration`

If optional PERT columns are missing, `duration` is reused.

---

## Quick start (Windows / PowerShell)

```powershell
Set-Location -LiteralPath 'c:\Users\Sumit Dutta\OneDrive\Desktop\New folder\CPM-Risk-And-Delay-Detector'
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

---

## AI model setup (optional)

If model files are missing, app will still run with rule-based scoring.
To regenerate model artifacts:

```powershell
python -m ai.data_generator
python -m ai.train
```

---

## Typical workflow in app

1. Upload task CSV
2. Review CPM and critical path
3. Review AON dependency chart
4. Inspect PERT schedule metrics
5. Select resource planning mode:
   - Resource levelling, or
   - Resource smoothing
6. Check risk table, top risky tasks, and explanation chart

---

## Contribution guide (collaborator flow)

```powershell
git checkout main
git pull origin main
git checkout -b <your-branch>
git add .
git commit -m "feat: ..."
git push -u origin <your-branch>
```

Then open a Pull Request from your branch to `main`.

---

## License

For academic/demo use unless otherwise specified by repository owner.
