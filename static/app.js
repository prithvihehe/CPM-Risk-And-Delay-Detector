function $(id) {
  return document.getElementById(id);
}

function setStatus(msg, isError) {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("error", !!isError);
}

function showPlot(containerId, figure) {
  const el = $(containerId);
  if (!figure || !figure.data) {
    el.textContent = "No chart data.";
    return;
  }
  const config = { responsive: true, displayModeBar: true };
  window.Plotly.purge(el);
  window.Plotly.newPlot(el, figure.data, figure.layout || {}, config);
}

function buildTable(tableId, rows, columns, rowClassForIndex) {
  const table = $(tableId);
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  if (!rows.length) {
    tbody.innerHTML = "<tr><td colspan=\"99\">No rows</td></tr>";
    return;
  }

  const cols = columns || Object.keys(rows[0]);
  const hr = document.createElement("tr");
  cols.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    hr.appendChild(th);
  });
  thead.appendChild(hr);

  rows.forEach((row, i) => {
    const tr = document.createElement("tr");
    const cls = rowClassForIndex && rowClassForIndex(i);
    if (cls) tr.className = cls;
    cols.forEach((c) => {
      const td = document.createElement("td");
      const v = row[c];
      td.textContent = v === null || v === undefined ? "—" : String(v);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function formatWarnings(payload) {
  const box = $("warnings");
  const parts = [];
  (payload.warnings || []).forEach((w) => parts.push(w));
  if (payload.importance_error) parts.push(payload.importance_error);
  if (!parts.length) {
    box.classList.add("hidden");
    box.textContent = "";
    return;
  }
  box.classList.remove("hidden");
  box.innerHTML = parts.map((p) => `<div>${escapeHtml(p)}</div>`).join("");
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = $("file");
  const threshold = parseFloat(String($("threshold").value || "1"));
  if (!fileInput.files || !fileInput.files[0]) {
    setStatus("Choose a CSV file first.", true);
    return;
  }

  const btn = $("submit");
  btn.disabled = true;
  setStatus("Running analysis…");

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("threshold", String(threshold));

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: fd,
    });
    const payload = await res.json();
    if (!res.ok || !payload.ok) {
      const msg =
        payload.message ||
        (payload.messages && payload.messages[0]) ||
        payload.error ||
        "Request failed";
      setStatus(msg, true);
      $("results").classList.add("hidden");
      return;
    }

    setStatus("Done.");
    $("results").classList.remove("hidden");

    $("metric-ef").textContent =
      payload.project_completion != null
        ? Number(payload.project_completion).toFixed(2)
        : "—";
    $("metric-cp").textContent =
      (payload.critical_path && payload.critical_path.length
        ? payload.critical_path.join(" → ")
        : "(none)");

    formatWarnings(payload);

    const charts = payload.charts || {};
    showPlot("chart-dag", charts.dependency);
    showPlot("chart-gantt", charts.gantt);
    showPlot("chart-risk", charts.delay_risk);

    const cpmCols = payload.cpm_table.length ? Object.keys(payload.cpm_table[0]) : [];
    buildTable("table-cpm", payload.cpm_table, cpmCols);

    const scored = payload.scored_table || [];
    const flags = payload.high_risk_flags || [];
    const scoredCols = scored.length ? Object.keys(scored[0]) : [];
    buildTable("table-scored", scored, scoredCols, (i) =>
      flags[i] ? "row-high-risk" : ""
    );

    const explainMap = charts.explanation_by_task || {};
    const insights = payload.task_insights || [];
    $("json-insights").textContent = JSON.stringify(insights, null, 2);

    const sel = $("task-select");
    sel.innerHTML = "";
    insights.forEach((ins) => {
      const opt = document.createElement("option");
      opt.value = ins.task_id;
      opt.textContent = ins.task_id;
      sel.appendChild(opt);
    });

    function renderExplain() {
      const tid = sel.value;
      showPlot("chart-explain", explainMap[tid]);
    }
    sel.onchange = renderExplain;
    renderExplain();
  } catch (err) {
    setStatus(err && err.message ? err.message : String(err), true);
    $("results").classList.add("hidden");
  } finally {
    btn.disabled = false;
  }
});
