"""
FastAPI server: serves the static web UI and POST /api/analyze for CPM + charts + probabilities.
Run: uvicorn web_app:app --reload
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analysis import analysis_to_api_payload, error_payload, run_analysis

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="CPM Risk & Delay")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.post("/api/analyze")
async def api_analyze(
    file: UploadFile = File(...),
    threshold: float = Form(1.0),
) -> JSONResponse:
    raw = await file.read()
    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": "csv_read", "message": str(e)},
            status_code=400,
        )

    result = run_analysis(df, risk_threshold=threshold)
    if not result.ok:
        return JSONResponse(error_payload(result), status_code=400)
    return JSONResponse(analysis_to_api_payload(result))
