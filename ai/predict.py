"""
Load trained RandomForest and predict probability of delay per task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ai.data_generator import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


class DelayPredictor:
    """Loads ai/model.pkl and exposes batch delay probabilities."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"Model not found at {self._model_path}. "
                "Train offline: python -m ai.data_generator && python -m ai.train"
            )
        self._clf = joblib.load(self._model_path)

    def predict_batch(self, feature_rows: pd.DataFrame | list[dict[str, Any]]) -> np.ndarray:
        """
        Return P(delayed=1) for each row.

        feature_rows must provide: duration, resource_count, is_critical, slack
        (is_critical may be bool or int 0/1).
        """
        if isinstance(feature_rows, list):
            df = pd.DataFrame(feature_rows)
        else:
            df = feature_rows.copy()

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[FEATURE_COLUMNS].copy()
        X["is_critical"] = X["is_critical"].astype(float)
        X["duration"] = X["duration"].astype(float)
        X["resource_count"] = X["resource_count"].astype(float)
        X["slack"] = X["slack"].astype(float)

        proba = self._clf.predict_proba(X)
        classes = np.asarray(self._clf.classes_)
        pos_idx = int(np.flatnonzero(classes == 1)[0])
        return proba[:, pos_idx].astype(float)
