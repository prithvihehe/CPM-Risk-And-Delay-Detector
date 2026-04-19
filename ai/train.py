"""
Train RandomForestClassifier on ai/training_data.csv and save ai/model.pkl.

Run after: python -m ai.data_generator
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ai.data_generator import FEATURE_COLUMNS, TARGET_COLUMN, default_output_path

MODEL_FILENAME = "model.pkl"
IMPORTANCE_FILENAME = "feature_importance.json"


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / MODEL_FILENAME


def default_importance_path() -> Path:
    return Path(__file__).resolve().parent / IMPORTANCE_FILENAME


def train_and_save(
    csv_path: Path | None = None,
    model_path: Path | None = None,
    importance_path: Path | None = None,
    random_state: int = 42,
) -> Path:
    csv_path = csv_path or default_output_path()
    model_path = model_path or default_model_path()
    importance_path = importance_path or default_importance_path()

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Generate it first: python -m ai.data_generator"
        )

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    imp: dict[str, float] = {
        FEATURE_COLUMNS[i]: float(clf.feature_importances_[i])
        for i in range(len(FEATURE_COLUMNS))
    }
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(imp, f, indent=2)
    print(f"Saved feature importances to {importance_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")
    return model_path


def main() -> None:
    train_and_save()


if __name__ == "__main__":
    main()
