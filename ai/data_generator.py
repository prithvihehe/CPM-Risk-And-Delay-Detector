"""
Generate synthetic training data for delay prediction.

Features: duration, resource_count, is_critical, slack
Target: delayed (binary)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["duration", "resource_count", "is_critical", "slack"]
TARGET_COLUMN = "delayed"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_dataset(
    n_samples: int = 5000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Synthetic labels loosely motivated by project risk:
    longer tasks, more resources, critical path, and low slack
    increase the chance of a 'delayed' label (with noise).
    """
    rng = np.random.default_rng(seed)
    duration = rng.uniform(0.5, 20.0, size=n_samples)
    resource_count = rng.integers(1, 8, size=n_samples)
    is_critical = rng.integers(0, 2, size=n_samples)
    slack = rng.exponential(scale=3.0, size=n_samples)

    z = (
        0.12 * (duration - 8.0)
        + 0.15 * (resource_count - 3)
        + 0.45 * is_critical
        - 0.25 * slack
        + rng.normal(0.0, 0.6, size=n_samples)
    )
    p_delay = _sigmoid(z)
    delayed = (rng.random(n_samples) < p_delay).astype(np.int8)

    return pd.DataFrame(
        {
            "duration": duration.astype(float),
            "resource_count": resource_count.astype(int),
            "is_critical": is_critical.astype(int),
            "slack": slack.astype(float),
            "delayed": delayed,
        }
    )


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "training_data.csv"


def main() -> None:
    out = default_output_path()
    df = generate_dataset()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
