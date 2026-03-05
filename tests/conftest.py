"""Shared pytest fixtures for heart_disease tests."""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal DataFrames with all 15 schema columns
# ---------------------------------------------------------------------------

# Raw data — integer-coded (as it comes straight from the CSV before transform)
_RAW_ROWS = [
    {
        "id": 0, "Age": 55, "Sex": 1, "Chest pain type": 4,
        "BP": 130, "Cholesterol": 250, "FBS over 120": True,
        "EKG results": 0, "Max HR": 170, "Exercise angina": 0,
        "ST depression": 1.4, "Slope of ST": 2,
        "Number of vessels fluro": 0, "Thallium": 3,
        "Heart Disease": "Presence",
    },
    {
        "id": 1, "Age": 45, "Sex": 0, "Chest pain type": 2,
        "BP": 120, "Cholesterol": 201, "FBS over 120": False,
        "EKG results": 1, "Max HR": 150, "Exercise angina": 1,
        "ST depression": 0.0, "Slope of ST": 1,
        "Number of vessels fluro": 1, "Thallium": 7,
        "Heart Disease": "Absence",
    },
    {
        "id": 2, "Age": 62, "Sex": 1, "Chest pain type": 3,
        "BP": 160, "Cholesterol": 340, "FBS over 120": False,
        "EKG results": 2, "Max HR": 90, "Exercise angina": 0,
        "ST depression": 3.1, "Slope of ST": 3,
        "Number of vessels fluro": 2, "Thallium": 6,
        "Heart Disease": "Presence",
    },
]

# Cleaned data — string-encoded (as produced by DataTransformer)
_CLEAN_ROWS = [
    {
        "id": 0, "Age": 55, "Sex": "male", "Chest pain type": 4,
        "BP": 130, "Cholesterol": 250, "FBS over 120": True,
        "EKG results": 0, "Max HR": 170, "Exercise angina": "no",
        "ST depression": 1.4, "Slope of ST": 2,
        "Number of vessels fluro": 0, "Thallium": 3,
        "Heart Disease": "Presence",
    },
    {
        "id": 1, "Age": 45, "Sex": "female", "Chest pain type": 2,
        "BP": 120, "Cholesterol": 201, "FBS over 120": False,
        "EKG results": 1, "Max HR": 150, "Exercise angina": "yes",
        "ST depression": 0.0, "Slope of ST": 1,
        "Number of vessels fluro": 1, "Thallium": 7,
        "Heart Disease": "Absence",
    },
    {
        "id": 2, "Age": 62, "Sex": "male", "Chest pain type": 3,
        "BP": 160, "Cholesterol": 340, "FBS over 120": False,
        "EKG results": 2, "Max HR": 90, "Exercise angina": "no",
        "ST depression": 3.1, "Slope of ST": 3,
        "Number of vessels fluro": 2, "Thallium": 6,
        "Heart Disease": "Presence",
    },
]


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Raw DataFrame with integer-coded binary columns."""
    return pd.DataFrame(_RAW_ROWS)


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    """Cleaned DataFrame with string-encoded binary columns."""
    return pd.DataFrame(_CLEAN_ROWS)


@pytest.fixture()
def raw_csv(tmp_path, raw_df) -> "Path":
    """Write ``raw_df`` to a temporary CSV file and return its path."""
    path = tmp_path / "heart_disease.csv"
    raw_df.to_csv(path, index=False)
    return path
