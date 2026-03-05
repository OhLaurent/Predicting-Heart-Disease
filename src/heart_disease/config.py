"""Project-wide paths and constants.

All other modules should import paths from here rather than
constructing them inline, so that the project works regardless of
the current working directory.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root directories
# ---------------------------------------------------------------------------

# Repository root — two levels up from this file (src/heart_disease/config.py)
REPO_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_INTERIM_DIR: Path = DATA_DIR / "interim"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
DATA_EXTERNAL_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = REPO_DIR / "models"

REPORTS_DIR: Path = REPO_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

CONFIG_DIR: Path = REPO_DIR / "config"
SCHEMA_PATH: Path = CONFIG_DIR / "schema.yaml"

NOTEBOOKS_DIR: Path = REPO_DIR / "notebooks"

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

ID_COLUMN: str = "id"
TARGET_COLUMN: str = "Heart Disease"

CATEGORICAL_COLUMNS: list[str] = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

NUMERICAL_COLUMNS: list[str] = [
    "Age",
    "BP",
    "Cholesterol",
    "Max HR",
    "ST depression",
]

# Raw integer → cleaned string label mappings (applied in dataset.py)
BINARY_MAPPINGS: dict[str, dict[int, str]] = {
    "Sex": {1: "male", 0: "female"},
    "FBS over 120": {1: "true", 0: "false"},
    "Exercise angina": {1: "yes", 0: "no"},
}

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_SPLITS: int = 10
