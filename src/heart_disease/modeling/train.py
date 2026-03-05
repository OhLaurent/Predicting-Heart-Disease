"""Model training pipeline.

Note: this module is a work in progress.
A production-ready training pipeline will be implemented here,
consolidating the logic currently in ``notebooks/02_modelling.ipynb``.
"""

from __future__ import annotations

# TODO: implement production training pipeline
#
# Planned responsibilities:
#   - Load and validate data via dataset.DataLoader / DataValidator
#   - Apply feature engineering via features.DataTransformer
#   - Build sklearn Pipeline (preprocessor + model)
#   - Run cross-validated evaluation and log metrics to MLflow
#   - Persist the best model artefact under models/
