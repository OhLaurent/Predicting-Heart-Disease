"""Model inference pipeline.

Note: this module is a work in progress.
A production-ready inference pipeline will be implemented here.
"""

from __future__ import annotations

# TODO: implement production inference pipeline
#
# Planned responsibilities:
#   - Accept a raw DataFrame (validated against schema in inference mode)
#   - Apply the same feature engineering used at training time
#   - Load a serialised model artefact from models/
#   - Return a Series of predicted probabilities / labels
