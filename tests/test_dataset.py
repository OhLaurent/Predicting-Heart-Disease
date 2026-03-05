"""Tests for heart_disease.dataset (DataLoader + DataValidator)."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from heart_disease.dataset import DataLoader, DataValidator


# ===========================================================================
# DataLoader
# ===========================================================================


class TestDataLoader:
    def test_load_returns_dataframe(self, raw_csv):
        df = DataLoader(raw_csv).load()
        assert isinstance(df, pd.DataFrame)

    def test_load_shape(self, raw_csv, raw_df):
        df = DataLoader(raw_csv).load()
        assert df.shape == raw_df.shape

    def test_load_columns(self, raw_csv, raw_df):
        df = DataLoader(raw_csv).load()
        assert list(df.columns) == list(raw_df.columns)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader(tmp_path / "nonexistent.csv").load()

    def test_load_drop_target_removes_column(self, raw_csv):
        df = DataLoader(raw_csv, drop_target=True).load()
        assert "Heart Disease" not in df.columns

    def test_load_keeps_target_by_default(self, raw_csv):
        df = DataLoader(raw_csv).load()
        assert "Heart Disease" in df.columns

    def test_load_drop_target_false_keeps_column(self, raw_csv):
        df = DataLoader(raw_csv, drop_target=False).load()
        assert "Heart Disease" in df.columns

    def test_load_does_not_mutate_file(self, raw_csv):
        """Loading with drop_target=True does not alter the CSV on disk."""
        DataLoader(raw_csv, drop_target=True).load()
        df_reloaded = pd.read_csv(raw_csv)
        assert "Heart Disease" in df_reloaded.columns


# ===========================================================================
# DataValidator
# ===========================================================================


class TestDataValidator:
    # --- mode='inference' guard ---

    def test_inference_mode_raises_if_target_present(self, clean_df):
        with pytest.raises(ValueError, match="Heart Disease"):
            DataValidator(mode="inference").validate(clean_df)

    def test_inference_mode_accepts_df_without_target(self, clean_df):
        df = clean_df.drop(columns=["Heart Disease"])
        result = DataValidator(mode="inference").validate(df)
        assert isinstance(result, pd.DataFrame)

    # --- mode='training' ---

    def test_training_mode_accepts_full_df(self, clean_df):
        result = DataValidator(mode="training").validate(clean_df)
        assert result.shape == clean_df.shape

    def test_training_mode_returns_dataframe(self, clean_df):
        result = DataValidator(mode="training").validate(clean_df)
        assert isinstance(result, pd.DataFrame)

    # --- schema checks ---

    def test_invalid_sex_value_raises(self, clean_df):
        bad = clean_df.copy()
        bad.loc[0, "Sex"] = "other"
        with pytest.raises(pa.errors.SchemaError):
            DataValidator(mode="training").validate(bad)

    def test_out_of_range_age_raises(self, clean_df):
        bad = clean_df.copy()
        bad.loc[0, "Age"] = 200  # exceeds max_value=120
        with pytest.raises(pa.errors.SchemaError):
            DataValidator(mode="training").validate(bad)

    def test_invalid_chest_pain_type_raises(self, clean_df):
        bad = clean_df.copy()
        bad.loc[0, "Chest pain type"] = 9
        with pytest.raises(pa.errors.SchemaError):
            DataValidator(mode="training").validate(bad)

    def test_invalid_thallium_value_raises(self, clean_df):
        bad = clean_df.copy()
        bad.loc[0, "Thallium"] = 5  # only 3, 6, 7 are valid
        with pytest.raises(pa.errors.SchemaError):
            DataValidator(mode="training").validate(bad)

    def test_invalid_exercise_angina_raises(self, clean_df):
        bad = clean_df.copy()
        bad.loc[0, "Exercise angina"] = "maybe"
        with pytest.raises(pa.errors.SchemaError):
            DataValidator(mode="training").validate(bad)

    def test_negative_st_depression_raises(self, clean_df):
        bad = clean_df.copy()
        bad.loc[0, "ST depression"] = -0.1
        with pytest.raises(pa.errors.SchemaError):
            DataValidator(mode="training").validate(bad)

    # --- schema file resolution ---

    def test_missing_schema_file_raises(self, clean_df, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataValidator(schema_path=tmp_path / "missing.yaml", mode="training").validate(clean_df)

    # --- schema caching ---

    def test_schema_is_cached_after_first_call(self, clean_df):
        validator = DataValidator(mode="training")
        validator.validate(clean_df)
        schema_first = validator._schema
        validator.validate(clean_df)
        assert validator._schema is schema_first
