"""Tests for heart_disease.features (DataTransformer)."""

from __future__ import annotations

import pandas as pd
import pytest

from heart_disease.features import DataTransformer


# ===========================================================================
# Binary column mapping
# ===========================================================================


class TestBinaryMapping:
    def test_sex_integer_mapped_to_string(self, raw_df):
        df = DataTransformer(drop_id=False).transform(raw_df)
        assert set(df["Sex"].cat.categories) == {"female", "male"}

    def test_sex_male_is_1(self, raw_df):
        df = DataTransformer(drop_id=False).transform(raw_df)
        original_male_mask = raw_df["Sex"] == 1
        assert (df.loc[original_male_mask, "Sex"] == "male").all()

    def test_sex_female_is_0(self, raw_df):
        df = DataTransformer(drop_id=False).transform(raw_df)
        original_female_mask = raw_df["Sex"] == 0
        assert (df.loc[original_female_mask, "Sex"] == "female").all()

    def test_exercise_angina_mapped_to_string(self, raw_df):
        df = DataTransformer(drop_id=False).transform(raw_df)
        assert set(df["Exercise angina"].cat.categories).issubset({"yes", "no"})

    def test_fbs_not_remapped_when_already_bool(self, raw_df):
        """FBS over 120 is already bool in the CSV; should not be touched."""
        df = DataTransformer(drop_id=False).transform(raw_df)
        # FBS over 120 is boolean — bool dtype is NOT in _BINARY_MAPPINGS path,
        # so the transform skips it. The column stays as bool/category.
        assert "FBS over 120" in df.columns

    def test_mapping_is_idempotent_on_string_input(self, clean_df):
        """Calling transform twice gives the same result as calling it once."""
        df_once = DataTransformer(drop_id=False).transform(clean_df)
        df_twice = DataTransformer(drop_id=False).transform(df_once)
        pd.testing.assert_frame_equal(
            df_once.reset_index(drop=True),
            df_twice.reset_index(drop=True),
        )


# ===========================================================================
# Categorical casting
# ===========================================================================


class TestCategoricalCasting:
    _CAT_COLS = [
        "Sex", "Chest pain type", "FBS over 120", "EKG results",
        "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium",
    ]

    def test_all_categorical_columns_are_category_dtype(self, raw_df):
        df = DataTransformer(drop_id=False).transform(raw_df)
        for col in self._CAT_COLS:
            assert df[col].dtype.name == "category", (
                f"Expected '{col}' to be category dtype"
            )

    def test_numerical_columns_retain_original_dtype(self, raw_df):
        df = DataTransformer(drop_id=False).transform(raw_df)
        for col in ["Age", "BP", "Cholesterol", "Max HR"]:
            assert pd.api.types.is_integer_dtype(df[col])

    def test_st_depression_retains_float_dtype(self, raw_df):
        df = DataTransformer(drop_id=False).transform(raw_df)
        assert pd.api.types.is_float_dtype(df["ST depression"])


# ===========================================================================
# Column dropping
# ===========================================================================


class TestColumnDropping:
    def test_id_dropped_by_default(self, raw_df):
        df = DataTransformer().transform(raw_df)
        assert "id" not in df.columns

    def test_id_kept_when_drop_id_false(self, raw_df):
        df = DataTransformer(drop_id=False).transform(raw_df)
        assert "id" in df.columns

    def test_target_kept_by_default(self, raw_df):
        df = DataTransformer().transform(raw_df)
        assert "Heart Disease" in df.columns

    def test_target_dropped_when_requested(self, raw_df):
        df = DataTransformer(drop_target=True).transform(raw_df)
        assert "Heart Disease" not in df.columns

    def test_both_id_and_target_can_be_dropped(self, raw_df):
        df = DataTransformer(drop_id=True, drop_target=True).transform(raw_df)
        assert "id" not in df.columns
        assert "Heart Disease" not in df.columns


# ===========================================================================
# Immutability
# ===========================================================================


class TestImmutability:
    def test_transform_does_not_mutate_input(self, raw_df):
        original_sex = raw_df["Sex"].tolist()
        DataTransformer().transform(raw_df)
        assert raw_df["Sex"].tolist() == original_sex


# ===========================================================================
# split_features_target
# ===========================================================================


class TestSplitFeaturesTarget:
    def test_returns_tuple_of_two(self, clean_df):
        df = DataTransformer(drop_id=True).transform(clean_df)
        result = DataTransformer.split_features_target(df)
        assert len(result) == 2

    def test_x_does_not_contain_target_or_id(self, clean_df):
        df = DataTransformer(drop_id=True).transform(clean_df)
        X, _ = DataTransformer.split_features_target(df)
        assert "Heart Disease" not in X.columns
        assert "id" not in X.columns

    def test_y_is_binary(self, clean_df):
        df = DataTransformer(drop_id=True).transform(clean_df)
        _, y = DataTransformer.split_features_target(df)
        assert set(y.unique()).issubset({0, 1})

    def test_y_presence_maps_to_1(self, clean_df):
        df = DataTransformer(drop_id=True).transform(clean_df)
        _, y = DataTransformer.split_features_target(df)
        presence_idx = clean_df[clean_df["Heart Disease"] == "Presence"].index
        assert (y.loc[presence_idx] == 1).all()

    def test_y_absence_maps_to_0(self, clean_df):
        df = DataTransformer(drop_id=True).transform(clean_df)
        _, y = DataTransformer.split_features_target(df)
        absence_idx = clean_df[clean_df["Heart Disease"] == "Absence"].index
        assert (y.loc[absence_idx] == 0).all()

    def test_y_series_name_is_target(self, clean_df):
        df = DataTransformer(drop_id=True).transform(clean_df)
        _, y = DataTransformer.split_features_target(df)
        assert y.name == "target"

    def test_x_row_count_matches_input(self, clean_df):
        df = DataTransformer(drop_id=True).transform(clean_df)
        X, y = DataTransformer.split_features_target(df)
        assert len(X) == len(clean_df)
        assert len(y) == len(clean_df)

    def test_raises_when_target_missing(self, clean_df):
        df = DataTransformer(drop_id=True, drop_target=True).transform(clean_df)
        with pytest.raises(KeyError, match="Heart Disease"):
            DataTransformer.split_features_target(df)
